import os
import re
import math
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from dataset import train_loader, val_loader  
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import MiniLMEfficientNetModel
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def smape(y_true, y_pred, eps=1e-6):
    denom = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape_val = torch.mean(torch.abs(y_pred - y_true) / (denom + eps)) * 100
    return smape_val

def find_latest_checkpoint(output_dir):
    ckpts = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    return os.path.join(output_dir, ckpts[0])

def train_epoch(model, dataloader, optimizer, device, criterion, scaler, accum_steps, scheduler):
    model.train()
    total_loss, total_smape, n = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        prices = batch["price"].to(device)

        with torch.amp.autocast('cuda'):
            preds = model(input_ids, attention_mask, images)
            loss = criterion(preds, prices) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()

        batch_smape = smape(prices, preds.detach()).item()
        total_loss += loss.item() * accum_steps * prices.size(0)
        total_smape += batch_smape * prices.size(0)
        n += prices.size(0)
        pbar.set_postfix_str(f"Loss={loss.item() * accum_steps:.4f} | SMAPE={batch_smape:.2f}%")

    return total_loss / n, total_smape / n

def eval_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_smape, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            prices = batch["price"].to(device)

            with torch.amp.autocast('cuda'):
                preds = model(input_ids, attention_mask, images)
                loss = criterion(preds, prices)

            total_loss += loss.item() * prices.size(0)
            total_smape += smape(prices, preds).item() * prices.size(0)
            n += prices.size(0)
    return total_loss / n, total_smape / n

def main(train_loader, val_loader, epochs=15, accum_steps=4,
         device=None, output_dir="checkpoints_minilm_effnet"):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    latest_ckpt = find_latest_checkpoint(output_dir)
    model = MiniLMEfficientNetModel().to(device)

    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        best_smape = float(re.findall(r"([0-9]+\.[0-9]+)", latest_ckpt)[-1]) if re.findall(r"([0-9]+\.[0-9]+)", latest_ckpt) else 21.04
    else:
        print("Starting Full Fine-tuning (All Layers Unfrozen)")
        best_smape = float('inf')

    for p in model.parameters():
        p.requires_grad = True
    print("All layers unfrozen for fine-tuning.")

    criterion = nn.SmoothL1Loss(beta=0.5)
    scaler = torch.amp.GradScaler('cuda')

    optimizer = AdamW([
                            {'params': model.text_encoder.parameters(), 'lr': 1e-6},
                            {'params': model.image_encoder.parameters(), 'lr': 2e-6},
                            {'params': list(model.text_proj.parameters()) +
                                        list(model.img_proj.parameters()) +
                                        list(model.head.parameters()), 'lr': 7e-6}
                        ], weight_decay=1e-6)


    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-7)


    for epoch in range(1, epochs + 1):
        print(f"\n==== Epoch {epoch}/{epochs} (Full Fine-tuning) ====")
        train_loss, train_smape = train_epoch(model, train_loader, optimizer, device, criterion, scaler, accum_steps, scheduler)
        val_loss, val_smape = eval_epoch(model, val_loader, device, criterion)
        print(f"Train Loss: {train_loss:.4f} | Train SMAPE: {train_smape:.2f}% | Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.2f}%")

        if val_smape < best_smape:
            best_smape = val_smape
            ckpt_path = os.path.join(output_dir, f"best_fullunfreeze_{best_smape:.2f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best model saved: {ckpt_path}")

    print(f"\nTraining Complete! Best Validation SMAPE: {best_smape:.2f}%")

if __name__ == "__main__":
    main(train_loader=train_loader, val_loader=val_loader, epochs=20, accum_steps=4)
