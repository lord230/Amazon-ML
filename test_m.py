import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from io import BytesIO
import requests
import numpy as np
from tqdm import tqdm
from model import MiniLMEfficientNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHECKPOINT_PATH = "best_fullunfreeze_20.68.pt"  
DATASET_FOLDER = "dataset"
IMAGE_FOLDER = os.path.join("data/images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

TEST_CSV = os.path.join(DATASET_FOLDER, "test.csv")
OUTPUT_CSV = os.path.join(DATASET_FOLDER, "test_out_1.csv")

tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print(f"Loading model checkpoint: {CHECKPOINT_PATH}")
model = MiniLMEfficientNetModel().to(device)

import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print("Model loaded successfully.\n")


def download_images(df, image_folder=IMAGE_FOLDER):
    print(f"Downloading {len(df)} images to '{image_folder}'...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        img_path = os.path.join(image_folder, f"{row['sample_id']}.jpg")
        if os.path.exists(img_path):
            continue  

        try:
            response = requests.get(row["image_link"], timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(img_path)
        except Exception as e:
            print(f"Failed to download {row['sample_id']}: {e}")
            Image.new("RGB", (224, 224), (128, 128, 128)).save(img_path)

    print("All images downloaded.\n")
    return df


def predictor(sample_id, catalog_content, image_path):
    try:
        encoding = tokenizer(
            catalog_content or "",
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        image = image_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            try:
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(input_ids, attention_mask, image)
            except TypeError:
                with torch.amp.autocast('cuda'):
                    pred = model(input_ids, attention_mask, image)

        pred = torch.relu(pred).item()
        price = np.expm1(pred)
        return round(float(price), 2)

    except Exception as e:
        print(f"Prediction failed for {sample_id}: {e}")
        return 0.0


if __name__ == "__main__":
    print(f"Loading test CSV: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Total test samples: {len(test_df)}\n")


    test_df = download_images(test_df)


    predictions = []
    print("Running inference on downloaded images...\n")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        sample_id = row["sample_id"]
        catalog_content = row["catalog_content"]
        img_path = os.path.join(IMAGE_FOLDER, f"{sample_id}.jpg")

        price = predictor(sample_id, catalog_content, img_path)
        predictions.append(price)


    test_df["predicted_price"] = predictions
    test_df[["sample_id", "predicted_price"]].to_csv(OUTPUT_CSV, index=False)

    print(f"Predictions saved to: {OUTPUT_CSV}")
    print(test_df.head())
