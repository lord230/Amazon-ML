import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import DistilBertTokenizer
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoTokenizer

def download_images(df, image_folder="data/images"):
    os.makedirs(image_folder, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        img_path = os.path.join(image_folder, f"{row['sample_id']}.jpg")
        if os.path.exists(img_path):
            continue
        try:
            response = requests.get(row['image_link'], timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image.save(img_path)
        except Exception as e:
            print(f"Failed to download {row['image_link']}: {e}")

    df["image_path"] = df["sample_id"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))
    return df

class ProductPriceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=None, is_train=True, image_folder="data/images"):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_train = is_train
        self.image_folder = image_folder
        os.makedirs(self.image_folder, exist_ok=True)
        self.df["catalog_content"] = self.df["catalog_content"].fillna("")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        image_path = os.path.join(self.image_folder, f"{sample_id}.jpg")

        # ----- TEXT -----
        encoding = self.tokenizer(
            row["catalog_content"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ----- IMAGE -----
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                response = requests.get(row["image_link"], timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(image_path)
        except Exception as e:
            print(f"Skipping image for {sample_id}: {e}")
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        
        if self.is_train and "price" in self.df.columns:
            price = torch.tensor(np.log1p(row["price"]), dtype=torch.float)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image": image,
                "price": price
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image": image,
                "sample_id": sample_id
            }

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_df = pd.read_csv("data/train.csv")

train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

train_dataset = ProductPriceDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
    transform=image_transform,
    is_train=True
)
val_dataset = ProductPriceDataset(
    dataframe=val_df,
    tokenizer=tokenizer,
    transform=image_transform,
    is_train=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for batch in train_loader:
    print(batch['input_ids'].shape)
    print(batch['image'].shape)
    print(batch['price'].shape)
    break
