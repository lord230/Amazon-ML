import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

class MiniLMEfficientNetModel(nn.Module):
    def __init__(self,
                 text_model_name='nreimers/MiniLM-L6-H384-uncased'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_emb_dim = 384

        
        effnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        effnet.classifier = nn.Identity()
        image_emb_dim = 1280
        self.image_encoder = effnet

        
        self.text_proj = nn.Linear(text_emb_dim, 256)
        self.img_proj = nn.Linear(image_emb_dim, 256)

        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, images):
    
        text_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.mean(1)
        text_feat = self.text_proj(text_out)

        
        img_feat = self.img_proj(self.image_encoder(images))

     
        fused = torch.cat([text_feat, img_feat], dim=1)
        out = self.head(fused)
        return out.squeeze(1)
    