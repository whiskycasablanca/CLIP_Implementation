
import torch
from torch import nn
from Encoders import ImageEncoder
from Encoders import TextEncoder
import numpy as np
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, return_logits = False):
        super().__init__()
        self.return_logits = return_logits
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_encoder = ImageEncoder(embed_dim=768, proj_dim=256).to(self.device)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256).to(self.device)
        self.temperature = nn.Parameter(torch.ones([])*np.log(1/7)).to(self.device)

    def forward(self, x):   #인스턴스 이용할때 (이미지, input_ids, mask)를 key로 하는 딕셔너리를 입력으로 받아 loss계산함
        image_embedding = self.image_encoder(x["image"])
        text_embedding = self.text_encoder(x["input_ids"], x["mask"])

        logits = image_embedding@text_embedding.T * torch.exp(self.temperature)

        labels = torch.arange(image_embedding.size(0)).to(self.device)

        image_loss = F.cross_entropy(logits.T, labels, reduction='mean')
        text_loss = F.cross_entropy(logits, labels, reduction='mean')

        loss = (image_loss + text_loss)/2.0 

        return loss if self.return_logits == False else (loss, logits)