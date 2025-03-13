
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision.models import vit_b_16, ViT_B_16_Weights
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class CLIPModel(nn.Module):
    def __init__(self):     #생성시 입력 없음
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 사전 학습된 ViT-B/16 모델 불러오기
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # 분류 헤드(MLP head)를 Identity로 바꿔서 특징 추출만 하도록 함
        vit.heads = nn.Identity()

        self.image_encoder = ImageEncoder(base_model=vit, embed_dim=768, proj_dim=256)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256)
        self.temperature = nn.Parameter(torch.ones([])*np.log(1/7)).to(self.device)

    def forward(self, x, return_logits=False):   #인스턴스 이용할때 (이미지, input_ids, mask)를 key로 하는 딕셔너리를 입력으로 받아 loss계산함
        I_t = self.image_encoder(x["image"])
        T_t = self.text_encoder(x["input_ids"], x["mask"])

        logits = I_t@T_t.T * torch.exp(self.temperature)

        labels = torch.arange(I_t.size(0)).to(self.device)

        loss_I = F.cross_entropy(logits.T, labels)
        loss_T = F.cross_entropy(logits, labels)

        loss = (loss_I + loss_T)/2.0 
        if not return_logits:
            return loss #, logits => 필요시 이것도 리턴하자
        else:
            return loss, logits