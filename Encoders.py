import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import vit_b_16, ViT_B_16_Weights

#mlp헤드 떼어낸 vit가져와서 선형변환하기 768=>256
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()

        # Model Parameter Freeze
        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.projection(x)
        x = self.layer_norm(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):    #인스턴스화 할 떄 입력차원과 출력차원을 입력으로 받음
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Model Parameter Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, attention_mask):   #인스턴스를 사용할 때, 토크나이저의 리턴값을 입력으로 받아서 256차원의 임베딩을 출력함
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :] # B, T[cls], E   #cls토큰 뽑아내기   
        x = self.projection(x)  #선형변환하기
        x = self.layer_norm(x)

        return x

