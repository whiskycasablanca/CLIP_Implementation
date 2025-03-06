import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer  # 필요한 임포트 추가

class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):    #인스턴스화 할 떄 입력차원과 출력차원을 입력으로 받음
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, attention_mask):   #인스턴스를 사용할 때, 토크나이저의 리턴값을 입력으로 받아서 256차원의 임베딩을 출력함
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x[:, 0, :] # B, T[cls], E   #cls토큰 뽑아내기   
        x = self.projection(x)  #선형변환하기

        return self.layer_norm(x)

