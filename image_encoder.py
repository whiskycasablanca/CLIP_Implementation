import torch
import torch.nn as nn

#mlp헤드 떼어낸 vit가져와서 선형변환하기 768=>256
class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = base_model

        for param in self.model.parameters():
            param.requires_grad = False #freezing 시키기

        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.projection(self.model(x))
        return self.layer_norm(x)