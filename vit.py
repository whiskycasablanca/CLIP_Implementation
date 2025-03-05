import torch
from torch import nn

class VissionTransformer(nn.Module):
    def __init__(self, num_layers, img_size, emb_size, patch_size, num_head, num_class=False):
        super().__init__()
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.num_head = num_head

        self.proj = nn.Conv2d(in_channels=3, out_channels=emb_size,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_cls_token = nn.Identity()
        if num_class:
            self.mlp_head = nn.Linear(emb_size, num_class)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # Patch embedding
        x = x.flatten(2).transpose(1, 2)  # Reshape to (B, num_patches, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])

        return x