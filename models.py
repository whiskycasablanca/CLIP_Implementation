from vit import VissionTransformer # Importing ViT from previous implementaton (GitHub: Ml-Models)
import numpy as np
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ViT = VissionTransformer( 
            num_layers=8,
            img_size=224,
            emb_size=768,
            patch_size=16,
            num_head=6,
            num_class=False).to(self.device)
        self.image_encoder = ImageEncoder(base_model=ViT, embed_dim=768, proj_dim=256)
        self.text_encoder = TextEncoder(embed_dim=768, proj_dim=256)

        self.temperature = nn.Parameter(torch.ones([])*np.log(1/7)).to(self.device)

    def forward(self, x):
        I_t = self.image_encoder(x["image"])
        T_t = self.text_encoder(x["input_ids"], x["mask"])

        logits = I_t@T_t.T * torch.exp(self.temperature)

        labels = torch.arange(I_t.size(0)).to(self.device)

        loss_I = F.cross_entropy(logits.T, labels)
        loss_T = F.cross_entropy(logits, labels)

        loss = (loss_I + loss_T)/2.0 

        return loss, logits