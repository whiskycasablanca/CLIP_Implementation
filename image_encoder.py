class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = base_model

        for param in self.model.parameters():
            param.requires_grad = True

        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.projection(self.model(x))
        return self.layer_norm(x)