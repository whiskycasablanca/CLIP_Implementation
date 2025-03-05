class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.model = DistilBertModel(config=DistilBertConfig())
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = x[:, 0, :] # B, T[cls], E
        
        x = self.projection(x)

        return self.layer_norm(x)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
texts = ["This is a sample sentence.", "This is another example."]
inputs= tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device) 

encoder = ImageEncoder(embed_dim=768, proj_dim=256)
inputs = encoder(inputs['input_ids'], inputs['mask'])