
import os

os.sys.path.append(os.path.dirname(os.curdir))
import torch.nn as nn
import torch
from ViT.transformer import TransformerEncoder, TransformerEncoderLayer
import math

class ViT(nn.Module):
    def __init__(self, 
                 image_size=224, 
                 patch_size=32, 
                 num_classes=1000, 
                 dim=768, 
                 depth=12, 
                 heads=8, 
                 pool='cls', 
                 channels=3, 
                 dropout=0.1, 
                 ):
        
        
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        # print(num_patches)
        self.patches = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embeddings = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = self.create_sinusoidal_embeddings(num_patches + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(dim=dim, nhead=heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, L=depth)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def create_sinusoidal_embeddings(self, num_embeddings, embedding_dim):
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        embeddings = torch.zeros(num_embeddings, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(embeddings.unsqueeze(0), requires_grad=False)

    def forward(self, img):
        x = self.patches(img)
        x = x.flatten(2).transpose(1, 2)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == "__main__":
    model = ViT()
    model.eval()

    # Create a dummy input tensor with the shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    print("Input shape:", dummy_input.shape)

    # Perform inference
    with torch.no_grad():
        output = model(dummy_input)

    print("Output shape:", output.shape) # should be (1, 1000)
    output = nn.Softmax(dim=1)(output)
    print("Sum:", output.sum()) # should be 1.0





