import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from preprocess_data import load_data
import torch.nn as nn
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SHAPE = (28, 28, 1)
EPOCHS = 10

T = 200
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1 - betas
alphas_cumprod = torch.tensor(np.cumprod(alphas), dtype=torch.float32).to(device)



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class UNet(nn.Module):
    def __init__(self, time_emb_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        #downsampling
        self.conv1_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_norm1 = nn.GroupNorm(4, 32)
        self.conv1_norm2 = nn.GroupNorm(4, 32)
        self.conv1_skip = nn.Conv2d(1, 32, 1)

        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)

        self.conv2_conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_norm1 = nn.GroupNorm(4, 64)
        self.conv2_norm2 = nn.GroupNorm(4, 64)
        self.conv2_skip = nn.Conv2d(32, 64, 1)

        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)

        #bottleneck
        self.bottleneck_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bottleneck_norm1 = nn.GroupNorm(4, 64)
        self.bottleneck_norm2 = nn.GroupNorm(4, 64)

        #Upsampling
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)

        self.conv3_conv1 = nn.Conv2d(128, 32, 3, padding=1)
        self.conv3_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_norm1 = nn.GroupNorm(4, 32)
        self.conv3_norm2 = nn.GroupNorm(4, 32)
        self.conv3_skip = nn.Conv2d(128, 32, 1)

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.conv4_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4_norm1 = nn.GroupNorm(4, 32)
        self.conv4_norm2 = nn.GroupNorm(4, 32)
        self.conv4_skip = nn.Conv2d(64, 32, 1)

        self.out = nn.Conv2d(32, 1, 1)
        self.activation = nn.ReLU()

    

    def forward(self, x, t_emb):
        t_emb = t_emb.float()
        t_emb = self.time_mlp(t_emb)

        #ResBlock 1
        h1 = self.activation(self.conv1_norm1(self.conv1_conv1(x)))
        t1 = self.activation(self.conv1_skip(t_emb))[:, :, None, None]
        h1 = h1 + t1
        h1 = self.activation(self.conv1_norm2(self.conv1_conv2(h1)))
        x1 = h1 + self.conv1_skip(x)

        #resBlock 2
        d1 = self.down1(x1)
        h2 = self.activation(self.conv2_norm1(self.conv2_conv1(d1)))
        t2 = self.activation(self.conv2_skip(t_emb))[:, :, None, None]
        h2 = h2 + t2
        h2 = self.activation(self.conv2_norm2(self.conv2_conv2(h2)))
        x2 = h2 + self.conv2_skip(d1)

        #bottleneck
        d2 = self.down2(x2)
        h3 = self.activation(self.bottleneck_norm1(self.bottleneck_conv1(d2)))
        t3 = self.activation(t_emb)[:, :, None, None]
        h3 = h3 + t3
        h3 = self.activation(self.bottleneck_norm2(self.bottleneck_conv2(h3)))
        x3 = h3 + d2  # skip in bottleneck is identity

        # ResBlock 3
        u1 = self.up1(x3)
        cat1 = torch.cat([u1, x2], dim=1)
        h4 = self.activation(self.conv3_norm1(self.conv3_conv1(cat1)))
        t4 = self.activation(self.conv3_skip(t_emb))[:, :, None, None]
        h4 = h4 + t4
        h4 = self.activation(self.conv3_norm2(self.conv3_conv2(h4)))
        x4 = h4 + self.conv3_skip(cat1)

        #ResBlock4
        u2 = self.up2(x4)
        cat2 = torch.cat([u2, x1], dim=1)
        h5 = self.activation(self.conv4_norm1(self.conv4_conv1(cat2)))
        t5 = self.activation(self.conv4_skip(t_emb))[:, :, None, None]
        h5 = h5 + t5
        h5 = self.activation(self.conv4_norm2(self.conv4_conv2(h5)))
        x5 = h5 + self.conv4_skip(cat2)

        return self.out(x5)
    