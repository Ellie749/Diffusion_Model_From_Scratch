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


# Sinusoidal timestep embedding
def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.activation = nn.ReLU()

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.activation(self.norm1(self.conv1(x)))
        time_emb = self.activation(self.time_proj(t_emb))[:, :, None, None]
        h = h + time_emb
        h = self.activation(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        #downsampling
        self.conv1 = ResBlock(1, 32, time_emb_dim)
        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv2 = ResBlock(32, 64, time_emb_dim)
        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)

        # bottleneck
        self.bottleneck = ResBlock(64, 64, time_emb_dim)

        #upsampling
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv3 = ResBlock(64 + 64, 32, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.conv4 = ResBlock(32 + 32, 32, time_emb_dim)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Down
        x1 = self.conv1(x, t_emb)
        x2 = self.conv2(self.down1(x1), t_emb)
        x3 = self.bottleneck(self.down2(x2), t_emb)

        # Up
        x = self.conv3(torch.cat([self.up1(x3), x2], dim=1), t_emb)
        x = self.conv4(torch.cat([self.up2(x), x1], dim=1), t_emb)
        return self.out(x)

    

def q_sample(x0, t, noise):
    a_bar = alphas_cumprod[t]
    a_bar = a_bar.reshape((-1,1,1,1))
    out = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise 
    return out 


def main():
    train_data, test_data = load_data()
    net = UNet().to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    #train
    for epoch in range(EPOCHS):
        pbar = tqdm(train_data)
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn((x.size(0), 1, 28, 28), device=device)

            x_t = q_sample(x, t, noise)
            noise_pred = net(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(net.state_dict(), 'diffusion_model_fine_tuned.pth')


@torch.no_grad()
def p_sample(x, t):
    noise_pred = model(x, t)
    beta_t = torch.tensor(betas[t])
    sqrt_reverse_alpha = 1 / torch.sqrt(torch.tensor(alphas[t]))
    radical_one_minus_alpha_bar_t = torch.sqrt(1 - torch.tensor(alphas_cumprod[t]))
        
    sigma = torch.sqrt(torch.tensor(betas[t])).to(device)
    noise = torch.randn_like(x, device=device)
    image = sqrt_reverse_alpha * (x - ((beta_t*noise_pred)/radical_one_minus_alpha_bar_t)) + sigma*noise

    return image


@torch.no_grad()
def create_image():
    x = torch.randn(1, 1, 28, 28).to(device)
    
    for i in reversed(range(T)):
        t = torch.tensor([i]).to(device)
        x = p_sample(x, t)
        print(i)

    return x
        

if __name__ == '__main__':
    # main()

    model = UNet().to(device=device)
    model.load_state_dict(torch.load('diffusion_model_fine_tuned.pth'))
    
    x = create_image()
    # x = x.cpu().squeeze().permute(1, 2, 0).numpy()
    x = x.cpu().squeeze()
    x = (x + 1) / 2  #Denormalize
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()
