import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configs
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 28
batch_size = 128
T = 1000
lr = 1e-4
epochs = 5

# -----------------------------
# Noise schedule
# -----------------------------
betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)

betas = torch.tensor(betas, dtype=torch.float32).to(device)
alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32).to(device)
sqrt_alphas_cumprod = torch.tensor(sqrt_alphas_cumprod, dtype=torch.float32).to(device)
sqrt_one_minus_alphas_cumprod = torch.tensor(sqrt_one_minus_alphas_cumprod, dtype=torch.float32).to(device)

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2. - 1.)
])

dataloader = DataLoader(
    datasets.MNIST(root='.', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# -----------------------------
# Model
# -----------------------------
class SimpleDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Positional embedding (optional, simple time embedding)
        t = t[:, None, None, None].float() / T
        t_embed = t.expand_as(x)
        return self.net(x + t_embed)

model = SimpleDDPM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Forward diffusion helper
# -----------------------------
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

# -----------------------------
# Training
# -----------------------------
print("🧨 Starting training...")
model.train()
for epoch in range(epochs):
    pbar = tqdm(dataloader)
    for x, _ in pbar:
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device).long()
        noise = torch.randn_like(x)
        x_t = q_sample(x, t, noise)
        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# -----------------------------
# Sampling (reverse process)
# -----------------------------
@torch.no_grad()
def p_sample(x, t):
    beta_t = betas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_recip_alpha = (1. / torch.sqrt(alphas[t])).view(-1, 1, 1, 1)

    noise_pred = model(x, t)
    model_mean = sqrt_recip_alpha * (x - beta_t * noise_pred / sqrt_one_minus_alpha)

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        sigma = torch.sqrt(betas[t]).view(-1, 1, 1, 1)
        return model_mean + sigma * noise

@torch.no_grad()
def sample_ddpm(n_samples):
    x = torch.randn(n_samples, 1, image_size, image_size).to(device)
    for t_ in reversed(range(T)):
        t = torch.full((n_samples,), t_, device=device, dtype=torch.long)
        x = p_sample(x, t)
    return x

# -----------------------------
# Generate and show samples
# -----------------------------
samples = sample_ddpm(16)
samples = samples.cpu().clamp(-1, 1) * 0.5 + 0.5  # Rescale to [0, 1]

grid = torch.cat([s for s in samples], dim=2).squeeze()
plt.imshow(grid.numpy(), cmap='gray')
plt.axis('off')
plt.title("DDPM Samples on MNIST")
plt.show()
