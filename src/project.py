import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from preprocess_data import load_data
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SHAPE = (28, 28, 1)
EPOCHS = 10

T = 200
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1 - betas
alphas_cumprod = torch.tensor(np.cumprod(alphas), dtype=torch.float32).to(device)



class simpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()


    def forward(self, x, t):
        t = t[:, None, None, None].float() / T
        x = x + t
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)

        return x
    

def q_sample(x0, t, noise):
    a_bar = alphas_cumprod[t]
    a_bar = a_bar.reshape((-1,1,1,1))
    out = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise 
    return out 


def main():
    train_data, test_data = load_data()
    net = simpleUNet().to(device=device)
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

    torch.save(net.state_dict(), 'diffusion_model.pth')


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
    x = torch.randn(1, 28, 28).to(device)
    
    for i in reversed(range(T)):
        t = torch.tensor([i]).to(device)
        x = p_sample(x, t)
        print(i)

    return x
        

if __name__ == '__main__':
    # main()

    model = simpleUNet().to(device=device)
    model.load_state_dict(torch.load('diffusion_model.pth'))
    
    x = create_image()
    # x = x.cpu().squeeze().permute(1, 2, 0).numpy()
    x = x.cpu().squeeze()
    x = (x + 1) / 2  #Denormalize
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.show()
