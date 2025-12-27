"""
© Ashkan M., NTNU
MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import json
import os

class SingleFolderDataset(Dataset):
    def __init__(self, root, transform=None, img_lim=None):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_lim is not None:
            self.paths = self.paths[:img_lim]
        self.transform = transform
        self.loader = default_loader
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = self.loader(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img
    
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
        super(Encoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size
        
        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_4, padding=0)
        
        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, x):
        
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)
        
        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y+x
        
        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y+x
        
        y = self.proj(y)
        return y

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(2), x.size(3))
    
    def retrieve_random_codebook(self, random_indices):
        B,W,H = random_indices.shape
        random_indices = random_indices.reshape(-1) # flatten incices
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.view(B, quantized.size(1), W, H)
        # quantized = quantized.transpose(1, 3)
        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity

class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
        
        self.in_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        
        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
        
    def forward(self, x):

        x = self.in_proj(x)
        
        y = self.residual_conv_1(x)
        y = y+x
        x = F.relu(y)
        
        y = self.residual_conv_2(x)
        y = y+x
        y = F.relu(y)
        
        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)
        
        return y
    
class Model(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
                
    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)
        
        return x_hat, commitment_loss, codebook_loss, perplexity
    

def plot_sample_image(x, x_hat, save_img=False):
    fig, axes = plt.subplots(2, 1, figsize=(14, 4))
    axes[0].imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[0].set_ylabel( "GT images", fontsize=10, fontweight="bold", rotation=90, labelpad=20 )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(np.transpose(make_grid(x_hat.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[1].set_ylabel( "Reconst. images", fontsize=10, fontweight="bold", rotation=90, labelpad=20 )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    if save_img:
        return fig
    else:
        plt.show()


@torch.no_grad()
def generate_samples_with_mode(encoder, codebook, decoder, x_real, device, n_embeddings=512, n_latent=16, mode="random", save_imgs=False):
    encoder.eval()
    decoder.eval()
    x_real = x_real.to(device)

    if mode == "Random":
        z_continuous = encoder(x_real)
        _, indices = codebook.encode(z_continuous)  
        random_indices = torch.randint(0, n_latent, indices.shape, device=device)
        z_quantized = codebook.retrieve_random_codebook(random_indices)

    else:
        z_continuous = encoder(x_real)                      # (B, C, H, W)
        _, indices = codebook.encode(z_continuous)          # (B, H, W)

        if mode == "Reconstruct":
            z_indices = indices

        elif mode == "Perturb":
            perturb_prob = 0.5 # fraction of the indices that are perturbed
            noise = torch.randint(-1, 2, indices.shape, device=device)
            mask = (torch.rand(indices.shape, device=device) < perturb_prob)
            z_indices = (indices + mask * noise).clamp(0, n_embeddings - 1)

        elif mode == "Interpolate":
            num_samples = indices.size(0)
            assert num_samples >= 2, "Need at least 2 samples for interpolation"
            # Average of the first half to the second half
            z_indices = torch.round(0.5 * indices[:(num_samples//2)] + 0.5 * indices[(num_samples//2):num_samples]).long()

        else:
            raise ValueError("Invalid mode")
        z_quantized = codebook.retrieve_random_codebook(z_indices)

    x_hat = decoder(z_quantized)
    fig, axes = plt.subplots(2, 1, figsize=(14, 4))
    axes[0].imshow(np.transpose(make_grid(x_real.detach().cpu(),padding=2,normalize=True),(1, 2, 0)))
    axes[0].set_ylabel( "GT images", fontsize=10, fontweight="bold", rotation=90, labelpad=20 )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(np.transpose(make_grid(x_hat.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    axes[1].set_ylabel( f"{mode} images", fontsize=10, fontweight="bold", rotation=90, labelpad=20 )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    if save_imgs:
        return fig
    else: 
        plt.show()

def plot_json_learning_curves(json_dir, save_img=False):
    with open(json_dir, "r") as f:
        data = json.load(f)
    keys = [key for key in data.keys()]
    colors = ["#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#D62728"]
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharex=True)
    for i, key in enumerate(keys):
        axes[i].plot(
            range(1, len(data[key]) + 1),
            data[key],
            color=colors[i],
            linewidth=2.5,
            label=key.replace("_", " ").title(),)
        axes[i].set_title(key.replace("_", " ").title(), fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Iteration", fontsize=12)
        axes[i].set_ylabel(key.replace("_", " ").title(), fontsize=12)
        axes[i].tick_params(axis="both", labelsize=11)
        axes[i].legend(fontsize=11, frameon=False)
    plt.tight_layout()
    if save_img:
        return fig
    else:
        plt.show()