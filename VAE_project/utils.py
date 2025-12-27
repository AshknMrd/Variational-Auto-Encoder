"""
© Ashkan M., NTNU
MIT License
"""

import torch
import os
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import time
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

def checkpointer(chkpoint_dir, score, epoch, model, optimizer, best_score_dict={}):
    best = best_score_dict.get("best_score", float("-inf"))
    if score < best:  # or < best if lower is better
        print(f"New best score ({score:.4f}) at epoch {epoch}. Saving checkpoint...")
        save_path = os.path.join(chkpoint_dir, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score}, save_path)
        best_score_dict["best_score"] = score
        
def plot_samples(inputs, x_hat, epoch, save_imgs=False):
    inputs_np = inputs.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()
    n_samples = min(8, len(inputs_np))

    fig, axes = plt.subplots(2, n_samples, figsize=(1.05 * n_samples, 2))
    for i in range(n_samples):
        img_in = np.squeeze(inputs_np[i])
        img_out = np.squeeze(x_hat_np[i])
        if img_in.ndim == 2:  # grayscale
            axes[0, i].imshow(img_in, cmap="gray")
            axes[1, i].imshow(img_out, cmap="gray")
        else:  # RGB
            axes[0, i].imshow(np.moveaxis(img_in, 0, -1))
            axes[1, i].imshow(np.moveaxis(img_out, 0, -1))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.text(0.0, 0.70, "Input", fontsize=10, fontweight="bold",rotation=90, va="center")
    fig.text(0.0, 0.28, "Recons.", fontsize=10, fontweight="bold",rotation=90, va="center")
    plt.tight_layout()
    if save_imgs:
        return fig
    else:
        plt.show()


def run_epoch(
    model, optimizer, data_loader, loss_func, chkpoint_dir, device,
    results, score_funcs, prefix="", desc=None, keep=False, epoch=None, plot_cases=False
):
    """Run a single epoch for training/validation, compute loss, and update model parameters."""
    running_loss = []
    recon_loss_vec = []
    kl_loss_vec = []
    x_true, x_pred = [], []
    start_time = time.time()

    plot_idx=0
    for inputs in tqdm(data_loader, desc=desc, leave=keep):
        inputs = inputs.to(device)
        x_hat, mu, sigma = model(inputs)

        if plot_cases and plot_idx==0:
            print(f"Validation at epoch {epoch}: images saved.")

            if not os.path.exists(os.path.join(chkpoint_dir, "val_progress")):
                os.makedirs(os.path.join(chkpoint_dir, "val_progress"))
            figure = plot_samples(inputs, x_hat, epoch, save_imgs=True)
            figure.savefig(os.path.join(chkpoint_dir, "val_progress", f"val_epoch{epoch}.png"), dpi=300, bbox_inches="tight")
            plt.close(figure)
            plot_idx+=1

        # Compute reconstruction + KL divergence loss
        recon_loss = loss_func(x_hat, inputs)
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # kl_div = -0.5 * torch.mean(torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim=1))
        loss = recon_loss + kl_div

        # Backpropagation (only in training mode)
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss.append(loss.item())
        recon_loss_vec.append(recon_loss.item())
        kl_loss_vec.append(kl_div.item())

        # Collect predictions for scoring
        if score_funcs and isinstance(inputs, torch.Tensor):
            x_true.extend(inputs.detach().cpu().numpy().tolist())
            x_pred.extend(x_hat.detach().cpu().numpy().tolist())

    # Compute average loss
    avg_loss = np.mean(running_loss)
    avg_recon_loss = np.mean(recon_loss_vec)
    avg_kl_loss = np.mean(kl_loss_vec)

    results[f"{prefix} loss"].append(avg_loss)
    results[f"{prefix} recon-loss"].append(avg_recon_loss)
    results[f"{prefix} kl-loss"].append(avg_kl_loss)

    result_str = [f"{prefix} loss: {avg_loss:.4f}"]

    # Compute additional metrics
    for name, score_func in score_funcs.items():
        try:
            score = score_func(x_true, x_pred)
            results[f"{prefix} {name}"].append(score)
            result_str.append(f"{prefix} {name}: {score:.4f}")

            if prefix in {"val", "test"}:
                checkpointer(chkpoint_dir, score, epoch + 1, model, optimizer)

        except Exception:
            results[f"{prefix} {name}"].append(float("NaN"))
            result_str.append(f"{prefix} {name}: NaN")

    # print(" ".join(result_str))
    return time.time() - start_time

def train_loop(
    model, loss_func,
    train_loader, chkpoint_dir, test_loader=None, val_loader=None,
    score_funcs=None, epochs=50, device="cpu",
    optimizer=None, lr_schedule=None, keep=True, plot_cases_val=False
):
    """Full training loop for a neural network with optional validation and testing."""
    
    score_funcs = score_funcs or {}

    # Define metrics to track
    to_track = ["epoch", "total time", "train loss", "train recon-loss", "train kl-loss", "lr"]
    if val_loader is not None:
        to_track.extend(["val loss", "val recon-loss", "val kl-loss"])
    if test_loader is not None:
        to_track.extend(["test loss", "test recon-loss", "test kl-loss"])

    for metric in score_funcs:
        to_track.append(f"train {metric}")
        if val_loader is not None:
            to_track.append(f"val {metric}")
        if test_loader is not None:
            to_track.append(f"test {metric}")

    # Initialize tracking dictionary
    results = {item: [] for item in to_track}
    total_train_time = 0

    # Initialize optimizer if not provided
    del_opt = False
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        del_opt = True

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch", leave=keep):
        model.train()
        epoch_time = run_epoch(
            model, optimizer, train_loader, loss_func, chkpoint_dir, device,
            results, score_funcs, prefix="train", desc="Training", plot_cases=False)

        total_train_time += epoch_time
        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)
        results["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch} time: {round(epoch_time,2)} sec")
        print(f"Train Loss:        {round(results['train loss'][-1],2)}")

        # Validation phase
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                run_epoch(
                    model, optimizer, val_loader, loss_func, chkpoint_dir, device,
                    results, score_funcs, prefix="val", desc="Validation", epoch=epoch, plot_cases=plot_cases_val)

        # Testing phase
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                run_epoch(
                    model, optimizer, test_loader, loss_func, chkpoint_dir, device,
                    results, score_funcs, prefix="test", desc="Testing", epoch=epoch, plot_cases=False)

        # Learning rate scheduling
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()

        # Checkpointing
        if epoch == epochs - 1:
            checkpoint_file = os.path.join(chkpoint_dir, f"model_last_E{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results": results
            }, checkpoint_file)

    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)

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
    
class Encode(nn.Module):

    def __init__(self, channels, embeddings, kernel, image_size):
        super(Encode, self).__init__()

        self.relu = nn.ReLU()
        self.filters = [32, 64, 128, 256, 512]
        self.conv1 = nn.Conv2d(channels, self.filters[0], kernel_size=kernel, padding=1)
        self.bn1 = nn.BatchNorm2d(self.filters[0])
        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], kernel_size=kernel, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.filters[1])
        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], kernel_size=kernel, padding=1)
        self.bn3 = nn.BatchNorm2d(self.filters[2])
        self.conv4 = nn.Conv2d(self.filters[2], self.filters[3], kernel_size=kernel, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.filters[3])
        self.conv5 = nn.Conv2d(self.filters[3], self.filters[4], kernel_size=kernel, padding=1)
        self.bn5 = nn.BatchNorm2d(self.filters[4])

        self.pool_size = image_size[0]//16
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.mu = nn.Linear(self.filters[4] * self.pool_size * self.pool_size, embeddings)
        self.sigma = nn.Linear(self.filters[4] * self.pool_size * self.pool_size, embeddings)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.mu(x), self.sigma(x)
    
    
class Decode(nn.Module):
    def __init__(self, channels, embeddings, kernel, image_size):
        super(Decode, self).__init__()

        self.relu = nn.ReLU()
        self.filters = [512, 256, 128, 64, 32]
        self.pool_size = image_size[0]//16
        self.decode_in = nn.Linear(embeddings, self.filters[0] * self.pool_size * self.pool_size)
        self.deconv1 = nn.ConvTranspose2d(self.filters[0], self.filters[1], kernel_size=kernel, stride=2, output_padding=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.filters[1])
        self.deconv2 = nn.ConvTranspose2d(self.filters[1], self.filters[2], kernel_size=kernel, stride=2, output_padding=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.filters[2])
        self.deconv3 = nn.ConvTranspose2d(self.filters[2], self.filters[3], kernel_size=kernel, stride=2, output_padding=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.filters[3])
        self.deconv4 = nn.ConvTranspose2d(self.filters[3], self.filters[4], kernel_size=kernel, stride=2, output_padding=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.filters[4])
        self.deconv5 = nn.ConvTranspose2d(self.filters[4], channels, kernel_size=kernel, stride=1, padding=1)
        self.decode_out = nn.Sigmoid()

    def forward(self, x):
        x = self.decode_in(x)
        x = x.view(-1, self.filters[0], self.pool_size, self.pool_size)
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.decode_out(self.deconv5(x))
        return x

class ConvVariationalAutoEncoder(nn.Module):
    """variational autoencoder model"""

    def __init__(self, channels, embeddings, kernel, image_size):
        super(ConvVariationalAutoEncoder, self).__init__()
        self.encoder = Encode(channels, embeddings, kernel, image_size)
        self.decoder = Decode(channels, embeddings, kernel, image_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)            
        z = mu + torch.exp(0.5 * sigma) * epsilon
        x_hat = self.decoder(z)
        return x_hat, mu, sigma

def plot_random_images(dataloader, num_images=8, save_imgs=False):
    dataset = dataloader.dataset
    indices = random.sample(range(len(dataset)), num_images)

    cols = int(np.ceil(num_images / 2))
    fig, axes = plt.subplots(2, cols, figsize=(2 * cols, 4))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        img = dataset[idx]          # 👈 image only
        img = img.cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        axes[i].axis("off")
    for j in range(num_images, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if save_imgs:
        return fig
    else:
        plt.show()

def plot_learning_curves(results, save_img=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True)

    sns.lineplot(x="epoch", y="train loss", data=results,label="Train", color="red", ax=axes[0], linewidth=2)
    sns.lineplot(x="epoch", y="val loss", data=results,label="Validation", color="blue", ax=axes[0], linewidth=2)
    axes[0].set_title("Total Loss", fontweight="bold")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")

    sns.lineplot(x="epoch", y="train recon-loss", data=results,label="Train", color="red", ax=axes[1], linewidth=2)
    sns.lineplot(x="epoch", y="val recon-loss", data=results,label="Validation", color="blue", ax=axes[1], linewidth=2)
    axes[1].set_title("Reconstruction Loss", fontweight="bold")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")

    sns.lineplot(x="epoch", y="train kl-loss", data=results,label="Train", color="red", ax=axes[2], linewidth=2)
    sns.lineplot(x="epoch", y="val kl-loss", data=results,label="Validation", color="blue", ax=axes[2], linewidth=2)
    axes[2].set_title("KL Divergence Loss", fontweight="bold")
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False)
    plt.tight_layout()
    if save_img:
        return fig
    else:
        plt.show()

def get_digits(dataset):
    idx = torch.randint(len(dataset), (1,))
    return dataset[idx].unsqueeze(0)

def get_encodings(model, image):
    with torch.no_grad():
        mu, sigma = model.encoder(image)
    return mu, sigma

def get_reconstructed(model, image):
    with torch.no_grad():
        recon, _, _ = model(image)
    return recon

def inference(*, model, dataset, n_examples=1):
    out = []
    image = get_digits(dataset)
    mu, sigma = get_encodings(model, image)
    recon_img = get_reconstructed(model, image)
    for _ in range(n_examples):
        epsilon = torch.randn_like(sigma)         
        z = mu + torch.exp(0.5 * sigma) * epsilon
        out.append(model.decoder(z))
    return image, out, recon_img

def plot_generated_test_samples(model, test_dataset, n_examples=8, save_imgs=False):
    input_digit, outs, recon_img = inference(model=model, dataset=test_dataset, n_examples=n_examples)
    num_cols = 2 + n_examples  # Org + Recon + Generateds
    fig, axes = plt.subplots(1, num_cols,figsize=(num_cols * 1.6, 2.2),constrained_layout=True)

    axes[0].imshow(input_digit[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Org.", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(recon_img[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Recon.", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    for i in range(n_examples):
        axes[i + 2].imshow(outs[i].detach().cpu().permute(0, 2, 3, 1).squeeze().numpy())
        axes[i + 2].set_title(f"Gen {i+1}", fontsize=10)
        axes[i + 2].axis("off")
    fig.suptitle("Generated Samples", fontsize=14, fontweight="bold")
    if save_imgs:
        return fig
    else:
        plt.show()

def save_results_df2json(results_df, chkpoint_dir):
    results_json = {}
    for k, v in results_df.items():
        if hasattr(v, "to_dict"):
            results_json[k] = v.to_dict()
    with open(os.path.join(chkpoint_dir, f"results.json"), "w") as f:
        json.dump(results_json, f, indent=2)