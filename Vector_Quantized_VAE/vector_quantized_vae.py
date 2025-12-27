"""
© Ashkan M., NTNU
MIT License
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
import os
import json
from utils import * 

DATASET_DIR = "./workdir/celebA_dataset/img_align_celeba"
OUTPUT_DIR = "./workdir/output"

LIM_NUM_IMGS = 202599 # total number  202,599
EPOCHS = 500

BATCH_SIZE = 128
IMG_SIZE = (96, 96) 
INPUT_DIM = 3
HIDDEN_DIM = 512
LATENT_DIM = 16
N_EMBEDDINGS= 512
OUTPUT_DIM = 3
COMMITMENT_BETA = 0.25
LR = 2e-4
PRINT_STEP = 20
TEST_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Assigned device: {DEVICE}")


metrics_json = os.path.join(OUTPUT_DIR, f"training_metrics.json")
model_output = os.path.join(OUTPUT_DIR, f"model_{EPOCHS}_epochs_{LIM_NUM_IMGS}_imgs.pth")

data_transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available() and not torch.backends.mps.is_available()}

dataset = SingleFolderDataset(root=DATASET_DIR, transform=data_transform, img_lim=LIM_NUM_IMGS)
train_size = int((1-TEST_RATIO) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **kwargs)

print(f"Training: {len(train_loader.dataset)} images in {len(train_loader)} batches of size {BATCH_SIZE}.")
print(f"Test: {len(test_loader.dataset)} images in {len(test_loader)} batches of size {BATCH_SIZE}.")

encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=LATENT_DIM)
codebook = VQEmbeddingEMA(n_embeddings=N_EMBEDDINGS, embedding_dim=LATENT_DIM)
decoder = Decoder(input_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(DEVICE)
mse_loss = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)

model.train()
history = {
    "recon_loss": [],
    "commitment_loss": [],
    "codebook_loss": [],
    "perplexity": [],
    "total_loss": []}

for epoch in range(EPOCHS):
    epoch_recon = 0.0
    epoch_commit = 0.0
    epoch_codebook = 0.0
    epoch_perplexity = 0.0
    epoch_total = 0.0
    num_batches = 0

    for batch_idx, x in enumerate(train_loader):
        x = x.to(DEVICE)
        optimizer.zero_grad()

        x_hat, commitment_loss, codebook_loss, perplexity = model(x)
        recon_loss = mse_loss(x_hat, x)
        loss = recon_loss + commitment_loss * COMMITMENT_BETA + codebook_loss

        loss.backward()
        optimizer.step()

        epoch_recon += recon_loss.item()
        epoch_commit += commitment_loss.item()
        epoch_codebook += codebook_loss.item()
        epoch_perplexity += perplexity.item()
        epoch_total += loss.item()
        num_batches += 1

        if batch_idx % PRINT_STEP == 0:
            print(
                f"epoch: {epoch + 1} (batch {batch_idx + 1}) "
                f"recon_loss: {recon_loss.item():.3f}  "
                f"perplexity: {perplexity.item():.3f}  "
                f"commit_loss: {commitment_loss.item():.3f} "
                f"codebook_loss: {codebook_loss.item():.3f}  "
                f"total_loss: {loss.item():.3f}\n")

    history["recon_loss"].append(epoch_recon / num_batches)
    history["commitment_loss"].append(epoch_commit / num_batches)
    history["codebook_loss"].append(epoch_codebook / num_batches)
    history["perplexity"].append(epoch_perplexity / num_batches)
    history["total_loss"].append(epoch_total / num_batches)


with open(metrics_json, "w") as f:
    json.dump(history, f, indent=4)
torch.save(model.state_dict(), model_output)

## Evaluation
model.eval()
total_perplexity, total_commitment_loss, total_codebook_loss = 0, 0, 0
num_batches = 0
with torch.no_grad():
    for batch_idx, x in enumerate(tqdm(test_loader)):
        x = x.to(DEVICE)
        x_hat, commitment_loss, codebook_loss, perplexity = model(x)

        total_perplexity += perplexity.item()
        total_commitment_loss += commitment_loss.item()
        total_codebook_loss += codebook_loss.item()
        num_batches += 1

avg_perplexity = total_perplexity / num_batches
avg_commitment_loss = total_commitment_loss / num_batches
avg_codebook_loss = total_codebook_loss / num_batches

print(f"Average metrics over {num_batches} batches:")
print(f"Perplexity: {avg_perplexity:.3f}, Commitment loss: {avg_commitment_loss:.3f}, Codebook loss: {avg_codebook_loss:.3f}")
