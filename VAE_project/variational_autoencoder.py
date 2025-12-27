"""
© Ashkan M., NTNU
MIT License
"""

import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils import *
from warnings import filterwarnings
filterwarnings(action='ignore')

DATASET_DIR = "./workdir/celebA_dataset/img_align_celeba"
CHKPOINT_DIR = "./workdir/output"
if not os.path.exists(CHKPOINT_DIR):
    os.makedirs(CHKPOINT_DIR)
                
NUM_EPOCHS = 300

LIM_NUM_IMGS = 202599 # total number is more then 202,599
IMG_SIZE = (64, 64) 
BATCH_SIZE = 64
IN_CHANNELS = 3
NUM_EMBEDDINGS = 256 
KERNEL_SIZE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device assigned:", DEVICE)

data_transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available() and not torch.backends.mps.is_available()}

dataset = SingleFolderDataset(root=DATASET_DIR, transform=data_transform, img_lim=LIM_NUM_IMGS)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **kwargs)

print(f"Training: {len(train_dataloader.dataset)} images in {len(train_dataloader)} batches of size {BATCH_SIZE}.")
print(f"Test: {len(test_dataloader.dataset)} images in {len(test_dataloader)} batches of size {BATCH_SIZE}.")

model = ConvVariationalAutoEncoder(IN_CHANNELS, NUM_EMBEDDINGS, KERNEL_SIZE, IMG_SIZE)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_parameters:,}")

model = model.to(DEVICE)

figure = plot_random_images(train_dataloader, num_images=12, save_imgs=True)
figure.savefig(os.path.join(CHKPOINT_DIR, f"input_samples.png"), dpi=300, bbox_inches="tight")
plt.close(figure)

loss_func = nn.MSELoss(reduction='sum') 
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)

results = train_loop(model, 
                     loss_func,
                     train_loader=train_dataloader,
                     chkpoint_dir=CHKPOINT_DIR,
                     val_loader=test_dataloader,
                     optimizer=optimizer,
                     lr_schedule=scheduler,
                     device=DEVICE,
                     epochs=NUM_EPOCHS, plot_cases_val=True)

figure = plot_learning_curves(results, save_img=True)
figure.savefig(os.path.join(CHKPOINT_DIR, f"learning_curves.png"), dpi=300, bbox_inches="tight")
plt.close(figure)

save_results_df2json(results, CHKPOINT_DIR)

model = model.to("cpu")
figure = plot_generated_test_samples(model, test_dataset, n_examples=8, save_imgs=True)
figure.savefig(os.path.join(CHKPOINT_DIR, f"generated_samples.png"), dpi=300, bbox_inches="tight")
plt.close(figure)