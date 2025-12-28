# Variational Autoencoder (VAE) with PyTorch — CelebA

This repository implements a **Variational Autoencoder (VAE)** using **PyTorch**, inspired by the referenced [Kaggle project](https://www.kaggle.com/code/asheniranga/variational-autoencoder-with-pytorch), with the goal of learning a smooth latent representation of images and generating new, realistic samples. Unlike quantized or deterministic autoencoders, the VAE framework enforces a probabilistic latent space, enabling continuous interpolation and more coherent sample generation.

The model is trained on the **CelebA dataset**, a large-scale face attributes dataset commonly used for generative modeling research. By leveraging an encoder–decoder architecture with a variational bottleneck, this project focuses on understanding latent space regularization, reconstruction quality, and sample diversity in image generation tasks.

This repository documents the implementation step by step, with explanations accompanying each code block.

We trained the model for **300 epochs** on the full dataset.  
The corresponding learning curves are shown below:
<p align="center">
  <img src="imgs/learning_curves.png" width="1000">
</p>

A subset of the original input images is shown below:
<p align="center">
  <img src="imgs/input_samples.png" width="800">
</p>

Generated sample images produced by the **encoder–decoder model** are shown below:
<p align="center">
  <img src="imgs/generated_samples.png" width="800">
  <img src="imgs/generated_samples1.png" width="800">
</p>

The video demonstrates how the model progressively improves image reconstruction on a validation sample.
<p align="center">
  <img src="imgs/val_progress.gif" width="800">
</p>

---
© Ashkan M., NTNU  
Released under the MIT License
---


# Vector Quantized Variational Autoencoder (VQ-VAE)

This repository provides a concise and practical implementation of the **Vector Quantized Variational Autoencoder (VQ-VAE)**, originally introduced by [A. Oord et. al.](https://arxiv.org/pdf/1711.00937.pdf) (2017). Unlike standard VAEs, VQ-VAE replaces the continuous latent space with a **discrete, learnable codebook**, enabling more structured latent representations and sharper reconstructions.

The model is trained on the **CelebA dataset**, a widely used benchmark for generative modeling with face images. Through an encoder–decoder architecture combined with vector quantization, this project explores discrete latent representations, reconstruction fidelity, and sample generation behavior, serving as an accessible tutorial for understanding VQ-VAE fundamentals in practice.

Subsequent document provides an step by step implementation guide.

We trained the model for **500 epochs** on the full dataset.  
The corresponding learning curves are shown below:
<p align="center">
  <img src="imgs/learning_curves_celeba_vq_vae.png" width="1000">
</p>

A subset of the original input images is shown below:
<p align="center">
  <img src="imgs/original_inputs.jpg" width="800">
</p>

The corresponding reconstructed images produced by the **encoder–decoder model** are shown below:
<p align="center">
  <img src="imgs/reconstructed_outputs.jpg" width="800">
</p>


# Generating Samples from the VQ-VAE Codebook

Generates and visualizes images from the trained VQ-VAE under different latent manipulation modes. Given real input images, the encoder maps them to discrete codebook indices, which are then modified depending on the selected mode before decoding. The resulting outputs for different modes are plotted alongside the original inputs, enabling qualitative comparison of reconstruction quality and generative behavior under each manipulation strategy.


**Reconstruct**: Uses the original encoded indices to reconstruct the input images.
<p align="center">
  <img src="imgs/org_celeb_and_gen_reconst_samples.png" width="800">
</p>

**Random**: Replaces latent indices with random codebook entries to generate novel samples.
<p align="center">
  <img src="imgs/org_celeb_and_gen_random_samples.png" width="800">
</p>

**Perturb**: Randomly alters a fraction of latent indices to introduce controlled variation.
<p align="center">
  <img src="imgs/org_celeb_and_gen_perturb_samples.png" width="800">
</p>

**Interpolate**: Blends latent codes between pairs of samples to observe smooth transitions.
<p align="center">
  <img src="imgs/org_celeb_and_gen_interpolate_samples.png" width="800">
</p>

© Ashkan M., NTNU  
Released under the MIT License
