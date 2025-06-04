# Self-supervised Frameworks to Identify Novel Features of 3D Chromatin Structure

**Group 9**: Sam Sterling (ss14424), Muhammad Ali (ma8308), Lucia Galassi (lg3680)

---

## Table of Contents
- [Abstract](#abstract)
- [Introduction & Data](#introduction--data)
- [Methods](#methods)
  - [Masked Autoencoder (ViT)](#masked-autoencoder-vit)
  - [Convolutional Autoencoder (CAE)](#convolutional-autoencoder-cae)
  - [Graph Attention Network](#graph-attention-network)
- [Results](#results)
  - [Masked Autoencoder](#masked-autoencoder)
  - [Convolutional Autoencoder](#convolutional-autoencoder)
  - [Graph Attention Network](#graph-attention-network)
- [Discussion](#discussion)
- [Project Structure](#project-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Usage](#usage)
- [Supplementary Materials](#supplementary-materials)

---

## Abstract

Understanding the 3D structure of chromatin is key to understanding gene regulation, cellular function, and disease progression. Topologically associated domains (TADs) define self-contained 3D regions in which DNA sequences interact more frequently within the domain than with sequences outside it. Hi-C maps visualize this structure but can be challenging to interpret. 

Here, we apply three self-supervised frameworks:
1. Masked autoencoder  
2. Standard autoencoder  
3. Graph-based representation  

to characterize Hi-C data and test their ability to recover known chromatin features. Our masked autoencoder best distinguished gene-dense vs. gene-poor regions, highlighting its promise for downstream discovery of novel chromatin domains.

---

## Introduction & Data

Disruption of 3D chromatin architecture can underlie many diseases, motivating the need to detect TADs and other higher-order features. We explore whether representation-learning models can learn salient Hi-C patterns and pave the way for automated discovery of novel 3D structures.

- **Cell line**: IMR-90 (fetal lung fibroblast, 16-week female)  
- **Resolution**: 10,000 bp per pixel  

**Preprocessing**:
- Observed/expected normalization  
- Clamp at 99th percentile per chromosome  
- Log transform for approximate normality  
- Tiling for vision models: 240×240 tiles (~2.4 Mb), sliding by 60 px → ~4,600 tiles  
- **Graph conversion**: Each locus as a node; edges for interactions > 0.3 normalized frequency; node features = (genomic position, pairwise distance)

---

## Methods

### Masked Autoencoder (ViT)
- **Patch sizes**: 8, 12, 16, 20 px  
- **Mask ratios**: 50%, 65%, 75% (row/column masking)  
- **Tokens**: 768‑dim linear projection + 2D sine‑cosine positional embeddings  
- **Encoder**: 12 layers, 12 heads, 2‑layer MLP each  
- **Decoder**: Lightweight, reconstructs masked patches  
- **Loss**: MSE vs. SSIM (best: 12 px, 65% mask, MSE)  
- **Training**: 200 epochs, AdamW, batch sizes 4, 8, 12, mixed precision; hyperparameter search via Optuna

### Convolutional Autoencoder (CAE)
- **Encoder**: Four 2D conv layers (3×3 kernels, stride 2, pad 1), channels: 1→128 → flatten → FC → latent  
- **Decoder**: FC → reshape → four transposed conv layers → sigmoid output  
- **Latent dims**: 8, 16, 32, 64 (no loss difference; used 32)  
- **Loss**: MSE  
- **Training**: 200 epochs, Adam; Optuna for LR (~2.8×10⁻³) & batch size (8, 16, 32)

### Graph Attention Network
- **Graph construction**: Nodes = loci; edges if normalized contact > 0.3; node features = (position, distance)  
- **NT-Xent loss**: Positive = node pairs on an edge; negative = other nodes in subgraph  
- **Subgraph sampling**: Two hops around each edge’s nodes, merged  
- **GAT architecture**: Three GATConv layers (1 head each), LeakyReLU (slope 0.5), dropout 50%, batch norm  
- **Training**: Adam (LR = 1e‑3, weight decay = 0.05), temperature = 0.1; subset (90 nodes) due to GPU limits

---

## Results

### Masked Autoencoder
- Excellent reconstructions at 65% mask ratio (Fig. 1)  
- Embeddings (4,600 tiles) → k‑means → UMAP (Fig. 2)  
- Cluster 1: smaller loops, high insulation, high gene count (violin plots, Fig. 3)

### Convolutional Autoencoder
- Quality reconstructions; latent=32  
- Embeddings → k‑means → UMAP (Fig. 4A)  
- Cluster 1: low-intensity contacts, low gene count, low insulation (Fig. 4B,C)

### Graph Attention Network
- Trained on subset: NT-Xent loss ↓ to ~0.73 after 30 epochs (Fig. 5)  
- Embeddings can segregate chromosomes only when chromosome ID added as feature (Supp. Fig. 2)

---

## Discussion

Our masked autoencoder captured biologically meaningful chromatin patterns unsupervised, distinguishing gene‑dense, short‑looped domains. The CAE similarly separated low‑interaction regions. The GAT shows promise but is resource‑intensive; richer node features (e.g., chromosome ID) improve clustering. 

**Future work**:
- Multi‑cell‑line training  
- Multi‑omic integration  
- Optimized graph implementations

---

## Project Structure

_TBD_

---

## Installation & Dependencies

_TBD_

---

## Usage

_TBD_

---

## Supplementary Materials

- **Equation 1**: Log₁₀ Min-Max normalization  
- **Equation 2**: LeakyReLU activation  
- **Fig. S1**: Link-prediction MLP results  
- **Fig. S2**: GAT embeddings with/without chromosome feature  

> This project was developed as part of Group 9’s coursework on self-supervised learning for 3D chromatin analysis.
