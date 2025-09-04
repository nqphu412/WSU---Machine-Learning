# CIFAR-10 Manifold Learning & Visualization

## Table of Contents
- [Overview](#overview)
- [Dataset Sampling](#dataset-sampling)
- [Tasks](#tasks)
  - [Unsupervised Autoencoder](#unsupervised-autoencoder)
  - [Supervised Autoencoder](#supervised-autoencoder)
  - [Embedding Visualization](#embedding-visualization)
- [Requirements](#requirements)
- [Results Summary](#results-summary)

---

## Overview
This notebook implements both unsupervised and supervised manifold-learning on CIFAR-10. We compress three randomly chosen classes into a 2D “bottleneck” representation using convolutional autoencoders, then visualize and analyze the learned embeddings.

---

## Dataset Sampling
1. Set reproducible seeds.  
2. Randomly sample **100 images** from each of **three classes**.  
   - **Main Task classes**: Bird, Cat, Airplane  
   - **Bonus Task classes**: Airplane, Dog, Frog  
3. Re-map labels to `0, 1, 2`.  
4. Split into:  
   - **Train**: 200 images  
   - **Validation**: 95 images  
   - **Test**: 5 images  

---

## Tasks

### Unsupervised Autoencoder
- **Architecture**  
  - **Encoder**:  
    - Conv2d → BatchNorm → ReLU  
    - Conv2d → ReLU → MaxPool  
    - Conv2d → ReLU → MaxPool  
  - **Bottleneck**:  
    - Flatten → Linear(`8×8×8 → 2`)  
  - **Decoder**:  
    - ConvTranspose2d → ReLU (×3) → reconstruct to `3×32×32`  
- **Training**  
  - Loss: MSE (reconstruction)  
  - Optimizer: SGD (lr=0.01)  
  - Epochs: 50 (checkpoint on best validation loss)  
- **Key Result**  
  - Validation loss ↓ `0.1066` → `0.0661`

### Supervised Autoencoder (Bonus)
- **Architecture**  
  - **Feature Extractor**: deeper CNN with Dropout & BatchNorm  
  - **Bottleneck**: Linear(`128×4×4 → 2`)  
  - **Classifier**:  
    - Linear(`128×4×4 → 128`) → ReLU → Dropout  
    - Linear(`128 → 3`) → Softmax  
- **Training**  
  - Loss: CrossEntropy (classification)  
  - Optimizer: SGD (lr=0.01)  
  - Epochs: 15 (to avoid overfitting)  
- **Key Results**  
  - Train Acc: 83%  
  - Val Acc: ~71.6%  
  - Test  Acc: 100% (on 5 unseen samples)

### Embedding Visualization
- **Plot 1**: 200 training embeddings in 2D, color-/marker-coded by class.  
- **Plot 2**: Overlay 5 test embeddings with distinct “+” markers.  
- **Analysis**:  
  - Unsupervised AE: clusters overlap—limited separation.  
  - Supervised AE: Dog vs. Frog clearly separable; Airplane remains diffuse.

---

## Requirements
```bash
pip install torch torchvision numpy pandas matplotlib
```
---

## Results Summary

| Model                   | Validation Loss         | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------------------------|--------------------------|-------------------|---------------------|---------------|
| Unsupervised Autoencoder | 0.1066 → 0.0661         | —                 | —                   | —             |
| Supervised Autoencoder   | —                       | 83%               | 71.6%               | 100%          |

