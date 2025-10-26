# 🎓 CS6886W – Assignment 1: VGG6 Modular Implementation

This repository contains a **fully modular PyTorch implementation** of a **VGG6-style CNN** trained on the **CIFAR-10** dataset.  
It includes code for training, testing, Weights & Biases sweeps, and final model checkpoints — satisfying all deliverables for **CS6886W Assignment 1**.

---

## 📁 Project Structure

| File | Description |
|------|--------------|
| `model.py` | Defines the VGG6 CNN architecture (activation-configurable, with BatchNorm). |
| `data_loader.py` | Prepares CIFAR-10 dataset loaders with normalization, augmentation + Cutout. |
| `train.py` | Training script for a single configuration; saves `latest.pt` and `best.pt` checkpoints. |
| `test.py` | Loads a trained checkpoint and evaluates on the CIFAR-10 test set. |
| `utils.py` | Helper utilities (accuracy, training loop, optimizer/scheduler setup, seeding). |
| `sweep_full.py` | Complete 25-run W&B sweep pipeline + charts + final retrain and test. |
| `checkpoints/` | Contains trained model weights (`best.pt`, `latest.pt`, etc.) tracked with Git LFS. |
| `README.md` | This documentation and grader verification steps. |

---

## ⚙️ Best Configuration (Highest Validation Accuracy)

| Hyper-parameter | Value |
|-----------------|--------|
| Activation | `gelu` |
| Optimizer | `rmsprop` |
| Learning Rate | `5 × 10⁻⁴` |
| Batch Size | `128` |
| Epochs | `20` |
| Weight Init | Kaiming |
| Batch Norm | Enabled |
| Data Augmentation | Enabled (Crop + Flip + Cutout) |
| seed Config | seed=42 | 

Best trained model checkpoint:  

## 🧪 How to Verify the Trained Model

Follow these steps to evaluate the trained model directly.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/harieshwarg/cs6886w_a1_vgg6_modular_OP.git
cd cs6886w_a1_vgg6_modular_OP
```

### 2️⃣ (Optional) Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies
```
pip install torch torchvision matplotlib plotly pandas wandb
```

###  4 Training the Model (Optional) or else use the best.pt saved in checkpoint 
```
python train.py --activation gelu --optimizer rmsprop --lr 5e-4 --batch-size 128 --epochs 20 --augment --batch-norm
```
```
My output when I ran it 
/content/cs6886w_a1_vgg6_modular_OP
epoch 01 | train_loss 1.6257 | val_loss 1.6786 | val_acc 43.36
epoch 02 | train_loss 1.2557 | val_loss 1.2975 | val_acc 55.04
epoch 03 | train_loss 1.0940 | val_loss 1.0681 | val_acc 62.32
epoch 04 | train_loss 0.9879 | val_loss 1.0191 | val_acc 63.50
epoch 05 | train_loss 0.9099 | val_loss 1.0099 | val_acc 65.42
epoch 06 | train_loss 0.8536 | val_loss 0.9997 | val_acc 64.82
epoch 07 | train_loss 0.8080 | val_loss 0.8321 | val_acc 71.30
epoch 08 | train_loss 0.7825 | val_loss 0.7983 | val_acc 71.70
epoch 09 | train_loss 0.7460 | val_loss 0.8014 | val_acc 71.46
epoch 10 | train_loss 0.7149 | val_loss 0.7642 | val_acc 72.66
epoch 11 | train_loss 0.6943 | val_loss 0.7693 | val_acc 72.90
epoch 12 | train_loss 0.6740 | val_loss 0.8380 | val_acc 70.48
epoch 13 | train_loss 0.6526 | val_loss 0.7316 | val_acc 74.30
epoch 14 | train_loss 0.6325 | val_loss 0.6813 | val_acc 76.28
epoch 15 | train_loss 0.6160 | val_loss 0.7049 | val_acc 74.66
epoch 16 | train_loss 0.5974 | val_loss 0.7097 | val_acc 74.46
epoch 17 | train_loss 0.5862 | val_loss 0.6450 | val_acc 77.50
epoch 18 | train_loss 0.5777 | val_loss 0.6539 | val_acc 76.92
epoch 19 | train_loss 0.5641 | val_loss 0.7081 | val_acc 75.36
epoch 20 | train_loss 0.5526 | val_loss 0.6899 | val_acc 75.92
Done. Best val_acc: 77.5
```

### 5 Verifying the Model
```
python test.py --ckpt checkpoints/best.pt --activation gelu --batch-norm
```

```
Expected output
My output when I ran the same
/content/cs6886w_a1_vgg6_modular_OP
TEST — loss 0.5739 | acc 81.47% | from ckpt checkpoints/best.pt
(saved best val_acc: 77.50%)
total 9.1M
``` 


```
In short for recreating

git clone https://github.com/harieshwarg/cs6886w_a1_vgg6_modular_OP.git
cd cs6886w_a1_vgg6_modular_OP
pip install torch torchvision
python test.py --ckpt checkpoints/best.pt --activation gelu --batch-norm

```
