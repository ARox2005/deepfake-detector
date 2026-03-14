import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import glob
import json

from config import *
from dataset import DeepfakeDataset
from model import create_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    for images, labels in tqdm(loader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    auc = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, auc

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Class imbalance weight
    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    num_real = len(df_train[lambda x: x['label'] == 0])
    num_fake = len(df_train[lambda x: x['label'] == 1])
    print(f"Reals: {num_real}\nFakes: {num_fake}")

    # Model, loss, optimizer
    model = create_model(pretrained=True).to(DEVICE)
    pos_weight = torch.tensor([num_real / num_fake]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Datasets & Loaders
    train_loader = DataLoader(
        DeepfakeDataset(TRAIN_CSV, ROOT_DIR, train_transforms),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        DeepfakeDataset(VAL_CSV, ROOT_DIR, val_transforms),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Resume from checkpoint
    best_val_auc = 0.0
    start_epoch = 0
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Top-K tracking file
    top_k_acc_file = os.path.join(CHECKPOINT_PATH, "top_k_acc.json")
    top_k_acc_list = []

    if os.path.exists(top_k_acc_file):
        with open(top_k_acc_file, 'r') as f:
            top_k_acc_list = json.load(f)

    top_k_auc_file = os.path.join(CHECKPOINT_PATH, "top_k_auc.json")
    top_k_auc_list = []
    if os.path.exists(top_k_auc_file):
        with open(top_k_auc_file, 'r') as f:
            top_k_auc_list = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",
                        type=str,
                        default="latest",
                        help = "'latest', 'best', or path to a specific .pth file")
        
    args = parser.parse_args()

    resume_path = None
    if args.resume == "latest":
        checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_PATH, "checkpoint_*.pth")))
        if checkpoint_files:
            resume_path = checkpoint_files[-1]

    elif args.resume == "best":
        if os.path.exists(top_k_auc_file):
            with open(top_k_auc_file, 'r') as f:
                best_list = json.load(f)
            if best_list:
                resume_path = best_list[0]["path"]  # top-1 AUC checkpoi

    else:
        resume_path = args.resume

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint['best_val_auc']
            print(f"✅ Resumed from {resume_path} (best AUC: {best_val_auc:.4f})")
        else:
            model.load_state_dict(checkpoint)
            print("⚠️ Loaded model weights only (no optimizer state). Starting fresh epoch count.")
            print(f"✅ Resumed from {resume_path} (no optimizer/epoch info)")
    else:
        print("No checkpoint found. Training from scratch.")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(CHECKPOINT_PATH, f"checkpoint_{epoch+1}.pth"))
        print(f"💾 Checkpoint saved (epoch {epoch+1})")
        checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_PATH, "checkpoint_*.pth")))
        while len(checkpoint_files) > 3:
            old = checkpoint_files.pop(0)
            os.remove(old)
            print(f"🗑️ Removed old checkpoint: {os.path.basename(old)}")

        # Save top-K checkpoints by validation ACCURACY
        top_k_acc_entry = {
            "acc": val_acc,
            "epoch": epoch + 1,
            "path": os.path.join(CHECKPOINT_PATH, f"top_k_epoch_{epoch+1}.pth")
        }
        if len(top_k_acc_list) < TOP_K_ACC or val_acc > min(e["acc"] for e in top_k_acc_list):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, top_k_acc_entry["path"])
            top_k_acc_list.append(top_k_acc_entry)
            top_k_acc_list.sort(key=lambda x: x["acc"], reverse=True)
            if len(top_k_acc_list) > TOP_K_ACC:
                removed = top_k_acc_list.pop()
                if os.path.exists(removed["path"]):
                    os.remove(removed["path"])
                print(f"🗑️ Removed checkpoint epoch {removed['epoch']} (Acc: {removed['acc']:.4f})")   # ← acc
            with open(top_k_acc_file, "w") as f:
                json.dump(top_k_acc_list, f, indent=2)
            print(f"🏆 Top-{TOP_K_ACC} checkpoint saved (epoch {epoch+1}, Acc: {val_acc:.4f})")   # ← val_acc

        # Save top-K checkpoints by validation AUC
        top_k_auc_entry = {
            "auc": val_auc,
            "epoch": epoch + 1,
            "path": os.path.join(CHECKPOINT_PATH, f"best_auc_epoch_{epoch+1}.pth")
        }
        if len(top_k_auc_list) < TOP_K_AUC or val_auc > min(e["auc"] for e in top_k_auc_list):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, top_k_auc_entry["path"])
            top_k_auc_list.append(top_k_auc_entry)
            top_k_auc_list.sort(key=lambda x: x["auc"], reverse=True)
            if len(top_k_auc_list) > TOP_K_AUC:
                removed = top_k_auc_list.pop()
                if os.path.exists(removed["path"]):
                    os.remove(removed["path"])
                print(f"🗑️ Removed AUC checkpoint epoch {removed['epoch']} (AUC: {removed['auc']:.4f})")
            with open(top_k_auc_file, "w") as f:
                json.dump(top_k_auc_list, f, indent=2)
            print(f"🏆 Top-{TOP_K_AUC} AUC checkpoint saved (epoch {epoch+1}, AUC: {val_auc:.4f})")
        # Update best_val_auc tracker
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"🔝 New best AUC: {val_auc:.4f}")
