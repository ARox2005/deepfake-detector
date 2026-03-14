import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm
import json
import os

from config import *
from dataset import DeepfakeDataset
from model import create_model
from train import validate

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    for images, labels in tqdm(loader):
        images = images.to(device)
        outputs = model(images).squeeze(1)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load best model
    model = create_model(pretrained=False).to(DEVICE)

    top_k_auc_file = os.path.join(CHECKPOINT_PATH, "top_k_auc.json")
    with open(top_k_auc_file, 'r') as f:
        best_list = json.load(f)
    best_path = best_list[0]["path"]
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)['model_state_dict'])

    # Test loader
    test_loader = DataLoader(
        DeepfakeDataset(TEST_CSV, ROOT_DIR, val_transforms),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Evaluate
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_acc, test_auc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    # Confusion matrix
    y_true, y_pred = get_predictions(model, test_loader, DEVICE)
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
