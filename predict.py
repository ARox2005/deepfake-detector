import torch
import numpy as np
from PIL import Image
import json
import os

from config import DEVICE, val_transforms, CHECKPOINT_PATH
from model import create_model

def predict_single_image(model, image_path, device):
    model.eval()
    image = Image.open(image_path)
    image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze()
        prob = torch.sigmoid(output).item()

    label = "FAKE" if prob >= 0.5 else "REAL"
    print(f"Prediction: {label} (confidence: {prob:.4f})")
    return prob

def predict_video(model, frame_paths, device):
    probs = []
    for path in frame_paths:
        prob = predict_single_image(model, path, device)
        probs.append(prob)

    avg_prob = np.mean(probs)
    label = "FAKE" if avg_prob >= 0.5 else "REAL"
    print(f"\nVideo Verdict: {label} (avg confidence: {avg_prob:.4f})")
    return avg_prob

if __name__ == "__main__":
    import sys

    top_k_auc_file = os.path.join(CHECKPOINT_PATH, "top_k_auc.json")
    with open(top_k_auc_file, 'r') as f:
        best_list = json.load(f)
    best_path = best_list[0]["path"]
    model = create_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)['model_state_dict'])

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_single_image(model, sys.argv[1], DEVICE)
