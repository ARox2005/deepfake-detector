import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data.iloc[index]['file_path'])
        label = self.data.iloc[index]['label']
        image = Image.open(img_path)
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.float32)
