import torch
import torchvision.transforms as transforms

# Paths
ROOT_DIR = r"."
TRAIN_CSV = r"data\splitted_data\images_train.csv"
VAL_CSV = r"data\splitted_data\images_val.csv"
TEST_CSV = r"data\splitted_data\images_test.csv"
CHECKPOINT_PATH = r"model\checkpoints"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
TOP_K_ACC = 2
TOP_K_AUC = 2

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
