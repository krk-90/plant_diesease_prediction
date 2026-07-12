import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "plant-disease-dataset"
DATA_ROOT = Path(os.environ.get("PLANT_DISEASE_DATA_ROOT", DEFAULT_DATA_ROOT))

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)


def build_datasets(data_root: Path = DATA_ROOT, train_split: float = 0.8, seed: int = 42):
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_root}'.\n"
            f"The dataset is too large to bundle with the repo — download it first:\n"
            f"  bash scripts/download_data.sh\n"
            f"Or set PLANT_DISEASE_DATA_ROOT to point at an existing local copy."
        )

    dataset = datasets.ImageFolder(root=str(data_root), transform=None)

    generator = torch.Generator().manual_seed(seed)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_dataset = TransformSubset(dataset, train_subset.indices, train_transform)
    test_dataset = TransformSubset(dataset, test_subset.indices, test_transform)

    return train_dataset, test_dataset


def get_class_names(data_root: Path = DATA_ROOT) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found at '{data_root}'.")
    return sorted(d.name for d in data_root.iterdir() if d.is_dir())


if __name__ == "__main__":
    build_datasets()
    train_dataset, test_dataset = build_datasets()
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Classes: {get_class_names()}")
