import torch
from torch import nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

class Resnet(nn.Module):
    def __init__(self, output_class: int):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, output_class)

    def forward(self, x):
        return self.resnet(x)

def _build_resnet18(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = Resnet(output_class=num_classes)

    if freeze_backbone:
        for param in model.resnet.parameters():
            param.requires_grad = False
        for param in model.resnet.fc.parameters():
            param.requires_grad = True

    return model.to(device)


_REGISTRY = {
    "resnet18": _build_resnet18,
}


def get_model(architecture: str = "resnet18", num_classes: int = 6, freeze_backbone: bool = True) -> nn.Module:
    architecture = architecture.lower()
    if architecture not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")

    return _REGISTRY[architecture](num_classes=num_classes, freeze_backbone=freeze_backbone)

if __name__ == "__main__":
    model = get_model(architecture="resnet18", num_classes=6)
    print(model)
