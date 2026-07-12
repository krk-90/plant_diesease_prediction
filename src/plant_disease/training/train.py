import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from Models.registry import get_model
from data.dataset import train_dataset
Model = get_model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.resnet.fc.parameters(),lr= 0.01)

def training():
    epochs = 3
    for epoch in range(epochs):
        Model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in DataLoader(train_dataset, batch_size=32, shuffle=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = Model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = train_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_acc:.4f}")
    return Model

if __name__ == "__main__":
    Model = training()