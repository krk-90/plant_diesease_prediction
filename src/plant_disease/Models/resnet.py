import torch
from torch import nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self,output_class:int):
        super().__init__()
        self.resnet = models.resnet18(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features,output_class)

    def forward(self,x):
        return self.resnet(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_model():            
    Resnet_model = Resnet(output_class=6).to(device)
    for param in Resnet_model.resnet.parameters():
        param.requires_grad = False

    for param in Resnet_model.resnet.fc.parameters():
        param.requires_grad = True

    return Resnet_model    

if __name__ == "__main__":
    Model = get_model()
    

