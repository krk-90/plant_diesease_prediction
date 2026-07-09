import os
import sys
import torch
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torch import nn
from torchvision import datasets,transforms,models
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class plant_disease_prediction(nn.Module):
    def __init__(self,output_class:int):
        super().__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features,output_class)

    def forward(self,x):
        return self.resnet(x)
        

def main():
    file_path = Path(__file__).resolve().parent.parent
    model_path = file_path/"Resnet"/"plant_disease_model.pth"
    loaded_weights_bais = torch.load(model_path,map_location="cpu")
    model = plant_disease_prediction(output_class=6)
    model.load_state_dict(loaded_weights_bais)
    model.eval()
    print("model loaded successfully")

if __name__ == "__main__":
    main()    

