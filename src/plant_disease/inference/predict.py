import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from training.train import Model   

def main():
    file_path = Path(__file__).resolve().parent.parent
    model_path = file_path/"data"/"plant_disease_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    loaded_weights_bais = torch.load(model_path,map_location="cpu",weights_only=True)
    model = Model(output_class=6)
    model.load_state_dict(loaded_weights_bais)
    model.eval()
    return model

if __name__ == "__main__":
    main()    

