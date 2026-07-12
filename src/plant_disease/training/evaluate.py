import os,sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from training.train import Model,loss_func
from data.dataset import build_datasets
train_dataset, test_dataset = build_datasets()
def accuracy_function(y_true, y_pred):
    y_pred_tags = torch.argmax(y_pred, dim=1)
    correct_results_sum = (y_pred_tags == y_true).sum().item()
    return correct_results_sum

test_loss = 0.0
test_acc =0
Model.eval()
with torch.inference_mode():
    for images_test,labels_test in DataLoader(test_dataset,batch_size=32,shuffle=True):
        images_test,labels_test = images_test.to(device),labels_test.to(device)
        test_outputs= Model(images_test)
        test_loss += loss_func(test_outputs,labels_test).item()*images_test.size(0)
        test_acc += accuracy_function(labels_test,test_outputs)

    test_loss /=len(test_dataset)  
    test_acc /= len(test_dataset)

print(f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}%")
print(train_dataset.dataset.classes)
print(test_dataset.dataset.classes)

print(train_dataset.dataset.class_to_idx)
print(test_dataset.dataset.class_to_idx)
dict = Model.state_dict()
print(dict)
print(len(dict))
file_path = Path(__file__).resolve().parent.parent
model_path = file_path/"data"/"plant_disease_model.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(Model.state_dict(),model_path)