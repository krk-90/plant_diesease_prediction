from PIL import Image
import torchvision.transforms as transforms

class preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def apply(self,IMG:Image.Image):
        return self.transform(IMG)

def processed(img:Image.Image):
    processor = preprocessor()
    transformed_image = processor.apply(img)
    return transformed_image.unsqueeze(0)
if __name__ == "__main__":
    processed()