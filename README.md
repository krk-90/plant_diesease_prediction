## 🌱 Plant Disease Prediction
## 📌 Overview
This project uses deep learning to classify plant leaf images into six categories:
Canker, Greening, Gummosis, Healthy, Leaf‑miner, and Lemon‑butterfly.

By leveraging ResNet18, a powerful convolutional neural network, the model helps farmers and researchers detect plant diseases early. Early detection means healthier crops, improved yields, and reduced losses.

##  Goals
Provide an automated way to identify plant diseases from leaf images.

Support farmers with quick, reliable insights for better crop management.

Assist researchers in building scalable solutions for agricultural health monitoring.

## Dataset
The dataset contains ~18,000 images organized into six folders (one per class).

Each folder represents a disease type or healthy leaves.

Images are preprocessed and split into train and test sets with consistent class mappings.

## Features
ResNet18 backbone fine‑tuned for plant disease classification.

Six‑class output layer tailored to the dataset.

Training pipeline built with PyTorch.

Support for GPU acceleration to speed up training.

Evaluation metrics: accuracy, confusion matrix, and per‑class performance.

## How It Works
Load and preprocess leaf images.

Train ResNet18 on the dataset.

Predict disease category for new leaf samples.

Provide results that can guide farmers and researchers in decision‑making.

## Future Work
Experiment with deeper architectures (ResNet50, EfficientNet).

Add real‑time prediction support via IoT devices.

Expand dataset to cover more plant species and diseases.

Deploy as a web or mobile app for farmer‑friendly usage.

## Contribution
Contributions are welcome! You can:

Improve preprocessing techniques.

Suggest new architectures.

Add documentation or tutorials.

## License
This project is open‑source. Feel free to use, modify, and share with proper attribution.