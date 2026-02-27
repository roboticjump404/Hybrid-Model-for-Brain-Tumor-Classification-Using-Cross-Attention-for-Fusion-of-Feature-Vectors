# Hybrid-Model-for-Brain-Tumor-Classification-Using-Cross-Attention-for-Fusion-of-Feature-Vectors


A deep learning framework for brain tumor classification from MRI images using a hybrid architecture that combines:

ResNet-50 (CNN) → local texture & boundary features

Swin Transformer (ViT) → global spatial context

Cross-Attention Fusion → adaptive feature integration

The model performs multi-class classification (4 tumor categories) and includes training, evaluation, visualization, and early stopping.

# Dataset 

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

# Features

Hybrid CNN + Transformer architecture

Cross-Attention based feature fusion

Data augmentation for medical images

Early stopping to prevent overfitting

Confusion matrix & classification report

Training & validation curves automatically saved

# Training Strategy

Transfer learning using ImageNet pretrained weights

Data augmentation for robustness

Early stopping based on validation performance

Best model checkpoint saving

# Evaluation Metrics

The following metrics are automatically computed:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Training and validation curves
