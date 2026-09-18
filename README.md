# Class Activation Mapping for Brain Tumor Localization

Class activation mapping (CAM) is a technique to visualize which part of an image is most responsible for a CNN prediction. To put it simply, it highlights the image region which makes the model give its prediction. Although its main function is to make CNN models more transparent as to how it makes its prediction, CAM can also be used to localize object in an image without explicit localization labels. This task is commonly referred as Weakly Supervised Object Localization (WSOL).

This repo is focused on using CAMs to localize brain tumor, mainly in the BRISC 2025 dataset. The dataset contains 4 classes including no-tumor, glioma, meningioma, and pituitary. 

The dataset can be found here: https://www.kaggle.com/datasets/briscdataset/brisc2025
