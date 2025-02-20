# Chest X-Ray Classification with Traditional ML & ResNet

A comprehensive pipeline for classifying Pneumonia vs. Normal chest X-ray images using both traditional machine learning (FOS & LBP features + XGBoost, SVM, RF) and transfer learning (ResNet50).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Methods](#methods)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)
8. [License](#license)
9. [References](#references)

---

## Project Overview
In this project, we explore a complete pipeline for chest X-ray image classification:
- **Data Preprocessing**: Addressing imbalance, augmenting images, applying CLAHE.
- **Feature Extraction**: Using FOS (First-Order Statistics) and LBP.
- **Modeling**: Training SVM, RandomForest, XGBoost, an ensemble model, and fine-tuning with PCA & RandomizedSearchCV.
- **Deep Learning**: Transfer learning with ResNet50.
- **Evaluation**: Classification reports, confusion matrices, threshold tuning, and misclassification analysis.

The primary goal is to accurately detect pneumonia from chest X-ray images.

## Data
- **Dataset Source**: [Kaggle Chest X-Ray Dataset](<link-here>) or a local hospital dataset (depending on your source).
- **Structure**: 
  - `train/` : 4517 NORMAL, 3875 PNEUMONIA (after augmentation)
  - `test/` : 234 NORMAL, 390 PNEUMONIA
  - `val/`  : 8 NORMAL, 8 PNEUMONIA
- **Imbalance**: Original dataset was imbalanced. Additional NORMAL images were augmented to improve balance.

> **Important**: Only a small subset of the data (or none at all) is in this repo. Instructions to obtain the full dataset are [here](docs/dataset_instructions.md).

## Methods
### 1. Data Preprocessing
- **CLAHE** for contrast enhancement.
- **Resizing** all images to 128x128 (for classical ML) or 224x224 (for ResNet).
- **Data Augmentation** with `ImageDataGenerator` (random rotations, shifts, flips).

### 2. Feature Extraction (Classical ML)
- **FOS**: Mean, standard deviation.
- **LBP**: Local Binary Patterns with 8 points, radius=1.

### 3. Addressing Class Imbalance
- Augmentation of NORMAL images.
- **SMOTE** to oversample minority class in the feature space.

### 4. Model Training
- **Classical Models**: SVC, RandomForest, XGBoost.
  - Ensemble model (VotingClassifier) with higher weight on XGBoost.
  - Hyperparameter tuning with `RandomizedSearchCV`.
- **Deep Learning (ResNet50)**:
  - Transfer learning from ImageNet.
  - Freeze base layers, train custom head for binary classification.
  - Early stopping & checkpointing.

### 5. Evaluation
- **Metrics**: Precision, Recall, F1-score, Accuracy.
- **Threshold Tuning** to adjust trade-off between sensitivity (recall) and specificity.
- **Confusion Matrices** & misclassification analysis.

## Installation and Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/chest-xray-classification.git
   cd chest-xray-classification


Results

Best Classical Model: Fine-tuned XGBoost achieved ~86% accuracy on test set (F1 = 0.90 for Pneumonia).
Ensemble: Soft-voting ensemble slightly boosted performance to 87%.
ResNet50: ~78–82% accuracy, though further fine-tuning may improve results.
<img width="815" alt="image" src="https://github.com/user-attachments/assets/a4d1a9c6-ef62-4fce-8b0a-3b0db820cef4" />

