# Identifying Plant Disease Types Using Leaf Images
![Uploading photo_classification.png…]()

## 1. Task Statement:

- Predict the type of plant disease affecting a leaf based on its image.
- A multi-class classification problem with labels: "healthy," "rust," "scab," and "multiple diseases."

## 2. Motivation:
- Early and accurate identification of plant diseases is critical for reducing crop losses and improving food security.
-  Automating this task can save significant time and resources compared to manual inspections.

## 3.Tentative Models:
   - Baseline: Logistic Regression or Random Forest on basic image features (e.g., RGB histograms).
   - Advanced: Convolutional Neural Networks (CNNs) such as ResNet or EfficientNet for deep feature extraction and classification.

 ## 4.Tools and Libraries:
   - Python: Core programming language.
   - TensorFlow/Keras or PyTorch: For building and training CNN models.
   - Scikit-learn: For baseline models and evaluation.
     
## 5. Investigation Plan:
   - Experiment with different CNN architectures (e.g., pre-trained ResNet).
   - Apply data augmentation to handle imbalances in the dataset and improve generalization.
   - Explore SHAP or LIME for model interpretability.
     
## 6. Analysis
Dataset:
   - Source: Kaggle Plant Pathology 2020 competition dataset.
   - Size: ~3,000 labeled images across four classes.
   - Preprocessing:
     - Normalize images.
     - Perform data augmentation (e.g., rotations, flips).

Benchmark:
   - Baseline model: Simple Logistic Regression or Random Forest on handcrafted features.
   - Advanced model: Fine-tuned CNN (e.g., ResNet) trained on the dataset.

Evaluation Metrics:
   - Primary Metric: F1-Score (weighted), to account for class imbalance.
   - Secondary Metrics: Accuracy, Precision, Recall, and Confusion Matrix.

Goals for the Next Stage:
   - Have a basic pipeline running with a baseline model.
   - Preliminary results from training a CNN architecture.

 ## 7. References:
   - Kaggle competition dataset: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview.
   - Public GitHub repositories for model implementation ideas.


