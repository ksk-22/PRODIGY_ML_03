# PRODIGY ML Task 03 - Cats vs Dogs Classification using SVM

## Objective
Implement a Support Vector Machine (SVM) to classify images of cats and dogs.

## Dataset
- 500 Cat Images
- 500 Dog Images

## Libraries
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## Steps
1. Load images
2. Resize images to 64x64
3. Convert to grayscale
4. Flatten image pixels
5. Train SVM classifier
6. Evaluate model accuracy
7. Generate classification report

## Results

### Model Accuracy
**Accuracy: 37.5%**

### Classification Report

```text
              precision    recall  f1-score   support

Cat              0.39      0.33      0.36        21
Dog              0.36      0.42      0.39        19

accuracy                              0.38        40
macro avg        0.38      0.38      0.37        40
weighted avg     0.38      0.38      0.37        40
```

> Note: This project demonstrates the implementation of an SVM classifier for image classification. Accuracy can be improved by using a larger dataset and feature engineering techniques.

## Project Structure
```
PRODIGY_ML_03/
├── README.md
├── requirements.txt
├── main.py
├── Cats/
├── Dogs/
└── svm_results.png
```
