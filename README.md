1# Breast Cancer Classification using 10 Machine Learning Models

This repository contains code to classify breast cancer tumors into malignant and benign categories using various machine learning algorithms. The code utilizes the Breast Cancer Wisconsin (Diagnostic) dataset available in scikit-learn.

## Models Implemented
The following machine learning models were trained and evaluated for this classification task:
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes
- Neural Network (MLP Classifier)
- AdaBoost
- XGBoost

## Implementation
The code preprocesses the data by scaling features and splitting it into training and testing sets. Each model is trained on the training data and evaluated on the test set using standard evaluation metrics such as accuracy, precision, recall, F1-score, AUC-ROC, and precision-recall curves.

## Files
- `Predict_breast_cancer_with_10_models.ipynb`: Jupyter Notebook containing the code implementation.
- `README.md`: This file provides an overview of the project.

## Usage
To use this code:
1. Clone the repository: `git clone https://github.com/SohumGoel/Predicting-Breast-Cancer.git`
2. Install necessary dependencies using `pip install -r requirements.txt` (if any).
3. Execute the Jupyter Notebook `Predict_breast_cancer_with_10_models.ipynb` (in Google Colab if the local env is not set up) to run and experiment with different models.

## Results
The evaluation results of each model on the test set are provided in the notebook. The models achieve varying levels of accuracy and performance in distinguishing between malignant and benign tumors.

## Visualizations
The notebook includes visualizations of ROC curves and Precision-Recall curves for each model to showcase their performance.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas, Matplotlib, scikit-learn, XGBoost

