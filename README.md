# Maternal Health Risk Prediction using Machine Learning

This project focuses on predicting maternal health risk levels (low, medium, and high) using machine learning algorithms. The dataset used for this analysis contains key health indicators such as age, blood pressure, blood sugar levels, body temperature, and heart rate.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Results](#results)
- [How to Use](#how-to-use)


## Introduction
Maternal health is critical for ensuring the well-being of both mothers and their children. By leveraging machine learning, we aim to provide a predictive model to classify maternal health risk levels based on clinical data, which can aid healthcare providers in timely interventions.

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com) and includes the following features:
- **Age**
- **Systolic Blood Pressure (SystolicBP)**
- **Diastolic Blood Pressure (DiastolicBP)**
- **Blood Sugar (BS)**
- **Body Temperature (BodyTemp)**
- **Heart Rate**
- **RiskLevel** (Target variable: Low, Medium, High)

## Technologies Used
The project was developed using:
- Python
  Google Colab Notebook
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for data visualization
- TensorFlow/Keras for deep learning models (if applicable)

## Methodology
 **Data Preprocessing**: 
   - Handling missing values.
   - Feature scaling and normalization.
   - Exploratory data analysis to identify trends and correlations.
   


 **Model Training**:
   - Supervised algorithms: KNN, Gradient Boosting, Logistic Regression,  etc.
     

4. **Model Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-score

5. **Optimization**:
   - Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
   - Addressing overfitting with regularization and cross-validation.

## Results
- The best-performing model achieved an accuracy of **X%** with an F1-score of **Y%**.
- Detailed performance metrics and visualizations are available in the [results notebook](results.ipynb).

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/maternal-health-prediction.git
   cd maternal-health-prediction

