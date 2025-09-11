**Assignment-1**
---
**Credit Card Fraud Detection using KNN Algorithm**

This assignment focuses on detecting fraudulent credit card transactions using the **K-Nearest Neighbors (KNN)** supervised learning algorithm. It includes **data preprocessing** and **Principal Component Analysis (PCA)** with 2 components for dimensionality reduction.

---

## 📌 Assignment Overview

Credit card fraud is a major concern in the financial industry. This assignment demonstrates how machine learning can be used to detect fraudulent transactions using historical data. It uses the KNN algorithm due to its simplicity and effectiveness in classification problems.

---

## 🧠 Technologies & Tools

- Python
- NumPy
- Pandas
- Scikit-learn (sklearn)
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook / Google Colab

---

## 🔍 Dataset

- The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.
- It contains transactions made by European cardholders in September 2013.
- Total: **284,807 transactions**
- Fraudulent transactions: **492** (approx. 0.17%)

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Handling missing values (if any)
   - Feature scaling (StandardScaler)
   - Splitting the dataset into training and testing sets

2. **Dimensionality Reduction with PCA**
   - Reducing the feature space to **2 principal components** for visualization and efficiency

3. **Model Building**
   - Using **K-Nearest Neighbors (KNN)** for classification
   - Hyperparameter tuning for the best value of **K**

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score

5. **Visualization**
   - Plotting decision boundaries (using 2D PCA components)
   - Visualizing fraud vs. non-fraud in PCA space

---

## 📊 Results

- **Dimensionality reduction** to 2 PCA components retained significant variance for fraud detection.
- **KNN classifier** gave reasonable performance given the class imbalance.
- Evaluation metrics demonstrate the model's capability in identifying fraudulent transactions.
