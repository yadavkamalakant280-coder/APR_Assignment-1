**Credit Card Fraud Detection using KNN Algorithm**
-

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

## ⚙️ Assignment Workflow

1. **Data Preprocessing**
   i. **Feature Scaling**:
   - Applied Min-Max scaling to 'Time' and 'Amount' columns to scale them between -1 and 1.
   ii. **PCA**:
   - Reduced features to 2 principal components for better visualization and reduced computational complexity.
   iii. **Train-Test Split**:
   - The dataset was split into training and testing sets with a typical ratio (e.g., 80/20)

2. **Dimensionality Reduction with PCA**
   - Reducing the feature space to **2 principal components** for visualization and efficiency

3. **Model Building**
   - Using **K-Nearest Neighbors (KNN)** for classification
   - Hyperparameter tuning for the best value of **K**

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - Evaluation metrics demonstrate the model's capability in identifying fraudulent transactions.

5. **Visualization**
   - Plotting decision boundaries (using 2D PCA components)
   - Visualizing fraud vs. non-fraud in PCA space

---

## 📊 Results

| Metric     | Score      |
|------------|------------|
| Accuracy   | 0.9695     |
| F1 Score   | 0.9677     |
| Precision  | 0.9783     |

Despite the class imbalance, the KNN model performed well in detecting fraud with a high precision and F1 score.

- **Dimensionality reduction** to 2 PCA components retained significant variance for fraud detection.
- **KNN classifier** gave reasonable performance given the class imbalance.
- Evaluation metrics demonstrate the model's capability in identifying fraudulent transactions.

## 📊 Conclusion
KNN can be a simple yet effective method for fraud detection, especially with proper preprocessing and dimensionality reduction. However, due to its high computational cost on large datasets, further optimization or alternative models (e.g., Random Forest, XGBoost) may be explored.
