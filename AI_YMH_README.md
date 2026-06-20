# YMH (Youth Mental Health) Depression Detection using AI

This repository contains an end-to-end Machine Learning pipeline (`AI-YMH-Main.ipynb`) designed to detect signs of depression in text data. Using a youth wellness dataset, the project applies advanced Natural Language Processing (NLP) techniques, extensive feature engineering, and robust classification models to predict whether a given text indicates depression.

## 📝 Project Overview

The primary goal of this project is to build an accurate text classification model to flag depressive content (`is_depression` label). The notebook walks through:
1. **Data Ingestion:** Loading text data from an Excel dataset (`Integrating AI-DataSet_Youth_Wellness.xlsx`).
2. **Feature Engineering Store:** A custom `DepressionFeatureStore` class that extracts diverse text representations.
3. **Hyperparameter Tuning:** Utilizing `Optuna` to find the optimal parameters for models like Support Vector Machines (SVM) and XGBoost.
4. **Model Training & Evaluation:** Comparing multiple tree-based and ensemble machine learning algorithms.
5. **Error Analysis:** Identifying misclassified samples, cleaning the dataset, and re-evaluating model performance.

## 🧠 Feature Engineering

The text data is transformed into a rich feature space using the following methodologies:
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Captures unigram and bigram word importance.
* **Psycholinguistic Features:** * Sentiment analysis using NLTK's **VADER** (Compound, Positive, Negative scores).
  * **Pronoun Ratio** to detect self-referential language (e.g., *I, me, my*), which is often correlated with depression.
  * **Flesch Reading Ease** score via `textstat` to measure text complexity.
  * Word count metrics.
* **Semantic Embeddings:** Deep contextual embeddings extracted using Hugging Face's `sentence-transformers` (`all-mpnet-base-v2`).

## ⚙️ Models Evaluated

The notebook trains, optimizes, and evaluates several algorithms to find the best performing classifier:
* **Decision Tree** (Entropy & Gini criteria)
* **Linear Support Vector Classifier (LinearSVC)** (Optimized via Optuna)
* **Random Forest Classifier**
* **AdaBoost**
* **XGBoost** (Optimized via Optuna)
* **LightGBM**

*Note: The models consistently achieve ~96-97% accuracy on the test data, as evaluated by comprehensive classification reports, F1-scores, and Confusion Matrices.*

## 🛠️ Requirements & Installation

To run the notebook locally or in Google Colab, you will need the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
pip install optuna xgboost lightgbm
pip install textstat sentence-transformers nltk torch joblib