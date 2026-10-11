# Email Spam Classifier Using Machine Learning

## Project Overview
This project uses machine learning to classify email records as **Spam** or **Not Spam**. It uses the UCI Spambase dataset and compares Logistic Regression with Random Forest to select the best-performing model.

## Features
- Data preprocessing and cleaning
- Training and comparison of two machine learning models
- Spam classification using 57 numerical features
- Model evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix and model comparison visualization
- Saved model for making predictions on dataset records

## Dataset
- **Source:** UCI Machine Learning Repository — Spambase
- **Total records:** 4,601
- **Spam records:** 1,813
- **Non-spam records:** 2,788
- **Features:** 57 numerical attributes

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Model Evaluation
The held-out test set contains 921 records.

| Metric | Spam Class |
|---|---:|
| Precision | 0.94 |
| Recall | 0.93 |
| F1-score | 0.93 |
| Overall accuracy | 0.95 |

These results are from the current training run and may vary if the dataset, split, or model settings change.

## Project Structure
```text
Spam-Email-Classifier/
├── data/
│   ├── spam_dataset.csv
│   ├── model_comparison.csv
│   └── test_predictions.csv
├── graphs/
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── models/
│   └── spam_classifier.pkl
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset
```bash
python src/preprocess.py
```

### 3. Train the models
```bash
python src/train_model.py
```

### 4. Test a dataset record
```bash
python src/predict.py
```

Enter a row number between `0` and `4600` when prompted.

## Important Limitation
The model uses 57 numerical features extracted from email content. It does **not** directly accept arbitrary raw email text. The prediction script tests an existing record from the dataset.

## Learning Outcomes
- Data preprocessing
- Supervised machine learning
- Classification model comparison
- Model evaluation and visualization
- Saving and loading trained models

## Author
**Ansh Negi**

## Disclaimer
This project is developed for educational and internship purposes. Predictions are not guaranteed to be correct.
