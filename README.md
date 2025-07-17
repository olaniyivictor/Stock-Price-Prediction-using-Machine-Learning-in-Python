# Stock-Price-Prediction-using-Machine-Learning-in-Python

### Goal:
* Build a machine learning model to predict the future closing price of a stock, based on its historical prices.

### Inputs:

* Historical stock data (like Open, High, Low, Close, Volume)

### Outputs:

* Predicted closing price of the stock for future days.

### Why:

* To help investors make informed decisions on buying or selling stocks.

## Problem Statement:
* To develop a machine learning model that predicts the future closing price of a stock based on its historical data, helping investors estimate future trends and make informed investment decisions.
## 📊 Features in the Dataset

| Feature   | Description                                             |
|-----------|---------------------------------------------------------|
| **Date**      | The trading date (often used as index, not directly as a feature) |
| **Open**      | Stock price at the start of the trading day          |
| **High**      | Highest price reached during the trading day         |
| **Low**       | Lowest price reached during the trading day          |
| **Close**     | Price at the end of the trading day (often the target for prediction) |
| **Adj Close** | Adjusted closing price (corrected for splits/dividends) |
| **Volume**    | Number of shares traded during the day              |

## Models and Techniques
#### This project evaluates multiple regression models using pipelines:
*  XGBClassifier
*  LogisticRegression
*  SVC
#### Each model performance is measured using the grid best parameters on the dataset.The best performing model is selected for deployment.
## 🚀 Implementation Step
### 1. Import Libraries
* imported essential Python libraries like numpy, pandas, matplotlib and sklearn.
 ### 2.Preprocess Data
 * Selected the Close column to use as a feature.
 * Created a new target variable by shifting the Close column by one day (so the target is the next day’s price).
  ###  3.Train-Test Split
 * The data is split into training and test sets using an 90-10 split:
 ```python
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=2022, test_size=0.1)
 ```
 ### 4. Model Training
*  Pipelines are used to standardize features and fit models efficiently:
```python
pipe = Pipeline([
    ('preprocessing',StandardScaler()),
    ('classifier',XGBClassifier())
])
#Note that each classifier must be put in a dictionary
param_grid = [
    # For XGBClassifier
    {
        'preprocessing': [StandardScaler(), None],
        'classifier': [XGBClassifier(eval_metric='logloss')],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1]
    },
    # For LogisticRegression
    {
        'preprocessing': [StandardScaler(), None],
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2']
    },
    # For SVC
    {
        'preprocessing': [StandardScaler(),None],
        'classifier': [SVC(probability=True)],
        'classifier__kernel': [ 'poly'],
        
    }
]
```
 ### 5. Model Evaluation
* Each model's  is calculated, and the best model is identified:
```python
print("Best params:\n{}\n".format(grid.best_params_))
```
 ### 6. Results
 * The best model was Logistic Regression, achieving a Test-set score of 0.55.
 
 ### Prerequisites
 * Jupyter notebook
 * Python 3.x
 * Libraries: pandas, NumPy, scikit-learn, LogisticRegression,SVC,XGBClassifier

##### To install dependencies
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
```
### Usage
### 1. Clone the repository:
#### git clone https://github.com/olaniyivictor/Stock-Price-Prediction-using-Machine-Learning-in-Python
#### cd  Stock-Price-Prediction-using-Machine-Learning-in-Python
### 2. Prepare the dataset
* Place Tesla.csv in the project directory.
### 3. Run the script:
#### Project Structure

    .
    ├── Tesla.csv               # Dataset
    ├── notebook.ipynb         # Data preprocessing, model training, model evaluation
    ├── README.md              # Project documentation
    └── requirements.txt       # List of dependencies

## Acknowledgements
* The Tesla.csv dataset provided by the geeksforgeeks.org.
* Libraries and frameworks: scikit-learn, pandas, NumPy, scikit-learn, LogisticRegression,SVC,XGBClassifier.
