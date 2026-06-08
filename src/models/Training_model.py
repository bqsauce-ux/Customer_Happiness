from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV
)
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

print("🔥 RUNNING FILE:", os.path.abspath(__file__))

# Define models and hyperparameter grids
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(
        random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42
    ),
    'XGBoostClassifier': XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ),
    'LGBMClassifier': LGBMClassifier(
        random_state=42,
        verbose=-1
    ),
    'SGDClassifier': SGDClassifier(
        random_state=42
    ),
    'BernoulliNB': BernoulliNB(),
    'GaussianNB': GaussianNB(),
    'NearestCentroid': NearestCentroid(),
    'Perceptron': Perceptron(
        random_state=42
    ),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'RidgeClassifier': RidgeClassifier(),
    'RidgeClassifierCV': RidgeClassifierCV()
}

model_grids = {

    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    },

    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [1, 5]
    },

    'GradientBoosting': {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8]
    },

    'XGBoostClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    },

    'LGBMClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    },

    'SGDClassifier': {
        'loss': ['log_loss', 'hinge'],
        'alpha': [0.0001, 0.001, 0.01]
    },

    'BernoulliNB': {
        'alpha': [0.1, 0.5, 1.0]
    },

    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },

    'NearestCentroid': {
        'metric': ['euclidean', 'manhattan']
    },

    'Perceptron': {
        'penalty': [None, 'l2'],
        'alpha': [0.0001, 0.001]
    },

    'LinearDiscriminantAnalysis': {
        'solver' : ['svd', 'lsqr', 'eigen']
    },

    'RidgeClassifier': {
        'alpha': [0.1, 1.0, 10.0]
    },

    'RidgeClassifierCV': {
        'alphas': [[0.1, 1.0, 10.0]]
    }
}

data_path = 'data/processed/ACME-HappinessSurvey2020.csv' 
data = pd.read_csv(data_path)
X = data.drop('Y', axis=1)
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use XGBoost for RFE to stay consistent
xgb_model = XGBClassifier(objective='binary:logistic',
        random_state=42)

xgb_model.fit(X_train, y_train)

# RFE
rfe_selector = RFE(estimator=xgb_model, n_features_to_select=3)
rfe_selector.fit(X_train, y_train)
rfe_selected_features = X.columns[rfe_selector.support_]
rfe_ignored_features = X.columns[~rfe_selector.support_]

print("✅ Top 3 Selected Features by RFE:")
for feature in rfe_selected_features:
    print(f" - {feature}")

print("\n❌ Features Ignored by RFE:")
for feature in rfe_ignored_features:
    print(f" - {feature}")

# Store for config
selected_features_dict = {
    'rfe': list(rfe_selected_features)
}

# Filter datasets to use only selected features for experimentation
X_train = X[rfe_selected_features]
X_test = X[rfe_selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify= y)
scaler = StandardScaler()

#Fit the scaler on train feature set
X_train = scaler.fit_transform(X_train)


X_test = scaler.transform(X_test)

clf = LazyClassifier(verbose=1,ignore_warnings=True, custom_metric=None, random_state = 42)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

models

import mlflow

mlflow_tracking_uri = 'http://localhost:5000'  
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Customer_Happiness")
#
