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

    'LogisticRegression': {},

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

    'GaussianNB': {},

    'NearestCentroid': {},

    'Perceptron': {
        'penalty': [None, 'l2'],
        'alpha': [0.0001, 0.001]
    },

    'LinearDiscriminantAnalysis': {},

    'RidgeClassifier': {
        'alpha': [0.1, 1.0, 10.0]
    },

    'RidgeClassifierCV': {}
}

data_path = 'data/raw/ACME-HappinessSurvey2020.csv' 
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

mlflow_tracking_uri = 'http://localhost:5555'  
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Customer_Happiness")

def evaluate_model_with_gridsearch(name, model, grid, X_train, y_train, X_test, y_test):
    if grid:
        clf = GridSearchCV(model, grid, cv=5, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        best_params = clf.best_params_
    else:
        model.fit(X_train, y_train)
        best_model = model
        best_params = model.get_params()

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_scores = best_model.predict_proba(X_test)[:, 1]
    
    elif hasattr(best_model, "decision_function"):
        y_scores = best_model.decision_function(X_test)
    else:
        y_scores = best_model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_scores)

    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'model': best_model,
        'params': best_params
    }

print("MLflow tracking URI:", mlflow_tracking_uri)

results = {}

with mlflow.start_run(run_name="Customer_Happiness") if mlflow_tracking_uri else nullcontext():
    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True) if mlflow_tracking_uri else nullcontext():
            evaluation = evaluate_model_with_gridsearch(name, model, model_grids[name], X_train, y_train, X_test, y_test)
            results[name] = evaluation

            if mlflow_tracking_uri:
                mlflow.log_params(evaluation['params'])
                mlflow.log_metrics({
                    'accuracy': evaluation['accuracy'],  
                    'mae': evaluation['mae'],
                    'mse': evaluation['mse'],
                    'rmse': evaluation['rmse'],
                    'r2': evaluation['r2'],
                    'f1_score': evaluation['f1_score'],
                    'roc_auc': evaluation['roc_auc']
                })
                
            
            print(
            f"{name} Accuracy: {evaluation['accuracy']:.4f}, "
            f"R2: {evaluation['r2']:.4f}, "
            f"RMSE: {evaluation['rmse']:.2f}, "
            f"f1_score: {evaluation['f1_score']:.4f}, "
            f"ROC_AUC: {evaluation['roc_auc']:.4f}"
            )

import yaml
import os
# Save model config with selected features 
# Display information about the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_params = best_model.get_params()
best_accuracy = float(results[best_model_name]['accuracy'])
best_mae = float(results[best_model_name]['mae'])
best_rmse = float(results[best_model_name]['rmse'])
best_f1_score = float(results[best_model_name]['f1_score'])
best_roc_auc_score = float(results[best_model_name]['roc_auc'])

print(f"🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   f1 Score: {best_f1_score:.4f}")
print(f"   ROC_AUC: {best_roc_auc_score:.4f}")
print(f"   MAE: {best_mae:.4f}")
print(f"   RMSE: {best_rmse:.4f}")

model_config = {
    'model': {
        'name': 'happiness_model',
        'best_model': best_model_name,
        'parameters': best_params,
        'accuracy': best_accuracy,
        'mae': best_mae,
        'best_f1_score': best_f1_score,
        'best_roc_auc_score': best_roc_auc_score,
        'target_variable': 'Y',
        'feature_sets': selected_features_dict
    }
}

print(model_config)
config_path = '../../src/models/model_config.yaml'
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, 'w') as f:
    yaml.dump(model_config, f)

print(f"Saved model config to {config_path}")
