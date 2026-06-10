---
title: Customer Happiness
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Customer Happiness Project

The objective of the goal is to predict customers' happiness based on the following features of the logistic and delivery company. This is a supervised system, which means that the output of the model is labelled. The labels consist of two binary values, either 0 (unhappy) and 1 (happy). There are 6 variables or features, ranging from 1 to 5 and are defined as the following:

Data Description:

Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers

X1 = my order was delivered on time

X2 = contents of my order was as I expected

X3 = I ordered everything I wanted to order

X4 = I paid a good price for my order

X5 = I am satisfied with my courier

X6 = the app makes ordering easy for me

Exploratory Data Analysis & Preprocessing

This section is dedicated to uncovering different proprocessing methods to either remove outliers or replace them with the median value of the data points. I used a box-and-whisker plot to identify those outliers, which result in the skewness of the data (either left-skewed or right-skewed). Therefore, the approach I used is to replace the outliers with median of the column and the histogram later turned out more evenly distributed and showed less skewness. I also plotted a correlation chart, which shows the correlation for every paired combination between all the predictors and the label. Here you can see, as compared with Y, X1 has the greatest correlation (0.25), followed by X5 (0.18) and X6 (0.19). This later confirms our findings for feature selection, in which this combination (X1, X5, X6) gives the most accurate set of predictors for Y. 

Training models

The mission here is to train the dataset with all the possible machine learning algorithms. i also used a classifier package called LazyClassifier, which consists of different machine learning algorithms that are not included in the classifiers I have defined manually. 

However, the manually defined classifiers I was able to capture here are:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- LGBM Classifier
- Bernoulli NB
- Gaussian NB
- SGD Classifier
- NearestCentroid
- Perception
- Linear Discriminant Analysis
- Ridge Classifier
- Ridge Classifier CV
- XGBoostClassifier

Feature Selection
I have used the xgboost model with binary:logistic as the objective and an assigned random state of 42. Here, I specified the number of features to 3 and the result used further along to train the model with the best selected features. 

As you can see from the model_config.yaml file, the best features are [X1, X5, X6]. This corresponds with the earlier prediction we had in EDA. As part of the preprocessing, I imported StandardScaler() and then fit-transformed it into input data, which is fed into the model.

I used mlflow package to track the different runs and it is hosted on the 5555 port. It is convenient for tracking each run and stores the metrics in a centralized location. I have defined a function called evaluate_model_with_grid which takes a model and determines the best hyperparameters for the said model and records its accuracy, mae, mse, rmse, r2, f1_score, and roc_auc. Later, the model with the highest accuracy is selected and stored in a .joblib file called best_model.joblib and its configuration in a .yaml file called model_config.yaml. 

Here is the outcome of the evaluation of different machine learning algorithms.

Classifier	accuracy	mae	mse	rmse	r2	f1_score	roc_auc	model	params
0	LogisticRegression	0.653846	0.346154	0.346154	0.588348	-0.392857	0.727273	0.770833	LogisticRegression()	{'C': 1.0, 'class_weight': None, 'dual': False...
1	RandomForestClassifier	0.769231	0.230769	0.230769	0.480384	0.071429	0.8125	0.758929	(DecisionTreeClassifier(max_depth=5, max_featu...	{'max_depth': 5, 'n_estimators': 100}
2	GradientBoosting	0.769231	0.230769	0.230769	0.480384	0.071429	0.8125	0.75	([DecisionTreeRegressor(criterion='friedman_ms...	{'learning_rate': 0.2, 'max_depth': 5, 'n_esti...
3	XGBoostClassifier	0.807692	0.192308	0.192308	0.438529	0.22619	0.827586	0.83631	XGBClassifier(base_score=None, booster=None, c...	{'colsample_bytree': 0.8, 'learning_rate': 0.1...
4	LGBMClassifier	0.769231	0.230769	0.230769	0.480384	0.071429	0.785714	0.83631	LGBMClassifier(learning_rate=0.05, max_depth=3...	{'learning_rate': 0.05, 'max_depth': 3, 'n_est...
5	SGDClassifier	0.576923	0.423077	0.423077	0.650444	-0.702381	0.666667	0.491071	SGDClassifier(loss='log_loss', random_state=42)	{'alpha': 0.0001, 'loss': 'log_loss'}
6	BernoulliNB	0.692308	0.307692	0.307692	0.5547	-0.238095	0.733333	0.747024	BernoulliNB(alpha=0.1)	{'alpha': 0.1}
7	GaussianNB	0.653846	0.346154	0.346154	0.588348	-0.392857	0.709677	0.800595	GaussianNB()	{'priors': None, 'var_smoothing': 1e-09}
8	NearestCentroid	0.653846	0.346154	0.346154	0.588348	-0.392857	0.689655	0.78869	NearestCentroid()	{'metric': 'euclidean', 'priors': 'uniform', '...
9	Perceptron	0.730769	0.269231	0.269231	0.518875	-0.083333	0.787879	0.8125	Perceptron(penalty='l2', random_state=42)	{'alpha': 0.0001, 'penalty': 'l2'}
10	LinearDiscriminantAnalysis	0.653846	0.346154	0.346154	0.588348	-0.392857	0.727273	0.770833	LinearDiscriminantAnalysis()	{'covariance_estimator': None, 'n_components':...
11	RidgeClassifier	0.653846	0.346154	0.346154	0.588348	-0.392857	0.727273	0.770833	RidgeClassifier(alpha=0.1)	{'alpha': 0.1}
12	RidgeClassifierCV	0.653846	0.346154	0.346154	0.588348	-0.392857	0.727273	0.770833	RidgeClassifierCV()	{'alphas': (0.1, 1.0, 10.0), 'class_weight': N...


Creating the API

I created a directory src/api/ in which I stored four different .py files used to create the API:

Inference.py : contains definition of method predict_happiness, which takes the 3 variables [X1, X5, X6], scale the input, and feed this into the best model. Then, a PredictionResponse is returned with the happiness value.

main.py : initializes FastAPI with metadata, add the CORS middleware. Then, I configured the health check and prediction endpoints. 

Schemas.py: initializes CustomerHappinessRequest with different fields and PredictionResponse with predicted_happiness.

Continuous Integration and Continuous Deployment

I stored this yaml file in .github/workflows/ and this is triggered automatically when there is any pull or push request. Once CICD pipeline finishes running, one can find an interactive UI in which one can adjust the different values for X1 through X6 and receive an output signaling either 0 or 1.

Conclusion:
It was determined that the best set of features to predict a customer's happiness is [X1, X5, X6] and the best machine learning algorithm is xgboost classifier, which gives an accuracy level of approximately 0.81 and f1 score of 0.83.
