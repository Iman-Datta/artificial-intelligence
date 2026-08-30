import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data: pd.DataFrame = pd.read_csv("diabetes.csv")

print("First 5 rows of the dataset:")
print(data.head())

# SEPARATE INPUT FEATURES (X) AND TARGET (y)
# axis=1 means we are working with columns
X: pd.DataFrame = data.drop("Outcome", axis=1)

# y = target/output that we want to predict
y: pd.Series = data["Outcome"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# SPLIT DATA INTO TRAINING AND TESTING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("\nTraining data size:", X_train.shape)
print("\nTesting data size:",X_test.shape)

log_reg = LogisticRegression(max_iter=1000)

# TRAIN THE MODEL
log_reg.fit(X_train, y_train)

# Check what the  MODEL LEARNED
# coef_ contains the coefficients/weights
print("\nCoefficients (Weights):", log_reg.coef_)

# intercept_ contains the bias/intercept
print("\nIntercept (Bias):", log_reg.intercept_)

# MAKE PREDICTIONS
y_pred = log_reg.predict(X_test)
print("\nPredicted values:", y_pred)
