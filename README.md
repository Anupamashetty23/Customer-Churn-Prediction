# Customer-Churn-Prediction


 # Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")  # Example dataset

# -----------------------------
# 2. Data Exploration
# -----------------------------
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# -----------------------------
# 3. Data Cleaning & Preprocessing
# -----------------------------
# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    if col != 'customerID':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Drop irrelevant columns
df.drop('customerID', axis=1, inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Churn', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('Churn', axis=1).columns)
y = df['Churn']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# 5. Baseline Model
# -----------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_baseline = log_reg.predict(X_test)

print("Baseline Logistic Regression Results:")
print(classification_report(y_test, y_pred_baseline))

# -----------------------------
# 6. Decision Tree Model
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Results:")
print(classification_report(y_test, y_pred_dt))

# Hyperparameter tuning for Decision Tree
param_grid_dt = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train, y_train)
print("Best Decision Tree Params:", grid_dt.best_params_)

# -----------------------------
# 7. Support Vector Machine (SVM)
# -----------------------------
svm = SVC(probability=True, random_state=42)
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_svm.fit(X_train, y_train)
print("Best SVM Params:", grid_svm.best_params_)

y_pred_svm = grid_svm.predict(X_test)
print("SVM Results:")
print(classification_report(y_test, y_pred_svm))

# -----------------------------
# 8. Model Evaluation
# -----------------------------
def evaluate_model(name, y_true, y_pred, y_prob=None):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    if y_prob is not None:
        print("ROC-AUC:", roc_auc_score(y_true, y_prob))

# Evaluate all models
evaluate_model("Logistic Regression", y_test, y_pred_baseline, log_reg.predict_proba(X_test)[:,1])
evaluate_model("Decision Tree", y_test, y_pred_dt, dt.predict_proba(X_test)[:,1])
evaluate_model("SVM", y_test, y_pred_svm, grid_svm.predict_proba(X_test)[:,1])

# -----------------------------
# 9. Visualization
# -----------------------------
# Confusion Matrix for best model (SVM in this case)
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# gvfadded full program
