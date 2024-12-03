# DATS6103 Final Project 
# pip install ucimlrepo
#%%
import pandas as pd
import csv
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables)
# %%
# Check null values
rows = len(y)
missing = (y.isnull().sum() / rows * 100).to_frame('percentage of missing values')
print(missing)



# Junhua Deng - Building different models
# %%
overall = pd.concat([y,X])
overall.describe()
# %%
# heatmap
from statsmodels.stats.outliers_influence import variance_inflation_factor
correlation_matrix = X.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# %%
# Plot ROC curve
def rocplot(fpr, tpr, roc_auc, name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='dashdot')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")

# %%
# Logistic Regression model
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=39, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(model.coef_[0]),
    'Raw_Coefficient': model.coef_[0]
})

# Sort by absolute coefficient value
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

# Calculate p-values using statsmodels for statistical significance
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)
results = logit_model.fit()

# Add p-values to feature importance DataFrame
feature_importance['P_Value'] = results.pvalues[1:]  # Skip the constant term
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

# Print model performance
y_pred = model.predict(X_test_scaled)
print("\nModel Performance:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print feature importance with significance levels
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance in Diabetes Prediction')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()



# %%
# Select variables with absolute coefficient > 0.1
columns = ['GenHlth', 'BMI', 'Age','HighBP','HighChol','CholCheck','HvyAlcoholConsump',
           'Sex','Income']
X_new = X[columns]
X_new.describe()

#%%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=39)

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model_2 = LogisticRegression(random_state=39, max_iter=1000)
model_2.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_new.columns,
    'Coefficient': np.abs(model_2.coef_[0]),
    'Raw_Coefficient': model_2.coef_[0]
})

X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)
results_2 = logit_model.fit()

# Add p-values to feature importance DataFrame
feature_importance['P_Value'] = results_2.pvalues[1:]  # Skip the constant term
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)

# %%
# Calculate ROC curve and AUC
y_pred_proba = model_2.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

lr_plot = rocplot(fpr, tpr,roc_auc, 'ROC Curve of Logistic Regression Model')

# Print model performance metrics
y_pred = model_2.predict(X_test_scaled)
print("\nModel Performance:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# Create odds ratios
feature_importance['Odds_Ratio'] = np.exp(feature_importance['Raw_Coefficient'])
print("\nOdds Ratios:")
print(feature_importance[['Feature', 'Odds_Ratio', 'P_Value']].to_string(index=False))

print(f"\nAUC Score: {roc_auc:.3f}")

# %%
#Print model parameters
print('Intercept: \n', model_2.intercept_)
print('Coefficients: \n', np.exp(model_2.coef_))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# %%
# Random Forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
# Create and train model
rf = RandomForestRegressor(n_estimators=100, random_state=39)
rf.fit(X_train, y_train)

# Get predictions
y_pred = rf.predict(X_test)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
rf_plot = rocplot(fpr, tpr,roc_auc, 'ROC Curve of Random Forest Model')

# %%
from sklearn.neighbors import KNeighborsClassifier
# Initialize and train KNN model
# Using square root of n as rule of thumb for n_neighbors
n_neighbors = int(np.sqrt(len(X_train)))
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)

# Get predictions and probabilities
y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
knn_plot = rocplot(fpr, tpr,roc_auc, 'ROC Curve of KNN Model')

# %%
import xgboost as xgb
# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.1,
    max_depth=4,
    n_estimators=100,
    random_state=7
)

# Train the model
xgb_model.fit(
    X_train_scaled, 
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric='auc',
    early_stopping_rounds=20,
    verbose=False
)

# Get predictions and probabilities
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
xgboost_plot = rocplot(fpr, tpr,roc_auc, 'ROC Curve of XGBoost Model')
# %%






# 2nd Smart Question - siddharth
# Which habit causes more risk in high blood pressure? Consuming alcohol or smoking?
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Combine features with target variable for easier analysis
data = X.copy()
data['HighBP'] = y
required_columns = ['Smoker', 'HvyAlcoholConsump', 'Age', 'Sex']
# 1. Bar Plot for Smoking and High Blood Pressure
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Smoker',
    y='HighBP',
    data=data,
    estimator=lambda x: sum(x) / len(x),  # Proportion of HighBP
    ci=None,
    palette='Set2'
    )
plt.title("Proportion of High Blood Pressure by Smoking Status")
plt.ylabel("Proportion of High Blood Pressure")
plt.xlabel("Smoking Status (0 = Non-Smoker, 1 = Smoker)")
plt.xticks(ticks=[0, 1], labels=["Non-Smoker", "Smoker"])
plt.show()
