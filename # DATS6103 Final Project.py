
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






# 2nd Smart Question
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

# 2. Bar Plot for Heavy Alcohol Consumption and High Blood Pressure
data['HeavyAlcohol'] = (data['HvyAlcoholConsump'] > 0).astype(int)  # Categorize heavy alcohol consumption
plt.figure(figsize=(10, 6))
sns.barplot(
    x='HeavyAlcohol',
    y='HighBP',
    data=data,
    estimator=lambda x: sum(x) / len(x),  # Proportion of HighBP
    ci=None,
    palette='coolwarm'
    )
plt.title("Proportion of High Blood Pressure by Heavy Alcohol Consumption")
plt.ylabel("Proportion of High Blood Pressure")
plt.xlabel("Heavy Alcohol Consumption (0 = No/Low, 1 = Heavy)")
plt.xticks(ticks=[0, 1], labels=["No/Low", "Heavy"])
plt.show()

# 3. Logistic Regression
predictors = ['Smoker', 'HeavyAlcohol', 'Age', 'Sex']
X_model = data[predictors]
y_model = data['HighBP']
scaler = StandardScaler()
X_model[['Age']] = scaler.fit_transform(X_model[['Age']])
X_model = sm.add_constant(X_model)

logit_model = sm.Logit(y_model, X_model).fit()

print(logit_model.summary())
coefficients = logit_model.params
odds_ratios = np.exp(coefficients)

print("\nOdds Ratios:")
for variable in ["Smoker", "HeavyAlcohol"]:
    if variable in odds_ratios.index: #Check if the variables are present
        print(f"{variable}: {odds_ratios[variable]:.3f}")
    else:
        print(f"{variable} is not found in the model.")

print("\nInterpretation of Odds Ratios:")
selected_variables = ["Smoker", "HeavyAlcohol"]
for variable, odds_ratio in odds_ratios.items():
    if variable in selected_variables:
        if odds_ratio > 1:
            print(f"{variable}: Increases the odds of HighBP by {round((odds_ratio - 1) * 100, 2)}%.")
        elif odds_ratio < 1:
            print(f"{variable}: Decreases the odds of HighBP by {round((1 - odds_ratio) * 100, 2)}%.")
        else:
            print(f"{variable}: Has no effect on the odds of HighBP.")

## Interpretation

## 1. Check the p-values: Variables with p-values < 0.05 are statistically significant.
## 2. Coefficients indicate the direction of the relationship (positive or negative).
## 3. Odds ratios can be calculated as exp(coefficients) for easier interpretation.

#################################################################################
#################################################################################
#################################################################################
#################################################################################

#################################################################################
# SMART QUESTION 3:
# How do BMI and physical activity, individually and in combination, influence the risk of developing diabetes and other chronic conditions like high blood pressure or heart disease?
#################################################################################

#%%
# Import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                           roc_curve, auc, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

#%% Data Loading and Preprocessing
def load_and_preprocess_data(filepath, chunk_size=10000, sample_fraction=0.1):
    """Load and preprocess the diabetes dataset"""
    sampled_chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
        sampled_chunks.append(sampled_chunk)
    
    return pd.concat(sampled_chunks, ignore_index=True)

#%% Statistical Analysis Functions
def analyze_bmi_activity_relationship(data):
    """Analyze the relationship between BMI and physical activity"""
    correlation = stats.pearsonr(data['BMI'], data['PhysActivity'])
    print(f"Correlation between BMI and Physical Activity:")
    print(f"Correlation coefficient: {correlation[0]:.3f}")
    print(f"P-value: {correlation[1]:.3f}\n")

    data['BMI_Category'] = pd.cut(data['BMI'], 
                                 bins=[0, 18.5, 24.9, 29.9, float('inf')],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    contingency_table = pd.crosstab(data['BMI_Category'], data['PhysActivity'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for BMI categories and Physical Activity:")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"P-value: {p_value:.3f}\n")
    
    return data['BMI_Category']

def analyze_health_outcomes(data):
    """Analyze relationships between variables and health outcomes"""
    diabetes_groups = [group for _, group in data.groupby('Diabetes_012')['BMI']]
    f_stat, p_value = stats.f_oneway(*diabetes_groups)
    print(f"ANOVA test for BMI across diabetes groups:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.3f}\n")
    
    X_health = data[['BMI', 'PhysActivity']]
    y_diabetes = data['Diabetes_012']
    
    for feature in ['BMI', 'PhysActivity']:
        log_reg = LogisticRegression(multi_class='multinomial', max_iter=1000)
        log_reg.fit(X_health[[feature]], y_diabetes)
        score = log_reg.score(X_health[[feature]], y_diabetes)
        print(f"Individual effect of {feature} on Diabetes (Accuracy): {score:.3f}")

def calculate_risk_ratios(data):
    """Calculate risk ratios for different combinations of BMI and Physical Activity"""
    median_bmi = data['BMI'].median()
    median_activity = data['PhysActivity'].median()
    
    data['High_BMI'] = data['BMI'] > median_bmi
    data['High_Activity'] = data['PhysActivity'] > median_activity
    data['Has_Diabetes'] = data['Diabetes_012'] > 0
    
    groups = data.groupby(['High_BMI', 'High_Activity'])['Has_Diabetes'].agg(['mean', 'size'])
    print("\nRisk Ratios for Diabetes:")
    print(groups)
    
    baseline_risk = groups.loc[(False, False), 'mean']
    for idx in groups.index:
        relative_risk = groups.loc[idx, 'mean'] / baseline_risk
        print(f"\nRelative Risk for BMI={idx[0]}, Activity={idx[1]}: {relative_risk:.2f}")

#%% Visualization Functions
def plot_model_performance(results_df):
    """Plot model performance comparison"""
    bar_fig = go.Figure(data=[
        go.Bar(name="Accuracy", x=results_df.index, y=results_df["Accuracy"], marker_color="blue"),
        go.Bar(name="AUC", x=results_df.index, y=results_df["AUC"], marker_color="orange")
    ])
    
    bar_fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode="group",
        template="plotly_white"
    )
    bar_fig.show()

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    confusion_mat = confusion_matrix(y_true, y_pred)
    cm_fig = px.imshow(
        confusion_mat,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[0, 1, 2],
        y=[0, 1, 2]
    )
    cm_fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white"
    )
    cm_fig.show()

def plot_roc_curves(y_test, X_test_scaled, best_model):
    """Plot ROC curves for each class"""
    fig = go.Figure()
    colors = ['blue', 'red', 'green']
    
    for i, color in enumerate(colors):
        if hasattr(best_model, 'predict_proba'):
            y_true_binary = (y_test == i).astype(int)
            y_score = best_model.predict_proba(X_test_scaled)[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                   name=f'Class {i} (AUC = {roc_auc:.2f})',
                                   line=dict(color=color)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                            name='Random',
                            line=dict(color='black', dash='dash')))
    
    fig.update_layout(
        title='ROC Curves for Each Class',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white'
    )
    fig.show()

#%% Model Training Functions
def initialize_models(class_weights_dict):
    """Initialize all models with parameters"""
    param_grid_xgb = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    param_grid_lgbm = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "num_leaves": [20, 31, 50],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.8, 1.0]
    }

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, 
                                                class_weight=class_weights_dict),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            param_grid_xgb,
            scoring="roc_auc_ovr",
            cv=3,
            n_jobs=-1
        ),
        "LightGBM": GridSearchCV(
            LGBMClassifier(random_state=42),
            param_grid_lgbm,
            scoring="roc_auc_ovr",
            cv=3,
            n_jobs=-1
        )
    }

#%% Main Execution
def main():
    """Main execution function for diabetes analysis"""
    # 1. Load data
    print("Loading and preprocessing data...")
    dataset_path = "diabetes_012_health_indicators_BRFSS2015.csv"
    sampled_data = load_and_preprocess_data(dataset_path)
    
    # 2. Define features and target
    features = ['BMI', 'PhysActivity', 'HighBP', 'HeartDiseaseorAttack', 'HighChol', 'CholCheck']
    target = 'Diabetes_012'
    
    X = sampled_data[features]
    y = sampled_data[target]
    
    # 3. Split and scale data
    print("\nSplitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Handle class imbalance
    print("\nHandling class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 5. Compute class weights
    class_weights = compute_class_weight("balanced", classes=y.unique(), y=y)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # 6. Run statistical analyses
    print("\nPerforming statistical analyses...")
    analyze_bmi_activity_relationship(sampled_data)
    analyze_health_outcomes(sampled_data)
    calculate_risk_ratios(sampled_data)
    
    # 7. Initialize and train models
    print("\nTraining models...")
    models = initialize_models(class_weights_dict)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in ["Logistic Regression", "Random Forest"]:
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.best_estimator_.predict(X_test_scaled)
            y_pred_proba = model.best_estimator_.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        results[name] = {"Accuracy": accuracy, "AUC": auc_score}
        
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # 8. Create and display visualizations
    print("\nGenerating visualizations...")
    results_df = pd.DataFrame(results).T
    results_df.sort_values(by="AUC", ascending=False, inplace=True)
    
    print("\nModel Performance Summary:")
    print(results_df)
    
    # Plot model performance
    plot_model_performance(results_df)
    
    # Get best model
    best_model_name = results_df.index[0]
    if best_model_name in ["Logistic Regression", "Random Forest"]:
        best_model = models[best_model_name]
    else:
        best_model = models[best_model_name].best_estimator_
    
    # Plot confusion matrix for best model
    y_pred_best = best_model.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred_best, f"Confusion Matrix for {best_model_name}")
    
    # Plot ROC curves
    plot_roc_curves(y_test, X_test_scaled, best_model)
    
    # Feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature Importance:")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(importance)
        
        fig = px.bar(importance, x='Feature', y='Importance',
                    title='Feature Importance Analysis')
        fig.update_layout(template='plotly_white')
        fig.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

# %%
