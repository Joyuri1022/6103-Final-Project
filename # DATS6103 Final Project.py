# DATS6103 Final Project 
#%%
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
import string
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
rows = len(X)
missing = (X.isnull().sum() / rows * 100).to_frame('percentage of missing values')
print(missing)
# %%
