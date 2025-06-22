import pandas as pd
import sys
import os
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

from load_data import load_data
from explore_data import explore_data
from data_preparation import data_preparation, data_splitting
from feature_engineering import feature_engineering
from final_validation_and_summary import final_validation_and_summary
from handle_outliers import handle_outliers
from random_forest_model import random_forest_train, random_forest_optimization
from xgboost_model import xgboost_model_train, xgboost_model_optimization

file_path = "data/ai_job_dataset.csv"

ai_job_dataset = load_data(file_path)
explore_data(ai_job_dataset)
df_clean = data_preparation(ai_job_dataset)
X, y, final_df = feature_engineering(df_clean)
final_validation_and_summary(ai_job_dataset, df_clean, final_df, X, y)
# We skip it for now
#X, y = handle_outliers(X, y)
X_train, X_test, y_train, y_test = data_splitting(X, y)
random_forest_train(X_train, X_test, y_train, y_test)
random_forest_optimization(X_train, X_test, y_train, y_test)
xgboost_model_train(X_train, X_test, y_train, y_test)
xgboost_model_optimization(X_train, X_test, y_train, y_test)