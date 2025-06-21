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

file_path = "data/ai_job_dataset.csv"

ai_job_dataset = load_data(file_path)
explore_data(ai_job_dataset)

