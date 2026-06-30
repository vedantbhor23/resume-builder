#!/usr/bin/env python
# coding: utf-8

# In[3]:


# titanic_pipeline.py
# Run in Kaggle or locally (pip install xgboost shap seaborn if missing)

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

git pull origin main --allow-unrelated-histories



# In[6]:


# 1. load
train = pd.read_csv(r"D:\ML_Project_Odd_Sem\train.csv")
test  = pd.read_csv(r"D:\ML_Project_Odd_Sem\test.csv")
test_passenger_ids = test['PassengerId'].copy()



# In[7]:

