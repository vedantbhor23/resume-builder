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




# 2. helper feature functions
def extract_title(name):
    m = re.search(r',\s*([^.]*)\.', name)
    if m:
        return m.group(1).strip()
    return 'Unknown'

def simplify_title(title):
    title = title.lower()
    if title in ['mr','mrs','miss','master']:
        return title
    # map similar variants
    if 'mrs' in title: return 'mrs'
    if 'miss' in title: return 'miss'
    if 'master' in title: return 'master'
    if 'mr' in title: return 'mr'
    return 'rare'

# combine for consistent transforms
full = pd.concat([train.drop(columns='Survived'), test], axis=0, ignore_index=True)



# In[8]:


# 3. feature engineering
full['Title'] = full['Name'].apply(extract_title).apply(simplify_title)
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
full['Deck'] = full['Cabin'].fillna('U').apply(lambda x: str(x)[0] if x != 'U' else 'U')
# Fare per person (in case of shared tickets)
full['FarePerPerson'] = full['Fare'] / full['FamilySize']




