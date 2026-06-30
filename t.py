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

# In[9]:


# 4. Age imputation: median per Title + Pclass fallback
age_medians = full.groupby(['Title','Pclass'])['Age'].median()
def impute_age(row):
    if pd.isna(row['Age']):
        val = age_medians.get((row['Title'], row['Pclass']))
        if not np.isnan(val):
            return val
        return full['Age'].median()
    return row['Age']

full['Age'] = full.apply(impute_age, axis=1)



# In[10]:

# 5. Embarked: fill with mode
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# 6. Fare: fill missing (in test set) with median
full['Fare'] = full['Fare'].fillna(full['Fare'].median())

# 7. bins (optional)
full['AgeBin'] = pd.cut(full['Age'], bins=[0,12,20,40,60,120], labels=['child','teen','adult','mid','senior'])
full['FareBin'] = pd.qcut(full['Fare'], 4, labels=['low','med','high','vhigh'])

# 8. select features
# we'll use a set of engineered + original features
features = ['Pclass','Sex','Age','Fare','Embarked','Title','FamilySize','IsAlone','Deck','FarePerPerson']

# split back
train_feats = full.loc[:len(train)-1, features].copy()
test_feats  = full.loc[len(train):, features].copy()
y = train['Survived'].copy()

# In[11]:


# 9. build preprocessing pipelines
num_features = ['Age','Fare','FamilySize','FarePerPerson']
cat_features = ['Pclass','Sex','Embarked','Title','Deck','IsAlone']

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])





