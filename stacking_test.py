'''

'''

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

# read train.csv and test.csv
train_data = pd.read_csv('data/pre_processing/processed_train3.csv')
test_data = pd.read_csv('data/pre_processing/processed_test3.csv')

# 用正则取出我们要的属性值
train_df = train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title*')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3)
test_df = test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title*')
test_np = test_df.values
test = test_np[:, 0:]

# def stacking(train_x, train_y, test)
