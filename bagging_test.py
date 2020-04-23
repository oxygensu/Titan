'''

'''

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from xgboost import XGBClassifier

# 1 Bagging法，使用sklearn.ensemble中的VotingClassifier
from sklearn.ensemble import VotingClassifier
# 2 Stacking
from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


# 
print("start")
#

# read train.csv and test.csv
train_data = pd.read_csv('data/pre_processing/processed_train4.csv')
test_data = pd.read_csv('data/pre_processing/processed_test4.csv')

# 用正则取出我们要的属性值
train_df = train_data
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3)
test_df = test_data
test_np = test_df.values

test_x = test_np[:, 0:]

lr = LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
DTree = DecisionTreeClassifier(max_depth=20)
rfc = RandomForestClassifier(n_estimators=5000, n_jobs=-1)
xgbc=XGBClassifier(learning_rate=0.001, n_estimators=5000, max_depth=20, objective='binary:logitraw')

clf_vc = VotingClassifier(estimators=[('lr', lr),('DTree', DTree),('RTree', rfc),('svm',SVC(probability=True)), ('xgb', xgbc)])
clf_vc.fit(X,y)
print(clf_vc.score(X, y))

test_id = pd.read_csv('data/origin/test.csv')

predictions = clf_vc.predict(test_x)
result = pd.DataFrame({'PassengerId':test_id['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("data/predictions/bagging_test3.csv", index=False)


answer = pd.read_csv('data/predictions/submission.csv')
answer_np = answer['Survived'].values
accuracy_score(answer_np, predictions)

# 
print("end")
#