import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

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

# fit到RandomForestRegressor之中
model_LogisticRegression = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
model_LogisticRegression.fit(X, y)
print('model:\n')
print(model_LogisticRegression)

test_df = test_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title*')
test_np = test_df.values

test_data = pd.read_csv('data/origin/test.csv')

# use logistic regression
predictions = model_LogisticRegression.predict(test_np)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("data/predictions/logistic_regression_predictions_name.csv", index=False)
