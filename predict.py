import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

# read train.csv and test.csv
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


def fill_age(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr


def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


print("0:\n")
print(train_data.describe())

train_data, rfr = fill_age(train_data)
train_data = set_Cabin_type(train_data)
print("1:\n")
print(train_data.describe())


dummies_Cabin = pd.get_dummies(train_data['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(train_data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train_data['Pclass'], prefix= 'Pclass')
df = pd.concat([train_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print("2:\n")
print(df.describe())



scaler = preprocessing.StandardScaler()

age_reshaped = df['Age'].values.reshape(-1, 1)
age_scale_param = scaler.fit(age_reshaped)
df['Age_scaled'] = scaler.fit_transform(age_reshaped, age_scale_param)

fare_reshaped = df['Fare'].values.reshape(-1, 1)
fare_scale_param = scaler.fit(fare_reshaped)
df['Fare_scaled'] = scaler.fit_transform(fare_reshaped, fare_scale_param)
print("3:\n")
print(df.describe())


# train a model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
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

# test data processing
test_data.loc[ (test_data.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = test_data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[test_data.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
test_data.loc[ (test_data.Age.isnull()), 'Age' ] = predictedAges

test_data = set_Cabin_type(test_data)
dummies_Cabin = pd.get_dummies(test_data['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test_data['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(test_data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test_data['Pclass'], prefix= 'Pclass')


df_test = pd.concat([test_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
age_reshaped = df_test['Age'].values.reshape(-1, 1)
age_scale_param = scaler.fit(age_reshaped)
df_test['Age_scaled'] = scaler.fit_transform(age_reshaped, age_scale_param)
fare_reshaped = df_test['Fare'].values.reshape(-1, 1)
fare_scale_param = scaler.fit(fare_reshaped)
df_test['Fare_scaled'] = scaler.fit_transform(fare_reshaped, fare_scale_param)


test=df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# use logistic regression
predictions = model_LogisticRegression.predict(test)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("data/logistic_regression_predictions.csv", index=False)

# use tensorflow cnn
