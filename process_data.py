import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# read train.csv
train_data = pd.read_csv('data/train.csv')

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
print("2\n")
print(df.describe())


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
print("3\n")
print(df.describe())