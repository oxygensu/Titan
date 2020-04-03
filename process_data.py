"""
This is learn from https://blog.csdn.net/han_xiaoyang/article/details/49797143
==================================

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing


def fill_age(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # split Passenger from know and nuknow age
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


def preprocessing_data(train_data):
    train_data, rfr = fill_age(train_data)
    train_data = set_Cabin_type(train_data)

    dummies_Cabin = pd.get_dummies(train_data['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(train_data['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(train_data['Pclass'], prefix= 'Pclass')
    df = pd.concat([train_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    age_reshaped = df['Age'].values.reshape(-1, 1)
    age_scale_param = scaler.fit(age_reshaped)
    df['Age_scaled'] = scaler.fit_transform(age_reshaped, age_scale_param)
    fare_reshaped = df['Fare'].values.reshape(-1, 1)
    fare_scale_param = scaler.fit(fare_reshaped)
    df['Fare_scaled'] = scaler.fit_transform(fare_reshaped, fare_scale_param)

    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    # y即Survival结果
    y = train_np[:, 0]
    # X即特征属性值
    X = train_np[:, 1:]
    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    clf.fit(X, y)
    return train_np, clf