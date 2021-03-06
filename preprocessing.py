import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

org_train_data = pd.read_csv('data/origin/train.csv')
org_test_data = pd.read_csv('data/origin/test.csv')

#### drop Ticket and Cabin
train_data = org_train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = org_test_data.drop(['Ticket', 'Cabin'], axis=1)


def onehot(df):
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_SibSp = pd.get_dummies(df['SibSp'], prefix='SibSp')
    dummies_Parch = pd.get_dummies(df['Parch'], prefix='Parch')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    df = pd.concat([df, dummies_Pclass, dummies_Sex, dummies_SibSp, dummies_Parch, dummies_Embarked], axis=1)

    return df

train_data = onehot(train_data)
test_data = onehot(test_data)

#### Fare standscaler
scaler = StandardScaler()
fare = train_data['Fare'].values.reshape(-1, 1)
scaler.fit(fare)
train_data['Fare_scaler'] = scaler.transform(fare)

# imputer Fare
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
test_data['Fare'] = imputer.fit_transform(test_data['Fare'].values.reshape(-1,1))

scaler = StandardScaler()
fare = test_data['Fare'].values.reshape(-1, 1)
scaler.fit(fare)
test_data['Fare_scaler'] = scaler.transform(fare)

#### get title  
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles(train_data):
    # 对Name属性做提取
    train_data['Title'] = train_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # 对Name做映射
    train_data['Title'] = train_data.Title.map(Title_Dictionary)
    return train_data
#### Name特征处理
def process_names(train_data):
    # Name已经提取出我们有用的信息 删除Name列
    train_data.drop('Name', axis=1, inplace=True)
    
    # one-hot编码 将提取的特征转换成向量的形式
    titles_dummies = pd.get_dummies(train_data['Title'], prefix='Title')
    train_data = pd.concat([train_data, titles_dummies], axis=1)
    
    # 删除Title字段
    train_data.drop('Title', axis=1, inplace=True)
    return train_data

train_data = get_titles(train_data)
train_data = process_names(train_data)
test_data = get_titles(test_data)
test_data = process_names(test_data)

print(train_data.info())
print(test_data.info())

#### get known age df and unknow age
def age_fill(df):

    df = df.filter(regex="Age|Pclass_.*|Sex_.*|SibSp_.*|Parch_.*|Embarked_.*|Title_.*|Fare_.*")

    age_known = df[df.Age.notnull()].values
    age_unknown = df[df.Age.isnull()].values

    y = age_known[:, 0]
    X = age_known[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, max_depth=29, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(age_unknown[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    #### Age standscaler
    scaler = StandardScaler()
    age = df['Age'].values.reshape(-1, 1)
    scaler.fit(age)
    df['Age_scaler'] = scaler.transform(age)
    return df

train_data = age_fill(train_data)
test_data = age_fill(test_data)

train_data = train_data.drop("Age", axis=1)
test_data = test_data.drop("Age", axis=1)

# save train and test data to csv
train_result = pd.concat([org_train_data[['PassengerId', 'Survived']], train_data], axis=1)
train_result.to_csv("data/pre_processing/processed_train5.csv", index=False)

test_result = pd.concat([org_test_data['PassengerId'], test_data], axis=1)
test_result.to_csv("data/pre_processing/processed_test5.csv", index=False)