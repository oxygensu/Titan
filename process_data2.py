'''
https://kandianshare.html5.qq.com/v2/news/4335784489645201730?from_app=qb_10&url=http%3A%2F%2Fkuaibao.qq.com%2Fs%2F20200311A05EPM00&cardmode=1&dataSrc=96&docId=4335784489645201730&pid=1&queryId=1586053682342&sh_sid=5__16dedfa7282341ba__c40f6ba72a1ae97a7ede750a1f5988cb&subjectId=1090319&zimeitiId=qeh_17342458#
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.ensemble import RandomForestClassifier


# resovle matplotlib chinese show
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# drop id cabin name ticket
train_data = train_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_data = test_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

dataset = [train_data, test_data]


for eachrow in dataset:
    # fill age null
    mean = train_data['Age'].mean()
    std = test_data['Age'].std()
    is_null = eachrow['Age'].isnull().sum()
    rand_age = np.random.randint(mean-std,mean+std,size=is_null)

    age_slice = eachrow['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    eachrow['Age'] = age_slice
    eachrow['Age'] = train_data['Age'].astype(int)

    common_values = 'S'
    eachrow['Embarked'] = eachrow['Embarked'].fillna(common_values)

genders = {'male':0,'female':1}
ports = {'S':0, 'C':1, 'Q':2}

for eachrow in dataset:
    eachrow['Fare'] = eachrow['Fare'].fillna(0)
    eachrow['Fare'] = eachrow['Fare'].astype(int)

    eachrow['Sex'] = eachrow['Sex'].map(genders)

    eachrow['Embarked'] = eachrow['Embarked'].map(ports)

for eachrow in dataset:
    eachrow['Age'] = eachrow['Age'].astype(int)
    eachrow.loc[eachrow['Age']<=11, 'Age'] = 0
    eachrow.loc[(eachrow['Age']>11) & (eachrow['Age']<=18), 'Age'] = 1
    eachrow.loc[(eachrow['Age']>18) & (eachrow['Age']<=22), 'Age'] = 2
    eachrow.loc[(eachrow['Age']>22) & (eachrow['Age']<=27), 'Age'] = 3
    eachrow.loc[(eachrow['Age']>27) & (eachrow['Age']<=33), 'Age'] = 4
    eachrow.loc[(eachrow['Age']>33) & (eachrow['Age']<=40), 'Age'] = 5
    eachrow.loc[(eachrow['Age']>40) & (eachrow['Age']<=66), 'Age'] = 6
    eachrow.loc[eachrow['Age']>66, 'Age']=6

    eachrow.loc[eachrow['Fare']<=7.91, 'Fare'] = 0
    eachrow.loc[(eachrow['Fare']>11) & (eachrow['Fare']<=14.454), 'Fare'] = 1
    eachrow.loc[(eachrow['Fare']>14.454) & (eachrow['Fare']<=31), 'Fare'] = 2
    eachrow.loc[(eachrow['Fare']>32) & (eachrow['Fare']<=99), 'Fare'] = 3
    eachrow.loc[(eachrow['Fare']>99) & (eachrow['Fare']<=32503), 'Fare'] = 4
    eachrow.loc[eachrow['Fare']>250, 'Fare']=5
    eachrow['Fare'] = eachrow['Fare'].astype(int)

train_x = train_data.drop('Survived', axis=1)
train_y = train_data['Survived']
test_x = test_data

# use random forest
random_forest = RandomForestClassifier(n_estimators=100)

# save processed data
df = pd.DataFrame(train_data)
df.to_csv('data/processed_train2.csv')
test_x.to_csv('data/processed_test2.csv')

# random_forest.fit(train_x, train_y)

# prediction_y = random_forest.predict(test_x)

# test_data = pd.read_csv('data/test.csv')
# result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':prediction_y.astype(np.int32)})
# result.to_csv("data/random_forest_predictions.csv", index=False)

# print(random_forest.score(train_x, train_y))