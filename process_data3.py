'''
https://kandianshare.html5.qq.com/v2/news/4335784489645201730?from_app=qb_10&url=http%3A%2F%2Fkuaibao.qq.com%2Fs%2F20200311A05EPM00&cardmode=1&dataSrc=96&docId=4335784489645201730&pid=1&queryId=1586053682342&sh_sid=5__16dedfa7282341ba__c40f6ba72a1ae97a7ede750a1f5988cb&subjectId=1090319&zimeitiId=qeh_17342458#

and add name feature
https://zhuanlan.zhihu.com/p/46772301
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

train_data = pd.read_csv('data/origin/train.csv')
test_data = pd.read_csv('data/origin/test.csv')

# name feature
titles = set()
for name in train_data['Name']:
    title = name.split(',')[1].split('.')[0].strip()
    
#map 映射将Name特征映射成我们需要的特征
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
print('AA')
#
def get_titles(train_data):
    # 对Name属性做提取
    train_data['Title'] = train_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # 对Name做映射
    train_data['Title'] = train_data.Title.map(Title_Dictionary)
    return train_data
#Name特征处理
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

# read org processed train data and insert new data 
org_train_data = pd.read_csv('data/pre_processing/processed_train.csv')
org_test_data = pd.read_csv('data/pre_processing/processed_test.csv')
new_train_data = org_train_data.join(train_data.filter(regex='Title*'))
new_test_data = org_test_data.join(test_data.filter(regex='Title*'))

new_train_data.to_csv('data/pre_processing/processed_train3.csv')
new_test_data.to_csv('data/pre_processing/processed_test3.csv')

print('done')