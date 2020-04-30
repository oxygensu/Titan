import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl

# resovle matplotlib chinese show
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('data/origin/train.csv')
test_data = pd.read_csv('data/origin/test.csv')

print(train_data.info())
print("-" * 80)
print(test_data.info())

print("-" * 80)
print(train_data.head())

print("-" * 80)

sns.catplot(x="Pclass", hue="Survived", kind="count", data=train_data)

sns.catplot(x="Sex", hue="Survived", kind="count", data=train_data)

sns.catplot(x="Age", hue="Survived", kind="count", data=train_data)

sns.catplot(x="SibSp", hue="Survived", kind="count", data=train_data)
sns.catplot(x="SibSp", kind="count", data=test_data)

sns.catplot(x="Parch", hue="Survived", kind="count", data=train_data)

sns.catplot(x="Ticket", hue="Survived", kind="count", data=train_data)

sns.catplot(x="Fare", hue="Survived", kind="count", data=train_data)

sns.catplot(x="Embarked", hue="Survived", kind="count", data=train_data)

plt.show()