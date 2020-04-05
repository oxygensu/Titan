import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl

# resovle matplotlib chinese show
mpl.rcParams['font.sans-serif'] = ['STFangsong']
mpl.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

sns.heatmap(train_data.isnull(), yticklabels=False, 
cbar=False, cmap='viridis')
plt.tight_layout()


survived = '存活'
not_survived = '死亡'

# female survived/not survived histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
women = train_data[train_data['Sex']=='female']
men = train_data[train_data['Sex']=='man']
ax = sns.distplot(
    women[women['Survived']==1].Age.dropna(),
    bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.distplot(
    women[women['Survived']==0].Age.dropna(),
    bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title('女性')

# men survived/not survived histogram
ax = sns.distplot(
    men[men['Survived']==1].Age.dropna(),
    bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.distplot(
    men[men['Survived']==0].Age.dropna(),
    bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
_ = ax.set_title('男性')

plt.show()