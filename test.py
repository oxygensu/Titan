import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")
sns.relplot(x='timepoint', y='signal', kind='line', data=fmri)
sns.relplot(x='timepoint', y='signal', data=fmri)
plt.show()
