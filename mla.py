"""
    MLA : machine learning algorithm
"""
#
import pandas as pd

# common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import seaborn as sns

MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Process
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Tree
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost
    XGBClassifier()
]

# cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
cv_split = model_selection.ShuffleSplit(n_splits=10, train_size=.6, test_size=.3, random_state=0)

MLA_colums = ["MLA Name", "MLA Parameters", "MLA Train Accuracy Mean", "MLA Test Accuracy Mean", "MLA Test Accuracy 3*STD", "MLA Time"]
MLA_compare = pd.DataFrame(columns=MLA_colums)

data1 = pd.read_csv("data/pre_processing/processed_train5.csv")
X = data1.drop(["PassengerId", "Survived"], axis=1).values
Y = data1[["Survived"]].values

MLA_predict = Y

row_index = 0

for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, "MLA Name"] = MLA_name
    MLA_compare.loc[row_index, "MLA Parameters"] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, X, Y, cv=cv_split, return_train_score=True)

    MLA_compare.loc[row_index, "MLA Time"] = cv_results["fit_time"].mean()
    MLA_compare.loc[row_index, "MLA Train Accuracy Mean"] = cv_results["train_score"].mean()
    MLA_compare.loc[row_index, "MLA Test Accuracy Mean"] = cv_results["test_score"].mean()
    MLA_compare.loc[row_index, "MLA Test Accuracy 3*STD"] = cv_results["test_score"].std() * 3
    
    #alg.fit(X, Y)
    #MLA_predict[MLA_name] = alg.predict(X)

    row_index += 1

MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
print(MLA_compare)
sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')
plt.show()