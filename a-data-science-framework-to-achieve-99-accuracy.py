# import libraries
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import random
import time
import warnings
warnings.filterwarnings('ignore')

# load data modeling libraries
# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, \
naive_bayes, ensemble, discriminant_analysis,gaussian_process
from xgboost import XGBClassifier

# Common Model Helper
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Configure Visualization Defaults
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8
