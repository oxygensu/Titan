import pandas as pd
import numpy as np
import tensorflow as tf


# load train and test data
train_data = pd.read_csv('data/processed_train.csv')
test_data = pd.read_csv('data/processed_test.csv')

print(tf.__version__)