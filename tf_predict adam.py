from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

model = keras.models.load_model('model/tf_adam4.h5')

test_data = pd.read_csv('data/processed_test.csv')
test_df = test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.values

test_x = test_np[:, 0:]

test = pd.read_csv('data/test.csv')

test_predictions = model.predict(test_x)

# passenger_id = test.filter(regex='PassengerId').values
# a = pd.DataFrame({'PassengerId': passenger_id[:, 0], 'Survived':test_predictions[:, 0]})
# sns.kdeplot(x = 'PassengerId', y = 'Survived', data=a)
# import matplotlib.pyplot as plt
# plt.show()

passenger_id = test.filter(regex='PassengerId').values
test_predictions[test_predictions>=0.5] = 1.0
test_predictions[test_predictions<0.5] = 0
survied = test_predictions.astype(np.int32)

result = pd.DataFrame({'PassengerId': passenger_id[:, 0], 'Survived':survied[:, 0]})
result.to_csv("data/tf_predictions_adam4.csv", index=False)