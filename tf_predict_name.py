from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

model = keras.models.load_model('model/tf_name.h5')

test_data = pd.read_csv('data/pre_processing/processed_test3.csv')
test_df = test_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Title*')
test_np = test_df.values

test_x = test_np[:, 0:]


test_predictions = model.predict(test_x)

test = pd.read_csv('data/origin/test.csv')

pid = test.filter(regex='PassengerId').values
a = np.hstack((pid, test_predictions))
df = pd.DataFrame(a)
df.columns = ['a','b']
sns.lineplot(x = 'a', y = 'b', data=df)
import matplotlib.pyplot as plt
plt.show()

passenger_id = test.filter(regex='PassengerId').values
test_predictions[test_predictions>=1] = 1.0
test_predictions[test_predictions<1] = 0
survied = test_predictions.astype(np.int32)

result = pd.DataFrame({'PassengerId': passenger_id[:, 0], 'Survived':survied[:, 0]})
result.to_csv("data/predictions/tf_predictions_name.csv", index=False)