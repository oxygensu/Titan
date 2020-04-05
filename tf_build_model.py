from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns


# load train and test data
train_data = pd.read_csv('data/processed_train.csv')
test_data = pd.read_csv('data/processed_test.csv')

train_df = train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
# sns.pairplot(train_df[['Age_scaled', 'Fare_scaled', 'Pclass_1', 'Pclass_2', 'Pclass_3']])

# y equal to survival
y = train_np[:, 0]
# x equal to featrue
x = train_np[:, 1:]

train_x_dataset = tf.data.Dataset.from_tensor_slices((x))
train_y_dataset = tf.data.Dataset.from_tensor_slices((y))
test_df = test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.values

test_x = test_np[:, 0:]
test_dataset = tf.data.Dataset.from_tensor_slices((test_x))

# build a model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(14,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse']
                 )
    return model

model = build_model()
model.summary()

EPOCHS = 1000
history = model.fit(
    x, y,
    epochs=EPOCHS
)

## save model
model.save('model/tf_rms2.h5')

# test = pd.read_csv('data/test.csv')
# test = test.values
# test_passenger = test[:, 0]

# test_predictions = model.predict(test_dataset).flatten()
# result = pd.DataFrame({'PassengerId': test_data['PassengerId'].values, 'Survived':test_predictions.astype(np.int32)})
# result.to_csv("data/tf_predictions.csv", index=False)
