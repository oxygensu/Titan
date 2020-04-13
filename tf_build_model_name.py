from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns


# load train and test data
train_data = pd.read_csv('data/pre_processing/processed_train3.csv')
test_data = pd.read_csv('data/pre_processing/processed_test3.csv')

train_df = train_data.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*')
train_np = train_df.values
# sns.pairplot(train_df[['Age_scaled', 'Fare_scaled', 'Pclass_1', 'Pclass_2', 'Pclass_3']])

# y equal to survival
y = train_np[:, 0]
# x equal to featrue
x = train_np[:, 1:]

# split train and validation dataset
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2)


train_x_dataset = tf.data.Dataset.from_tensor_slices((x))
train_y_dataset = tf.data.Dataset.from_tensor_slices((y))



# build a model
def build_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(20,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(0.01)

    model.compile(loss='MSE',
                  optimizer=optimizer,
                  metrics=['accuracy']
                 )
    return model

model = build_model()
model.summary()

EPOCHS = 1000
history = model.fit(
    train_x, train_y,
    validation_data = (valid_x, valid_y),
    validation_steps = 10,
    epochs=EPOCHS,
    verbose = 1
)

## save model
model.save('model/tf_name.h5')

# test = pd.read_csv('data/test.csv')
# test = test.values
# test_passenger = test[:, 0]

# test_predictions = model.predict(test_dataset).flatten()
# result = pd.DataFrame({'PassengerId': test_data['PassengerId'].values, 'Survived':test_predictions.astype(np.int32)})
# result.to_csv("data/tf_predictions.csv", index=False)
