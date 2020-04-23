from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns


# load train and test data
train_data = pd.read_csv('data/pre_processing/processed_train4.csv')
test_data = pd.read_csv('data/pre_processing/processed_test4.csv')

train_df = train_data
train_np = train_df.values
# sns.pairplot(train_df[['Age_scaled', 'Fare_scaled', 'Pclass_1', 'Pclass_2', 'Pclass_3']])

# y equal to survival
y = train_np[:, 0]
# x equal to featrue
x = train_np[:, 1:]

# split train and validation dataset
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.3)


train_x_dataset = tf.data.Dataset.from_tensor_slices((x))
train_y_dataset = tf.data.Dataset.from_tensor_slices((y))



# build a model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=1, activation='relu', input_shape=(,14)),
        layers.MaxPooling2D((1,2)),
        layers.Conv2D(64, kernel_size=1, activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.Conv2D(64, kernel_size=1, activation='relu'),
        layers.MaxPooling2D((1,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy']
                 )
    return model

model = build_model()
model.summary()

EPOCHS = 500
history = model.fit(
    train_x, train_y,
    validation_data = (valid_x, valid_y),
    epochs=EPOCHS
)

## save model
model.save('model/tf_cnn.h5')

