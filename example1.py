import os

import keras.models
import numpy as np
import pandas as pd
from typing import List
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf


def model_config(input_shape: tuple) -> Sequential:
    initializer = keras.initializers.RandomNormal(stddev=10, mean=1, seed=123)
    bias_init = keras.initializers.RandomNormal(stddev=10, mean=0, seed=123)
    zero_init = keras.initializers.RandomNormal(mean=0.001, stddev=0.00001, seed=123)
    model = Sequential()
    # input_shape=x_train.shape[1:],
    model.add(Dense(units=16,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    activation='softmax'))
    model.add(Dropout(0.25))
    model.add(Dense(units=8,
                    kernel_initializer=initializer,
                    kernel_regularizer=tf.keras.regularizers.L2(l2=0.001),
                    bias_initializer=zero_init,
                    activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(units=2,
                    kernel_initializer=initializer,
                    bias_initializer=bias_init,
                    activation='tanh'))
    return model


def train(model: Sequential, x_train_: List[List[float]], y_train_: List[List[float]]):
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss=['mse'],
                  metrics=['mse', 'mae'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(x_train_, y_train_, validation_split=0.2, batch_size=4, epochs=1000,
                        callbacks=[early_stop, ])
    model.save(os.path.join(os.getcwd(), "example_model_1"))
    return model


if __name__ == "__main__":
    """ """
    file_p = r'example1_data.txt'
    df = pd.read_csv(file_p, sep='\t')
    df['a'] = 0.24
    df['b'] = -0.0632
    df['x'] = df['x']/50
    df['y'] = df['y']/1.27124915
    print(df.head(10))

    # X = df[['x', 'y+δ']].copy(deep=True)
    X = df[['x', 'y']].copy(deep=True)
    Y = df[['a', 'b']].copy(deep=True)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.001, random_state=123)

    example_model_1 = model_config(X.shape)
    example_model_1 = train(model=example_model_1,
                            x_train_=x_train,
                            y_train_=y_train)

    example_model_1 = keras.models.load_model('example_model_1')

    print(list(example_model_1.predict(x_test)))


