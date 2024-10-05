import math
import random
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
    initializer = 'he_normal'

    bias_init = keras.initializers.RandomNormal(stddev=10, mean=0, seed=123)
    zero_init = keras.initializers.RandomNormal(mean=0.001, stddev=0.00001, seed=123)
    model = Sequential()
    # input_shape=x_train.shape[1:],
    model.add(Dense(units=512,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.001),
                    activation='sigmoid'))
    model.add(Dense(units=256,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    # kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.001),
                    activation='sigmoid'))
    model.add(Dense(units=128,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    activation='sigmoid'))
    model.add(Dense(units=64,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    # kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.1),
                    activation='sigmoid'))
    model.add(Dense(units=2,
                    kernel_initializer=initializer,
                    bias_initializer=zero_init,
                    # activation='relu',
                    activation='sigmoid'))
    return model


def train_model(model: Sequential, x_train_: List[List[float]], y_train_: List[List[float]]):
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss=['mse'],
                  metrics=['mse', 'mae'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(x_train_, y_train_, validation_split=0.2, batch_size=4, epochs=200,
                        # callbacks=[early_stop, ])
                        )

    model.save(os.path.join(os.getcwd(), "models/example_2_model_1"))
    return model


def df_splitter(wb: pd.DataFrame, test_size: float, random_state: int = 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(random_state)
    if test_size > 1:
        return ValueError("test_size must be less than 1")
    test_n = int(round(wb.shape[0] * test_size))
    indexes = sorted(random.sample(range(0, wb.shape[0]), test_n))
    test_wb = wb.loc[indexes]
    train_wb = wb.loc[[x for x in range(0, wb.shape[0]) if x not in indexes]]
    return train_wb.copy(deep=True), test_wb.copy(deep=True)


def mean_squared_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Списки должны быть одной длины")

    mse = sum((a - b) ** 2 for a, b in zip(list1, list2)) / len(list1)
    return mse

if __name__ == "__main__":
    """4 признака:x1, x2, y1, y2 -> a, b; без шумов; модель: только Dense;"""
    file_p = r'example2_data.txt'
    df = pd.read_csv(file_p, sep='\t')

    train, test = df_splitter(df, 0.1)

    x_train = train[['x1', 'x2', "y1", "y2"]].copy(deep=True)
    x_test = test[['x1', 'x2', "y1", "y2"]].copy(deep=True)
    y_train = train[["a", "b"]].copy(deep=True)
    y_test = test[["a", "b"]].copy(deep=True)

    example_model_2 = model_config(x_train.shape)
    example_model_2 = train_model(model=example_model_2,
                                  x_train_=x_train,
                                  y_train_=y_train)

    # example_model_2 = keras.models.load_model('models/example_2_model_1')

    # test_data =
    print(df[["a", "b"]].loc[test.index.tolist()].values.tolist())
    y_pred = [list(x) for x in example_model_2.predict(x_test)]
    print(y_pred)
    y_test_list = [x[0]*math.exp(-0.5*x[-1]) for x in df[["a", "b"]].loc[test.index.tolist()].values.tolist()]
    y_pred_list = [x[0]*math.exp(-0.5*x[-1]) for x in y_pred]
    print(mean_squared_error(y_test_list, y_pred_list))
