import math

import numpy as np
import random

import pandas as pd


def ploting():
    """"""


if __name__ == "__main__":
    """генерация данных x1, x2, y1, y2"""
    """Вид уравнения: y = a*exp{-b*x}"""
    # Количество векторов
    N = 10000
    # по X для а и для B диапазон: [0; 1]
    possible_area = list([round(i, 2) for i in np.linspace(0, 1, 100)])

    # берем 3 случаных числа
    data = [random.sample(possible_area, 4) for x in range(0, N)]
    print(data)
    # сортируем первые два элемента по возрастанию
    data = [[*sorted(i[:3]), *i[3:]] for i in data]
    x1, x2, a, b = zip(*data)
    y1, y2 = zip(*[[round(i[3]*math.exp(-i[0]*i[3]), 2), round(i[3]*math.exp(-i[1]*i[3]), 2)] for i in data])

    pd.DataFrame({"x1": x1,
                  "x2": x2,
                  "y1": y1,
                  "y2": y2,
                  "a": a,
                  "b": b}).to_csv(r"example2_data.txt", index=False, sep="\t")
    print(data)
