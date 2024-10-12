import math
import numpy as np
import random
random.seed(100)
import pandas as pd


def ploting():
    """"""


def exp_func(x_: float, a_: float, b_: float) -> float:
    return a_*math.exp(-b_*x_)


if __name__ == "__main__":
    """генерация данных y1, y2, y3 на фиксированных расстояниях  друг от друга без шумов
    y1 на x = 0.2
    y2 на x = 0.5
    y3 на x = 0.8
    """
    """Вид уравнения: y = a*exp{-b*x}"""
    # Количество векторов
    N = 10000
    # Область значений для а и для b диапазон: [0; 1]
    possible_area = list([round(i, 2) for i in np.linspace(0, 1, 100)])


    # берем 3 случаных числа
    koeffs = [random.sample(possible_area, 2) for x in range(0, N)]
    print(koeffs[0])

    data = [[exp_func(x_=0.2, a_=a, b_=b),
             exp_func(x_=0.5, a_=a, b_=b),
             exp_func(x_=0.8, a_=a, b_=b)] for a, b in koeffs]

    y1, y2, y3 = zip(*data)
    a, b = zip(*koeffs)

    pd.DataFrame({"x1": 0.2,
                  "x2": 0.5,
                  "x3": 0.8,
                  "y1": y1,
                  "y2": y2,
                  "y3": y2,
                  "a": a,
                  "b": b}).to_csv(r"example3_data.txt", index=False, sep="\t")
    # print(data)
