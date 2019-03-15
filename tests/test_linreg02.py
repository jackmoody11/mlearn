from mlearn.sentdex import linreg02 as lr2
import numpy as np
import random
import pytest


def create_dataset(hm, var, step=2, corr=False):
    val = 1
    ys = list()
    for i in range(hm):
        y = random.randrange(-var, var)
        ys.append(y)
        if corr == 'pos':
            val += step
        elif corr == 'neg':
            val -= step
    xs = [i for i in range(hm)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def test_corr():
    # If correlation is set to false, then we should not see a correlation
    xs, ys = create_dataset(100, 40, 2, corr=False)
    m1, b1 = lr2.best_fit_slope_and_intercept(xs, ys)
    regression_line = [m1 * x + b1 for x in xs]
    r_squared = lr2.coef_of_determination(ys, regression_line)
    assert abs(r_squared) < 0.1

