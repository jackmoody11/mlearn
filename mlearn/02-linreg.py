from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(x, y):
    """
    Great proof of these equations: http://seismo.berkeley.edu/~kirchner/eps_120/Toolkits/Toolkit_10.pdf
    :param x: x-values for linear regression
    :type x: np.array
    :param y: y-values for linear regression
    :type y: np.array
    :return: slope and y-intercept for line of best fit
    :rtype: tuple
    """
    m = (mean(x) * mean(y) - mean(x * y)) / (mean(x) ** 2 - mean(x**2))
    # m = Cov(X,Y)/Var(X)
    b = mean(y) - m * mean(x)
    # b = EY - m * EX
    return m, b


def squared_error(y_orig, y_line):
    """
    :param y_orig: Array with y values
    :type y_orig: np.array
    :param y_line: Array with expected values for y based on line of best fit
    :type y_line: np.array
    :return: Squared error
    :rtype: float
    """
    return sum((y_line - y_orig)**2)


def coef_of_determination(y_orig, y_line):
    """
    :param y_orig: Original list of y values
    :type y_orig: np.array
    :param y_line: Flat y-hat line (list of constant value)
    :type y_line: np.array
    :return: R-squared
    :rtype: float
    """
    y_hat_line = [mean(y_orig), ] * len(y_orig)
    squared_error_regression = squared_error(y_orig, y_line)
    squared_error_y_hat = squared_error(y_orig, y_hat_line)
    return 1 - (squared_error_regression/squared_error_y_hat)


# Find line of best fit and regression values
m1, b1 = best_fit_slope_and_intercept(xs, ys)
regression_line = [m1 * x + b1 for x in xs]

# Find r-squared value
r_squared = coef_of_determination(ys, regression_line)
print(r_squared)
# Plot regression line
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
