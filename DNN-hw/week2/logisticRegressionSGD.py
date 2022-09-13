from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


class lossFunction:
    def __init__(self, func: Callable, derivative: Optional[Callable] = None) -> None:
        self.func = func
        self._derivative = derivative

    def __call__(self):
        return self.func

    @property
    def deriative(self):
        return self._derivative

    @deriative.setter
    def derivative(self, func: Callable):
        if func is None:
            raise NotImplementedError
        self._derivative = func


def logistic(X, Y, theta):
    return np.log(1 + np.exp(-Y * np.einsum("ij,j->i", X, theta))).mean()


def logistic_derivative(X, Y, theta):

    return 1 / (1 + np.exp(Y * np.dot(X, theta))) * (-Y * X)


def mean_square(X, Y, theta):
    return (
        np.clip(1 - Y * np.einsum("ij,j->i", X, theta), a_min=0, a_max=None).mean()
        + 0.1 * (np.abs(theta) ** 2).sum()
    )


def mean_square_derivative(X, Y, theta):

    value = 1 - Y * (X @ theta)
    assert value != 0, "SGD encountered an indifferentiable point"
    if value < 0:
        return 2 * 0.1 * theta
    return -Y * X + 2 * 0.1 * theta


InitialValues = namedtuple("InitialValues", "X Y theta")


@dataclass
class sgdSolver:
    initial_values: Optional[InitialValues] = None
    loss_function: lossFunction = lossFunction(lambda x: x, lambda x: 1)
    name: str = ""

    def calculate_mean_loss(self) -> float:
        result: np.ndarray = self.loss_function(
            self.initial_values.X, self.initial_values.Y, self.initial_values.theta
        )
        return result.mean()

    def train(self, epochs: int, learning_rate: float = 0.5) -> List:
        X, Y, theta = self.initial_values

        def stochastic_gradient_descent():
            idx = np.random.randint(Y.shape[0])
            return theta - learning_rate * self.loss_function.deriative(
                X[idx], Y[idx], theta
            )

        return [
            self.loss_function()(X, Y, theta := stochastic_gradient_descent())
            for _ in range(epochs)
        ]

    def __str__(self) -> str:
        return f"{self.name}"


if __name__ == "__main__":
    # Data declaration
    N, p = 30, 20
    np.random.seed(0)
    X = np.random.randn(N, p)
    Y = 2 * np.random.randint(2, size=N) - 1
    theta = np.random.normal(0.0, 1.0, p)

    # Plot option setting
    plt.xlabel("epochs")
    plt.ylabel("loss")

    # Declare sgdSolver
    logistic_regression = sgdSolver(
        loss_function=lossFunction(logistic, logistic_derivative),
        initial_values=InitialValues(X, Y, theta),
        name="logistic regression",
    )

    svm = sgdSolver(
        loss_function=lossFunction(mean_square, mean_square_derivative),
        initial_values=InitialValues(X, Y, theta),
        name="svm",
    )

    lr_result = logistic_regression.train(epochs=100_000, learning_rate=0.05)
    svm_result = svm.train(epochs=100_000, learning_rate=0.05)
    plt.plot(lr_result, label=str(logistic_regression))
    plt.plot(svm_result, label=str(svm))
    plt.legend()
    plt.show()
    # plt.savefig("plot.png")

    # Data decalration
    N = 30
    np.random.seed(0)
    X = np.random.randn(2, N)
    y = np.sign(X[0, :] ** 2 + X[1, :] ** 2 - 0.7)
    theta = 0.5
    c, s = np.cos(theta), np.sin(theta)
    X = np.array([[c, -s], [s, c]]) @ X
    X = X + np.array([[1], [1]])

    # plot raw data
    plt.subplot(2, 1, 1)
    plt.scatter(X[0][np.where(y == 1)], X[1][np.where(y == 1)])
    plt.scatter(X[0][np.where(y == -1)], X[1][np.where(y == -1)])
    plt.show()

    theta = np.random.normal(0.0, 1.0, p)
    w = theta
    xx = np.linspace(-4, 4, 1024)
    yy = np.linspace(-4, 4, 1024)
    xx, yy = np.meshgrid(xx, yy)
    Z = w[0] + (w[1] * xx + w[2] * xx**2) + (w[3] * yy + w[4] * yy**2)
    plt.contour(xx, yy, Z, 0)

    # SGD
    plt.subplot(2, 1, 2)
    lr_result = logistic_regression.train(epochs=100_000, learning_rate=0.05)
