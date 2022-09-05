from collections import namedtuple
from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, Iterable, List
import numpy as np
import matplotlib.pyplot as plt


# suppress warning caused by division by inf
np.seterr(invalid="ignore", over="ignore")


def f(x):
    return 1 / (1 + np.exp(3 * (x - 3))) * 10 * x**2 + 1 / (
        1 + np.exp(-3 * (x - 3))
    ) * (0.5 * (x - 10) ** 2 + 50)


def fprime(x):
    return (
        1 / (1 + np.exp((-3) * (x - 3))) * (x - 10)
        + 1 / (1 + np.exp(3 * (x - 3))) * 20 * x
        + (3 * np.exp(9))
        / (np.exp(9 - 1.5 * x) + np.exp(1.5 * x)) ** 2
        * ((0.5 * (x - 10) ** 2 + 50) - 10 * x**2)
    )


Point = namedtuple("Point", "x y")


@dataclass
class GradientDescentCalc:
    learning_rate: float = 1
    func: Callable = f
    func_prime: Callable = fprime
    iteration: int = 10

    def gradient_descent(self, param: float) -> Point:
        next_param = param - self.learning_rate * self.func_prime(param)
        return Point(next_param, self.func(next_param))

    def get_gd_results(self, starting_point: Point):
        result: List[Point] = []
        result.append(starting_point)
        for _ in range(self.iteration - 1):
            result.append(self.gradient_descent(result[-1].x))
        return result


def plot_line_pairwise(points: List[Point], *args, **kwargs):
    point_xs = [point.x for point in points]
    point_ys = [point.y for point in points]
    plt.plot(points[0].x, points[0].y, "go", markersize=15)
    plt.plot(point_xs, point_ys, *args, **kwargs)


if __name__ == "__main__":
    x = np.linspace(-5, 20, 100)
    plt.plot(x, f(x), "k")
    plt.xlim(-5, 20)
    plt.ylim(0, 160)

    starting_points = list(map(lambda x: 25 * x - 5, np.random.rand(2)))
    starting_points = [Point(x, f(x)) for x in starting_points]
    learning_rates = [0.01, 0.3, 4]
    for learning_rate in learning_rates:
        for starting_point in starting_points:
            gdc = GradientDescentCalc(learning_rate, iteration=10)
            plot_line_pairwise(
                gdc.get_gd_results(starting_point),
                "o",
                linestyle="--",
                label=f"sp:{starting_point.x:.2f} lr:{learning_rate}",
            )
    plt.legend()
    plt.show()
