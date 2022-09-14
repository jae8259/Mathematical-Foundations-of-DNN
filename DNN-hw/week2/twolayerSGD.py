import numpy as np
import matplotlib.pyplot as plt


def f_true(x):
    return (x - 2) * np.cos(x * 4)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def f_th(theta, x):
    return np.sum(
        theta[2 * p : 3 * p]
        * sigmoid(theta[0:p] * np.reshape(x, (-1, 1)) + theta[p : 2 * p]),
        axis=1,
    )


def diff_f_th(theta: np.ndarray, x: float):
    a, b, u = np.split(theta, 3)
    grad_u = sigmoid(a * x + b)
    grad_b = sigmoid_prime(a * x + b) * u
    grad_a = grad_b * x
    return np.concatenate([grad_a, grad_b, grad_u])


if __name__ == "__main__":
    # Data declaration
    K = 10000
    alpha = 0.007
    N, p = 30, 50
    np.random.seed(0)
    a0 = np.random.normal(loc=0.0, scale=4.0, size=p)
    b0 = np.random.normal(loc=0.0, scale=4.0, size=p)
    u0 = np.random.normal(loc=0, scale=0.05, size=p)
    theta = np.concatenate((a0, b0, u0))

    X = np.random.normal(loc=0.0, scale=1.0, size=N)
    Y = f_true(X)

    xx = np.linspace(-2, 2, 1024)
    plt.plot(X, f_true(X), "rx", label="Data points")
    plt.plot(xx, f_true(xx), "r", label="True Fn")

    for k in range(K):
        idx = np.random.randint(N)
        grad = (f_th(theta, X[idx]) - Y[idx]) * diff_f_th(theta, X[idx])
        theta = theta - alpha * grad
        if (k + 1) % 2000 == 0:
            plt.plot(xx, f_th(theta, xx), label=f"Learned Fn after {k+1} iterations")

    plt.legend()
    plt.show()
    # plt.savefig("plot.png")
