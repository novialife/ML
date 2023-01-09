import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"""---HYPER-PERAMETERS---"""

C = 0.001  # C --> 0 ==> Under fitting, C --> inf ==> Over fitting
KERNEL = "POLY"  # Change kernel
POWER = 5  # Used for Poly
SIGMA = 0.5  # Used for RBF

FILENAME = "SLACK0.001_POLY5_IMPOSSIBLE-DATAs.pdf"

"""----------------------"""


def kernel(x, y):
    match KERNEL:
        case "LINEAR":
            return np.dot(x, y)
        case "RBF":
            diff = np.subtract(x, y)
            return math.exp((-np.dot(diff, diff)) / (2 * SIGMA * SIGMA))
        case "POLY":
            return np.power((np.dot(x, y) + 1), POWER)


def calcP():
    for i in range(N):
        for j in range(N):
            P[i][j] = t[i] * t[j] * kernel(inputs[i], inputs[j])


def zerofun(alpha):
    return np.dot(alpha, t)


def objective(alpha):
    tot_sum = 0
    for i in range(N):
        for j in range(N):
            tot_sum += alpha[i] * alpha[j] * P[i][j]

    tot_sum = 1 / 2 * tot_sum - np.sum(alpha)
    return tot_sum


def bval():
    bsum = 0
    for value in nonzero:
        bsum += value[0] * value[2] * kernel(value[1], nonzero[0][1])
    return bsum - nonzero[0][2]


def indicator(x, y):
    totsum = 0
    for value in nonzero:
        totsum += value[0] * value[2] * kernel([x, y], value[1])
    return totsum - b


if __name__ == "__main__":
    np.random.seed(100)

    classA = np.concatenate((np.random.randn(10, 2) * .2 + [-1.0, -1.0],
                             np.random.randn(10, 2) * .4 + [-1, 1.0],
                             np.random.randn(10, 2) * .5 + [2, -1.0],
                             np.random.randn(10, 2) * .3 + [1.0, 1.0],
                             ))

    # classA = np.concatenate((
    #     np.random.randn(5, 2) * 0.3 + [1.5, 0.5],
    #     np.random.randn(5, 2) * 0.3 + [-1.5, 0.5],
    #     np.random.randn(5, 2) * 0.7 + [0.0, -1.0],
    # ))

    classB = np.concatenate((np.random.randn(25, 2) * .2 + [0.0, 0.0],
                             np.random.randn(10, 2) * .2 + [2.3, 1.5],
                             ))

    inputs = np.concatenate((classA, classB))
    t = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    t = t[permute]

    P = np.eye(N, N, dtype=float)  # <-- init 0 matrix N*N
    calcP()

    start = np.zeros(N)
    B = [(0, C) for _ in range(N)]
    XC = {'type': 'eq', 'fun': zerofun}
    ret = minimize(objective, start, bounds=B, constraints=XC)
    if not ret['success']:
        print("Unavalibe to minimize the data set")

    alpha = ret['x']
    nonzero = [(alpha[i], inputs[i], t[i])
               for i in range(N) if abs(alpha[i]) > 10e-5]

    print(objective(alpha))

    b = bval()

    # print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in nonzero]))

    plt.plot([p[0] for p in classA], [p[1] for p in classA], "b.")
    plt.plot([p[0] for p in classB], [p[1] for p in classB], "r.")

    plt.axis("equal")  # Force same scale on both axes
    # plt.show() #Show the plot on the screen

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=("red", "black", "blue"), linewidths=(1, 2, 1))

    plt.savefig(FILENAME)  # Save a copy in a file
    # plt.show()