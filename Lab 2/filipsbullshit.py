import numpy as np  # line:1
import random  # line:2
import math  # line:3
from scipy.optimize import minimize  # line:4
import matplotlib.pyplot as plt  # line:5

"""---HYPER-PERAMETERS---"""  # line:7
C = 1  # line:9
KERNEL = "POLY"  # line:10
POWER = 5  # line:11
SIGMA = 0.5  # line:12
FILENAME = "SLACK1_POLY5_IMPOSSIBLE-DATA.pdf"  # line:14
"""----------------------"""  # line:16


def kernel(OOOOOO000OO0O000O, O0OO00OOOO0000OOO):  # line:19
    if KERNEL == "LINEAR":  # line:20
        return np.dot(OOOOOO000OO0O000O, O0OO00OOOO0000OOO)  # line:21
    if KERNEL == "RBF":  # line:22
        OO0O0OOOOO0000OO0 = np.subtract(OOOOOO000OO0O000O, O0OO00OOOO0000OOO)  # line:23
        return math.exp((-np.dot(OO0O0OOOOO0000OO0, OO0O0OOOOO0000OO0)) / (2 * SIGMA * SIGMA))  # line:24
    if KERNEL == "POLY":  # line:25
        return np.power((np.dot(OOOOOO000OO0O000O, O0OO00OOOO0000OOO) + 1), POWER)  # line:26


def calcP():  # line:29
    for OOO000O00OO000OO0 in range(N):  # line:30
        for OOOOOOO000000O00O in range(N):  # line:31
            P[OOO000O00OO000OO0][OOOOOOO000000O00O] = t[OOO000O00OO000OO0] * t[OOOOOOO000000O00O] * kernel(
                inputs[OOO000O00OO000OO0], inputs[OOOOOOO000000O00O])  # line:32


def zerofun(OO00O00OO0OO0O000):  # line:35
    return np.dot(OO00O00OO0OO0O000, t)  # line:36


def objective(OOO0OO00O0OO00OOO):  # line:39
    O0O00OO00OO00O00O = 0  # line:40
    for OOOOO00O0OOO00OO0 in range(N):  # line:41
        for OO0O00O000OO000OO in range(N):  # line:42
            O0O00OO00OO00O00O += OOO0OO00O0OO00OOO[OOOOO00O0OOO00OO0] * OOO0OO00O0OO00OOO[OO0O00O000OO000OO] * \
                                 P[OOOOO00O0OOO00OO0][OO0O00O000OO000OO]  # line:43
    O0O00OO00OO00O00O = 1 / 2 * O0O00OO00OO00O00O - np.sum(OOO0OO00O0OO00OOO)  # line:45
    return O0O00OO00OO00O00O  # line:46


def bval():  # line:49
    OO0OO0OOOO0OOOOOO = 0  # line:50
    for O0000OO0OO00OO00O in nonzero:  # line:51
        OO0OO0OOOO0OOOOOO += O0000OO0OO00OO00O[0] * O0000OO0OO00OO00O[2] * kernel(O0000OO0OO00OO00O[1],
                                                                                  nonzero[0][1])  # line:52
    return OO0OO0OOOO0OOOOOO - nonzero[0][2]  # line:53


def indicator(O0O00OOOOO00OO0OO, OOOOO00O0O0OOOOO0):  # line:56
    OOO0OOO000OOOO0O0 = 0  # line:57
    for O0O0OO0O0OOOOO0O0 in nonzero:  # line:58
        OOO0OOO000OOOO0O0 += O0O0OO0O0OOOOO0O0[0] * O0O0OO0O0OOOOO0O0[2] * kernel(
            [O0O00OOOOO00OO0OO, OOOOO00O0O0OOOOO0], O0O0OO0O0OOOOO0O0[1])  # line:59
    return OOO0OOO000OOOO0O0 - b  # line:60


if __name__ == "__main__":  # line:63
    np.random.seed(100)  # line:64
    classA = np.concatenate((np.random.randn(10, 2) * .2 + [-1.0, -1.0], np.random.randn(10, 2) * .4 + [-1, 1.0],
                             np.random.randn(10, 2) * .5 + [2, -1.0],
                             np.random.randn(10, 2) * .3 + [1.0, 1.0],))  # line:70
    classB = np.concatenate(
        (np.random.randn(25, 2) * .2 + [0.0, 0.0], np.random.randn(10, 2) * .2 + [2.3, 1.5],))  # line:80
    inputs = np.concatenate((classA, classB))  # line:82
    t = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))  # line:83
    N = inputs.shape[0]  # line:85
    permute = list(range(N))  # line:87
    random.shuffle(permute)  # line:88
    inputs = inputs[permute, :]  # line:89
    t = t[permute]  # line:90
    P = np.eye(N, N, dtype=float)  # line:92
    calcP()  # line:93
    start = np.zeros(N)  # line:95
    B = [(0, C) for _O0O000OOOO0O000O0 in range(N)]  # line:96
    XC = {'type': 'eq', 'fun': zerofun}  # line:97
    ret = minimize(objective, start, bounds=B, constraints=XC)  # line:98
    if not ret['success']:  # line:99
        print("Unavalibe to minimize the data set")  # line:100
    alpha = ret['x']  # line:102
    nonzero = [(alpha[O0OOOO0OO00O00O0O], inputs[O0OOOO0OO00O00O0O], t[O0OOOO0OO00O00O0O]) for O0OOOO0OO00O00O0O in
               range(N) if abs(alpha[O0OOOO0OO00O00O0O]) > 10e-5]  # line:104
    print(objective(alpha))  # line:106
    b = bval()  # line:108
    plt.plot([OOOOOO0O0O0OO00O0[0] for OOOOOO0O0O0OO00O0 in classA],
             [OO000OO0OO00O0O00[1] for OO000OO0OO00O0O00 in classA], "b.")  # line:112
    plt.plot([O000OO0OO0OO0OOO0[0] for O000OO0OO0OO0OOO0 in classB],
             [O0OOO0OO000OOO000[1] for O0OOO0OO000OOO000 in classB], "r.")  # line:113
    plt.axis("equal")  # line:115
    xgrid = np.linspace(-5, 5)  # line:118
    ygrid = np.linspace(-4, 4)  # line:119
    grid = np.array(
        [[indicator(O0O0OOOOOOO0O0O00, O00000O0000O0OOOO) for O0O0OOOOOOO0O0O00 in xgrid] for O00000O0000O0OOOO in
         ygrid])  # line:120
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=("red", "black", "blue"), linewidths=(1, 2, 1))  # line:122
    plt.savefig(FILENAME)  # line:124
