import numpy as np


def f1(x, o, bias):
    return np.sum((x - o) * (x - o)) + bias


def f2(x, o, bias):
    x_o = np.abs(x - o)
    return np.sum(x_o) + bias


def f3(x, o, bias):
    z = x - o + 1
    tmp = z[:-1] * z[:-1] - z[1:]
    part1 = 100 * np.sum(tmp * tmp)
    part2 = np.sum(((z - 1) * (z - 1))[:-1])
    return part1 + part2 + bias


def f4(x, o, bias):
    z = x - o
    return np.sum(z * z - 10 * np.cos(2 * np.pi * z) + 10) + bias


def f5(x, o, bias):
    z = x - o
    part1 = np.sum(z * z / 4000)
    n = x.shape[0]
    I = np.arange(1, n + 1, 1)
    tmp = np.cos(z / np.sqrt(I))
    part2 = -tmp.prod()
    return part1 + part2 + bias + 1


def f6(x, o, bias):
    z = x - o
    n = x.shape[0]
    return -20 * np.exp(-0.2 * np.sqrt(1 / n * np.sum(z * z))) - np.exp(
        1 / n * np.sum(np.cos(2 * np.pi * z))) + 20 + np.e + bias


def f7(x, o, bias):
    pass


# def f1(x, o, bias):
#     return np.sum(x * x)


test_func_dict = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6,
    7: f7
}

test_func_bound = {
    1: [-100, 100],
    2: [-100, 100],
    3: [-100, 100],
    4: [-5, 5],
    5: [-600, 600],
    6: [-32, 32],
    7: [0, 0],
}

test_func_bias = [0, -450.0, -450.0, 390.0, -330.0, -180.0, -140.0]


class TestEnv(object):

    def __init__(self, d, problem_t):
        self.d = d
        if 0 < problem_t <= 7:
            self.o = np.load("data/f%d.npy" % problem_t)
            self.o = self.o[:d]
            self.bias = 0
        else:
            raise ValueError("wrong problem type")
        self.fuc = test_func_dict[problem_t]

    def evaluate(self, samples):
        """

        :param samples: An matrix of D * mu
        :return:
        """
        d = samples.shape[1]
        res = np.zeros(d, dtype=float)
        for i in range(d):
            res[i] += self.fuc(samples[:, i], self.o, self.bias)
        return res
