import numpy as np


class SearchProcess(object):
    def __init__(self, D, mu):
        self.x = np.zeros((D, mu))
        self.fit = np.zeros((1, mu))
        self.mean = np.zeros((D, 1))
        self.cov = np.zeros((D, 1))

    def dis(self):
        print('x: ', self.x)
        print('fit: ', self.fit)
        print('mean: ', self.mean)
        print('cov: ', self.cov)
        print()

