import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class MetaModel(object):
    x = None
    y = None

    def fit(self, x, y):
        self.m = KNeighborsRegressor()
        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = np.concatenate([self.x, x])
            self.y = np.concatenate([self.y, y])
        self.m.fit(self.x, self.y)

    def predict(self, x):
        return self.m.predict(x)
