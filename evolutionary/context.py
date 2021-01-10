import numpy as np


class Context(object):
    def __init__(self, seed=1):
        self.__seed = seed
        self.__random_state = np.random.RandomState(seed)

    @property
    def r(self):
        return self.__random_state

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed
        self.reset()

    @property
    def random_state(self):
        return self.__random_state

    @random_state.setter
    def random_state(self, random_state):
        self.__random_state = random_state
        self.__seed = random_state.get_state()[1][0]

    def reset(self):
        self.__random_state.seed(self.__seed)
