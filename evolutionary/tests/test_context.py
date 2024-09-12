import unittest
import numpy as np
from evolutionary.context import Context


def test_init_default():
    ctx = Context()
    assert ctx.seed == 1
    assert isinstance(ctx.r, np.random.RandomState)
    assert isinstance(ctx.random_state, np.random.RandomState)


def test_init_with_seed():
    seed = np.random.randint(1, 10000)
    ctx = Context(seed)
    assert ctx.seed == seed
    assert isinstance(ctx.r, np.random.RandomState)
    assert isinstance(ctx.random_state, np.random.RandomState)


def test_update_random_state():
    ctx = Context()
    seed = np.random.randint(1, 10000)
    ctx.random_state = np.random.RandomState(seed)
    assert ctx.seed == seed


def test_update_seed():
    ctx = Context()
    seed = np.random.randint(1, 10000)
    ctx.seed = seed
    assert ctx.seed == seed
    assert ctx.r.get_state()[1][0] == seed
