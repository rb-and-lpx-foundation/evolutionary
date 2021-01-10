import unittest
import numpy as np
from evolutionary.context import Context


class TestContext(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init_default(self):
        ctx = Context()
        self.assertEqual(ctx.seed, 1)
        self.assertIsInstance(ctx.r, np.random.RandomState)
        self.assertIsInstance(ctx.random_state, np.random.RandomState)

    def test_init_with_seed(self):
        seed = np.random.randint(1, 10000)
        ctx = Context(seed)
        self.assertEqual(ctx.seed, seed)
        self.assertIsInstance(ctx.r, np.random.RandomState)
        self.assertIsInstance(ctx.random_state, np.random.RandomState)

    def test_update_random_state(self):
        ctx = Context()
        seed = np.random.randint(1, 10000)
        ctx.random_state = np.random.RandomState(seed)
        self.assertEqual(ctx.seed, seed)

    def test_update_seed(self):
        ctx = Context()
        seed = np.random.randint(1, 10000)
        ctx.seed = seed
        self.assertEqual(ctx.seed, seed)
        self.assertEqual(ctx.r.get_state()[1][0], seed)
