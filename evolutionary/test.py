import unittest

from evolutionary.tests.test_example import test_example
from evolutionary.tests.test_chromosome import TestChromosome
from evolutionary.tests.test_nsga2 import TestNSGA2
from evolutionary.tests.test_context import TestContext

class countsuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d, %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))

def suite():
    s = countsuite()

    s.add(test_example)
    s.add(TestChromosome)
    s.add(TestNSGA2)
    s.add(TestContext)
    
    return s.s

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
