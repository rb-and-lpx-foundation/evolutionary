import unittest
import numpy as np
from numpy.linalg import norm

from evolutionary.context import Context
from evolutionary import chromosome
from evolutionary.chromosome import Chromosome


class TestChromosome(unittest.TestCase):

    def setUp(self):
        self.context = Context(42)
        self.chrome = Chromosome(self.context, [0.2, 0.4, 0.5, 0.5, 0.9, 0.8, 0.3])
        self.zeros = Chromosome(self.context, [0] * 32)
        self.ones = Chromosome(self.context, [1] * 32)

    def testCopy(self):
        copy = self.chrome.copy()
        self.assertEqual(self.chrome.genes, copy.genes)

        # copy should not share genes object with original
        copy.genes[0] = 0
        self.assertEqual(0, copy.genes[0])
        self.assertNotEqual(0, self.chrome.genes[0])

        # on the other hand, the original and the copy should be using the same random number generator
        r = np.random.RandomState(42)
        first5 = list(r.randint(10, size=5))
        next5 = list(r.randint(10, size=5))

        self.assertEqual([6, 3, 7, 4, 6], first5)
        self.assertEqual([9, 2, 6, 7, 4], next5)

        expected = first5
        actual = list(self.chrome.r.randint(10, size=5))
        self.assertEqual(expected, actual)

        expected = next5
        actual = list(copy.r.randint(10, size=5))
        self.assertEqual(expected, actual)

    def testMutationRate(self):
        # mutation rate should be the first element of chromosome
        self.assertEqual(0.2, self.chrome.mutationRate())
        self.assertEqual(0, self.zeros.mutationRate())
        self.assertEqual(1, self.ones.mutationRate())

    def testLen(self):
        self.assertEqual(7, len(self.chrome))
        self.assertEqual(32, len(self.zeros))

    def testMutate(self):
        self.chrome.mutate()
        expected = np.array([0.2, 0.4, 0.5, 0.5, 0.15599452033620265, 0.8661761457749352, 0.3])
        self.assertAlmostEqual(0, norm(expected - self.chrome.genes))
        
    def testCrossover(self):
        parent0 = Chromosome(self.context, [0, 1, 2, 3, 4, 5, 6])
        parent1 = Chromosome(self.context, [7, 8, 9, 10, 11, 12, 13])
        child0, child1 = parent0.crossover(parent1)
        
        # Chromosomes together should still contain all of the original genes.
        allgenes = child0.genes + child1.genes
        allgenes.sort()
        expected = list(range(14))
        self.assertEqual(expected, allgenes)

        # There is only 1 in 2^32 chance that two chromosomes of length 32 will 
        # remain changed after crossover. Consider that probability zero.
        child2, child3 = self.zeros.crossover(self.ones)
        self.assertNotEqual(child2.genes, [0] * 32)

        expected = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]
        self.assertEqual(expected, child2.genes)
        expected = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        self.assertEqual(expected, child3.genes)

    def testSoftmax(self):
        logits = np.array([0.05808361, 0.86617615, 0.60111501, 0.70807258])
        actual = chromosome.softmax(logits)
        expected = np.array([0.14534122, 0.32609109, 0.25016373, 0.27840397])
        self.assertAlmostEqual(0, norm(expected - actual))

    def testChromosomeSelect(self):
        objects = list('abc')
        expected = ['b', 'c']
        actual = self.chrome.select(objects)
        self.assertEqual(expected, actual)
        
        expected = ['c', 'b']
        actual = self.chrome.select(objects)
        self.assertEqual(expected, actual)
        
        expected = ['a', 'a']
        actual = self.chrome.select(objects)
        self.assertEqual(expected, actual)
        
        expected = ['a', 'c']
        actual = self.chrome.select(objects)
        self.assertEqual(expected, actual)
