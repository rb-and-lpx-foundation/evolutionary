import pytest
import numpy as np

from evolutionary.context import Context
from evolutionary import chromosome
from evolutionary.chromosome import Chromosome


@pytest.fixture
def context():
    return Context(42)


@pytest.fixture
def chrome(context):
    return Chromosome(context, [0.2, 0.4, 0.5, 0.5, 0.9, 0.8, 0.3])


@pytest.fixture
def zeros(context):
    return Chromosome(context, [0] * 32)


@pytest.fixture
def ones(context):
    return Chromosome(context, [1] * 32)


def test_copy(chrome):
    copy = chrome.copy()
    assert chrome.genes == copy.genes

    # copy should not share genes object with original
    copy.genes[0] = 0
    assert copy.genes[0] == 0
    assert chrome.genes[0] != 0

    # on the other hand, the original and the copy should be using the same random number generator
    r = np.random.RandomState(42)
    first5 = list(r.randint(10, size=5))
    next5 = list(r.randint(10, size=5))

    assert [6, 3, 7, 4, 6] == first5
    assert [9, 2, 6, 7, 4] == next5

    expected = first5
    actual = list(chrome.r.randint(10, size=5))
    assert expected == actual

    expected = next5
    actual = list(copy.r.randint(10, size=5))
    assert expected == actual


def test_mutation_rate(chrome, zeros, ones):
    # mutation rate should be the first element of chromosome
    assert 0.2 == chrome.mutationRate()
    assert 0 == zeros.mutationRate()
    assert 1 == ones.mutationRate()


def test_len(chrome, zeros):
    assert 7 == len(chrome)
    assert 32 == len(zeros)


def test_mutate(chrome):
    chrome.mutate()
    expected = np.array(
        [0.2, 0.4, 0.5, 0.5, 0.15599452033620265, 0.8661761457749352, 0.3]
    )
    np.testing.assert_allclose(expected, chrome.genes)


def test_crossover(context, zeros, ones):
    parent0 = Chromosome(context, [0, 1, 2, 3, 4, 5, 6])
    parent1 = Chromosome(context, [7, 8, 9, 10, 11, 12, 13])
    child0, child1 = parent0.crossover(parent1)

    # Chromosomes together should still contain the original genes.
    allgenes = child0.genes + child1.genes
    allgenes.sort()
    expected = list(range(14))
    assert expected == allgenes

    # There is only 1 in 2^32 chance that two chromosomes of length 32 will
    # remain changed after crossover. Consider that probability zero.
    child2, child3 = zeros.crossover(ones)
    assert child2.genes != [0] * 32

    expected = [
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
    ]
    assert expected == child2.genes
    expected = [
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
    ]
    assert expected == child3.genes


def test_softmax():
    logits = np.array([0.05808361, 0.86617615, 0.60111501, 0.70807258])
    actual = chromosome.softmax(logits)
    expected = np.array([0.14534122, 0.32609109, 0.25016373, 0.27840397])
    np.testing.assert_allclose(expected, actual)


def test_chromosome_select(chrome):
    objects = list("abc")
    expected = ["b", "c"]
    actual = chrome.select(objects)
    assert expected == actual

    expected = ["c", "b"]
    actual = chrome.select(objects)
    assert expected == actual

    expected = ["a", "a"]
    actual = chrome.select(objects)
    assert expected == actual

    expected = ["a", "c"]
    actual = chrome.select(objects)
    assert expected == actual
