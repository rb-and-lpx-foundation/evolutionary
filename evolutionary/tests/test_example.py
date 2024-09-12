import pytest
import random


@pytest.fixture
def seq():
    return list(range(10))


def test_shuffle(seq):
    # make sure the shuffled sequence does not lose any elements
    random.shuffle(seq)
    seq.sort()
    assert seq == list(range(10))

    # should raise an exception for an immutable sequence
    with pytest.raises(TypeError):
        random.shuffle((1, 2, 3))


def test_choice(seq):
    element = random.choice(seq)
    assert element in seq


def test_sample(seq):
    with pytest.raises(ValueError):
        random.sample(seq, 20)
    for element in random.sample(seq, 5):
        assert element in seq
