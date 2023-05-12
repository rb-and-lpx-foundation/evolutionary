import numpy as np


def softmax(logits):
    distribution = np.exp(logits)
    distribution /= np.sum(distribution)
    return distribution


class Chromosome(object):
    def __init__(self, context, genes=None, numberOfGenes=None):
        self.context = context

        if genes is not None:
            self.genes = list(genes)
        else:
            self.genes = [self.r.rand() for i in range(numberOfGenes)]

    @property
    def r(self):
        return self.context.r

    def __len__(self):
        return len(self.genes)

    def copy(self):
        return Chromosome(self.context, list(self.genes))

    def mutationRate(self):
        return self.genes[0]

    def mutate(self):
        rate = self.mutationRate()
        for i in range(len(self.genes)):
            if self.r.rand() < rate:
                self.genes[i] = self.r.rand()

    def crossover(self, other):
        child0 = self.copy()
        child1 = other.copy()
        for i in range(len(child0.genes)):
            if self.r.rand() < 0.5:
                child0.genes[i] = other.genes[i]
                child1.genes[i] = self.genes[i]
        return child0, child1

    def select(self, objects):
        selections = []
        for i in range(1, len(self), len(objects)):
            p = softmax(self.genes[i : i + len(objects)])
            selections.append(self.r.choice(objects, p=p))
        return selections
