# Modified from code by Emma Sawain
# https://github.com/newexo/cifar-ten/blob/master/code/NSGAII.py

import sys


def domination_count_and_set(population, element):
    # gives the domination count and dominating set of the element of the population

    domination_count = 0
    dominated_set = []
    for chromosome in population:
        dominated_sum = (
            0  # will equal number of objectives if chromosome is dominated by element
        )
        dominating_sum = (
            0  # will equal number of objectives if element is dominated by chromosome
        )
        for i in range(len(population[element])):
            if population[element][i] < population[chromosome][i]:
                dominated_sum += 1
            elif population[element][i] == population[chromosome][i]:
                dominated_sum += 1
                dominating_sum += 1
            else:
                dominating_sum += 1
        if (
            dominated_sum == len(population[element])
            and population[element] != population[chromosome]
        ):
            dominated_set.append(chromosome)
        if (
            dominating_sum == len(population[element])
            and population[element] != population[chromosome]
        ):
            domination_count += 1
    return [domination_count, dominated_set]


# what should we do if two chromosomes have the exact same objectives? here neither dominates the other


def fast_nondominated_sort(population):
    # gives the list of the domination fronts, in order, and a dictionary with the ranks of the chromosomes

    domination_dict = {}
    for chromosome in population:
        domination_dict[chromosome] = domination_count_and_set(population, chromosome)
    i = 0
    domination_fronts = []
    domination_fronts.append([])
    rank_list = {}
    for chromosome in domination_dict:
        if domination_dict[chromosome][0] == 0:
            domination_fronts[0].append(chromosome)
            rank_list[chromosome] = 1
    while domination_fronts[i] != []:
        domination_fronts.append([])
        for chromosome in domination_fronts[i]:
            for chrom in domination_dict[chromosome][1]:
                domination_dict[chrom][0] -= 1
                if domination_dict[chrom][0] == 0:
                    domination_fronts[i + 1].append(chrom)
                    rank_list[chrom] = i + 2
        i += 1
    return [rank_list, domination_fronts[: len(domination_fronts) - 1]]


def crowding_distance(population):
    # gives a dictionary in which the value of each chromosome is its crowding distance

    l = len(population)
    num_objectives = len(list(population.values())[0])
    crowding_distances = {}
    for chromosome in population:
        crowding_distances[chromosome] = 0
    for i in range(num_objectives):
        ordered_by_objective = sorted(population, key=lambda x: population[x][i])
        if (
            population[ordered_by_objective[0]]
            == population[ordered_by_objective[l - 1]]
        ):
            for j in range(1, l - 1):
                crowding_distances[ordered_by_objective[j]] -= 1
        else:
            crowding_distances[ordered_by_objective[0]] -= float("inf")
            crowding_distances[ordered_by_objective[l - 1]] -= float("inf")
            for j in range(1, l - 1):
                crowding_distance = (
                    float(population[ordered_by_objective[j + 1]][i])
                    - float(population[ordered_by_objective[j - 1]][i])
                ) / (
                    float(population[ordered_by_objective[l - 1]][i])
                    - float(population[ordered_by_objective[0]][i])
                )
                crowding_distances[ordered_by_objective[j]] -= crowding_distance
    return crowding_distances


def log_stdout(generation, population, objectives, fronts):
    print("Generation %d" % generation)
    for p in population:
        print(p, population[p].genes)
    print(objectives)
    print(fronts)
    sys.stdout.flush()


class log_file(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename, "w")
        f.close()

    def log(self, generation, population, objectives, fronts):
        log_stdout(generation, population, objectives, fronts)
        f = open(self.filename, "a")
        f.write("Generation %d\n" % generation)
        for p in population.values():
            f.write(p.genes.__str__())
            f.write("\n")
        f.close()


def log_discard(generation, population, objectives, fronts):
    pass


class NSGAII(object):
    def __init__(self, context, initial_population):
        self.context = context

        def poppair(i):
            key = i, 0
            return key, initial_population[i]

        self.population = dict([poppair(i) for i in range(len(initial_population))])

    def get_objectives(self, population):
        objectives = {}
        for chromosome in population:
            objectives[chromosome] = population[chromosome].getObjectives()
        return objectives

    def new_population(self, parents, parents_objectives, children):
        # combines the two new populations and then selects the best chromosome with respect to the crowded comparison order

        children_objectives = self.get_objectives(children)

        combined_population_hyperparameters = dict(
            list(parents.items()) + list(children.items())
        )
        combined_population_objectives = dict(
            list(parents_objectives.items()) + list(children_objectives.items())
        )
        fns = fast_nondominated_sort(combined_population_objectives)
        fronts = fns[1]

        new_pop_hyperparameters = {}
        new_pop_objectives = {}
        spaces_remaining = len(parents)
        front = 0
        while spaces_remaining > 0:
            if spaces_remaining >= len(fronts[front]):
                for chromosome in fronts[front]:
                    new_pop_objectives[chromosome] = combined_population_objectives[
                        chromosome
                    ]
                    new_pop_hyperparameters[chromosome] = (
                        combined_population_hyperparameters[chromosome]
                    )
                    spaces_remaining -= 1
                front += 1
            else:
                crowding_distances = crowding_distance(combined_population_objectives)
                front_with_crowding = dict(
                    (key, crowding_distances[key]) for key in fronts[front]
                )
                sorted_by_crowding = sorted(
                    front_with_crowding, key=lambda x: front_with_crowding[x]
                )
                for i in range(spaces_remaining):
                    new_pop_objectives[sorted_by_crowding[i]] = (
                        combined_population_objectives[sorted_by_crowding[i]]
                    )
                    new_pop_hyperparameters[sorted_by_crowding[i]] = (
                        combined_population_hyperparameters[sorted_by_crowding[i]]
                    )
                spaces_remaining = 0
        return new_pop_hyperparameters, new_pop_objectives, fronts

    def make_children(self, parents, generation):
        children = {}
        keys = list(parents.keys())
        while len(keys) > 1:
            choice1, choice2 = self.context.r.choice(len(keys), size=2, replace=False)
            key1 = keys[choice1]
            key2 = keys[choice2]
            keys.remove(key1)
            keys.remove(key2)
            child0, child1 = parents[key1].crossover(parents[key2])
            child0.mutate()
            child1.mutate()
            new_key1 = key1[0], generation + 1
            new_key2 = key2[0], generation + 1
            children[new_key1] = child0
            children[new_key2] = child1
        return children

    def evolve(self, num_generations=100, log_out=log_discard):
        objectives = self.get_objectives(self.population)
        for generation in range(num_generations):
            self.population, objectives, self.fronts = self.new_population(
                self.population,
                objectives,
                self.make_children(self.population, generation),
            )
            log_out(generation, self.population, objectives, self.fronts)

        return self.population.values()
