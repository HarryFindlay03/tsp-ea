import random
import math
import matplotlib.pyplot as plt

from cities import cities
from typing import List

"""
TODO
- add in convergence checks
- not really working too well, I think it is my tournament selection so I want to change this or maybe add in some form of elitism
- I need to add in some adjustable rates like crossover rates and mutation rates as this may be where some of my problems lie because everything seems to be working okay
- also cleaning up the code
"""

MAX_CITIES = 30
POP_SIZE = 8
LIMIT = 5

def main():

    # generating initial permutations of cities
    original_permutations = generate_permutations()

    # finding the best permutation from original population for test
    min_fitness = tour_fitness(original_permutations[0])
    min_pos = 0
    for i in range(1, POP_SIZE):
        curr_fitness = tour_fitness(original_permutations[i])
        if curr_fitness < min_fitness:
            min_fitness = curr_fitness
            min_pos = i

    print(f"BEST STARTING TOUR LENGTH: {min_fitness}({min_pos})")

    # running ea
    ea_best_order = evolve(original_permutations)
    print(f"BEST EA TOUR LENGTH: {tour_fitness(ea_best_order)}")

    show = str(input("SHOW PLOT (Y/N): "))
    if show.lower() == "y":
        # generating points to show
        order = []
        for i in range(0, len(ea_best_order)):
            order.append(cities[ea_best_order[i]])

        plt.scatter(*zip(*order))
        plt.plot(*zip(*order))
        plt.show()
    else:
        print("GOODBYE! 🕺")


def evolve(original_permutations):
    """
    driver function for the evolutionary algorithm
    returns the best tour found until max limit or convergence
    """
    pop = original_permutations
    gen = 0
    while (gen < LIMIT):
        # tournament selection
        pop = tournament(pop)

        # crossover
        next_gen = []
        for j in range(0, len(pop) - 1):
            res1, res2 = ordered_crossover(pop[j], pop[j + 1])
            next_gen.append(res1)
            next_gen.append(res2)

        # mutation
        for j in range(0, len(next_gen)):
            next_gen[j] = mutation(next_gen[j])

        # resetting population to next generation
        pop = next_gen

        gen += 1


    # returning min fitness from best population
    min_fitness = tour_fitness(pop[0])
    best_pos = 0
    for i in range(1, len(pop)):
        curr_fitness = tour_fitness(pop[i])
        if curr_fitness < min_fitness:
            min_fitness = curr_fitness
            best_pos = i

    return pop[best_pos]


def tournament(perms):
    """This will return the new population ready for crossover and mutation"""
    new_pop_size = POP_SIZE / 2  # this has changed until crossover is fixed
    new_pop = []
    
    # elitism - promoting the best performing
    min_pos = 0
    min_fitness = tour_fitness(perms[0])
    for i in range(1, len(perms)):
        curr_fitness = tour_fitness(perms[i])
        if curr_fitness < min_fitness:
            min_fitness = curr_fitness
            min_pos = i

    new_pop.append(perms[min_pos])
    perms.pop(min_pos)

    while len(new_pop) < new_pop_size:
        # getting random players for 2 player tournament
        indx1 = random.randint(0, len(perms) - 1)
        indx2 = indx1
        while indx1 == indx2:
            indx2 = random.randint(0, len(perms) - 1)

        p1 = perms[indx1]
        p2 = perms[indx2]

        if tour_fitness(p1) < tour_fitness(p2):
            new_pop.append(p1)
            perms.pop(indx1)
        else:
            new_pop.append(p2)
            perms.pop(indx2)

    return new_pop


def tour_fitness(tour: List[int]):
    path_length = 0
    for i in range(0, len(tour) - 1):
        path_length += sld(tour[i], tour[i + 1])

    # making the path circular
    path_length += sld(tour[-1], tour[0])
    return path_length


def sld(indx1, indx2):
    pos1 = cities[indx1]
    pos2 = cities[indx2]
    return math.sqrt(
        math.pow((pos1[0] - pos2[0]), 2) + math.pow((pos1[1] - pos2[1]), 2)
    )


def crossover(tour1, tour2):
    """
    Takes 2 tours and performs a crossover operation on them at a random crossover point
    returns a tuple containing the new tours
    """
    crossover_point = random.randint(0, MAX_CITIES - 1)
    tour1_first, tour1_second = tour1[:crossover_point], tour1[crossover_point:]
    tour2_first, tour2_second = tour2[:crossover_point], tour2[crossover_point:]
    return (tour1_first + tour2_second, tour2_first + tour1_second)


def ordered_crossover(tour1, tour2):
    child1 = [-1] * len(tour1)
    child2 = [-1] * len(tour2)

    # creating child1
    # generate random swath
    indx1 = random.randint(0, len(tour1) - 2)
    indx2 = random.randint(indx1, len(tour1) - 2)

    for i in range(indx1, indx2 + 1):
        child1[i] = tour1[i]

    child_pos = 0

    for i in range(0, len(tour2)):
        if tour2[i] not in child1:
            if child_pos == indx1:
                child_pos = indx2 + 1
            child1[child_pos] = tour2[i]
            child_pos += 1

    # creating child 2
    indx1 = random.randint(0, len(tour2) - 2)
    indx2 = random.randint(indx1, len(tour2) - 2)

    for i in range(indx1, indx2 + 1):
        child2[i] = tour2[i]

    child_pos = 0

    for i in range(0, len(tour1)):
        if tour1[i] not in child2:
            if child_pos == indx1:
                child_pos = indx2 + 1
            child2[child_pos] = tour1[i]
            child_pos += 1
    
    return (child1, child2)


def mutation(tour):
    # generate two random positions for a swap mutation
    indx1 = random.randint(0, len(tour) - 1)
    indx2 = indx1
    while indx1 == indx2:
        indx2 = random.randint(0, len(tour) - 1)

    tour[indx1], tour[indx2] = tour[indx2], tour[indx1]
    return tour


def generate_permutations():
    perms = [[0] for _ in range(POP_SIZE)]
    for i in range(0, POP_SIZE):
        for j in range(1, MAX_CITIES):
            perms[i].append(j)
        random.shuffle(perms[i])

    return perms


main()
