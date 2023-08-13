import random
import math
from threading import current_thread
import matplotlib.pyplot as plt
from numpy import cross

from cities import cities
from typing import List

MAX_X = 10000
MAX_Y = 10000
MAX_CITIES = 100
POP_SIZE = 8

def main():
    # generating random cities
    # cities = []
    # for _ in range(MAX_CITIES):
    #     cities.append((random.randint(0, MAX_X), random.randint(0, MAX_Y))) 

    print(cities)

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

    print(f'BEST RANDOM PERMUTATION TOUR: {min_fitness}({min_pos})')
    
    best_order = []
    for i in range(0, MAX_CITIES):
        best_order.append(cities[original_permutations[min_pos][i]])

    # adding final pos (back to start)
    best_order.append(cities[original_permutations[min_pos][0]])

    new_pop = tournament(original_permutations)

    # plt.scatter(*zip(*best_order))
    # plt.plot(*zip(*best_order))
    # plt.show()

def tour_fitness(tour: List[int]):
    path_length = 0
    for i in range(0, len(tour)-1):
        path_length += sld(tour[i], tour[i+1])
    
    # making the path circular
    path_length += sld(tour[-1], tour[0])
    return path_length

def sld(indx1, indx2):
    pos1 = cities[indx1]
    pos2 = cities[indx2]
    return math.sqrt(math.pow((pos1[0]-pos2[0]), 2) + math.pow((pos1[1]-pos2[1]), 2))

def evolve():
    """driver function for the evolutionary algorithm"""
    pass

def tournament(perms):
    """This will return the new population ready for crossover and mutation"""
    new_pop_size = POP_SIZE / 2
    new_pop = []
    while len(new_pop) < new_pop_size:
        # getting random players for 2 player tournament
        indx1 = random.randint(0, len(perms)-1)
        indx2 = indx1
        while indx1 == indx2:
            indx2 = random.randint(0, len(perms)-1)

        p1 = perms[indx1]
        p2 = perms[indx2]

        if tour_fitness(p1) > tour_fitness(p2):
            new_pop.append(p1)
            perms.pop(indx1)
        else:
            new_pop.append(p2)
            perms.pop(indx2)

    return new_pop

def crossover(tour1, tour2):
    """
    Takes 2 tours and performs a crossover operation on them at a random crossover point 
    returns a tuple containing the new tours
    """
    crossover_point = random.randint(0, MAX_CITIES-1)
    tour1_first, tour1_second = tour1[:crossover_point], tour1[crossover_point:]
    tour2_first, tour2_second = tour2[:crossover_point], tour2[crossover_point:]
    return(tour1_first + tour2_second, tour2_first + tour1_second)


def mutation(indx):
    pass

def generate_permutations():
    perms = [[0] for _ in range(POP_SIZE)]
    for i in range(0, POP_SIZE):
        for j in range(1, MAX_CITIES):
            perms[i].append(j)
        random.shuffle(perms[i])

    return perms    

main()
