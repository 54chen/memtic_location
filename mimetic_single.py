import csv
import pandas as pd
from numpy import random
import numpy as np

POP_SIZE = 20
CITY_SIZE = 73
WAREHOUSE_LEVEL = [0, 1, 2, 3, 4, 5]
WAREHOUSE_LEVEL_DISTANCE = [0,0.1,0.2,0.3,0.4,0.5] # cover distance is normalised already
WAREHOUSE_LEVEL_COST = [0,0.1,0.2,0.3,0.4,0.5] # cost of warehouse is normalised already
DISTANCE_DATA = pd.read_csv("distance_data.csv",index_col=0)
FINAL_DATA = pd.read_csv("final_data.csv",index_col=0)
CONVERGED_CNT=20

# print(FINAL_DATA.iloc[10])
# print(FINAL_DATA.loc[FINAL_DATA["name"] == 'Beijing Shi'])
# print(DISTANCE_DATA.loc[DISTANCE_DATA['city1'] == 'Beijing Shi'])
# print(DISTANCE_DATA.loc[(DISTANCE_DATA['city1'] == 'Beijing Shi') & (DISTANCE_DATA['distance'] <= 0.5)])

def cost(solution):
    cover_population = 0
    cost_total = 0
    i = 0
    to_cities = []
    for vector in solution:
        if vector!=0:
            cost_total = cost_total + WAREHOUSE_LEVEL_COST[vector]
            city_name = FINAL_DATA.iloc[i]['name']
            city2_list = DISTANCE_DATA.loc[(DISTANCE_DATA['city1'] == city_name) & (DISTANCE_DATA['distance'] <= WAREHOUSE_LEVEL_DISTANCE[vector])]
            temp = city2_list.merge(FINAL_DATA[['name','increase','population']], left_on='city2', right_on='name')
            temp['cost'] = (1-temp.increase) * temp.distance * temp.population
            cost_total = cost_total + temp['cost'].sum()
            to_cities.extend(city2_list['city2'].values.tolist())    
        i = i + 1

    to_cities = list(set(to_cities))
    cover_population = FINAL_DATA.loc[FINAL_DATA['name'].isin(to_cities)]['population'].sum()
    # cost_total = warehouse cost + (1-increase(traget)) * distance*population(target). 
    return cover_population, cost_total


def generate_neighbourhoods(solution):
    neighbors = []
    length = len(solution)
    for i in range(int(length*0.5)):
        for j in range(int(length*0.5), length):
            temp = np.copy(solution)
            swap = temp[i]
            temp[i] = temp[j]
            temp[j] = swap

            neighbors.append(temp)
    print("generate neighbours", len(neighbors))
    return neighbors


def local_search(solution):
    solutions = generate_neighbourhoods(solution)
    best_solution = solution
    cover_population, cost_total = cost(solution)
    print("cost:",cover_population, cost_total)
    best_cost = cost_total
    best_cover = cover_population
    same_count = 0
    for i in range(0, len(solutions)):
        p,c = cost(solutions[i])
        print("cost for neighbour", p, c)
        if (c <= best_cost and p >= best_cover):
            best_cost = c
            best_cover = p
            best_solution = solutions[i]
            same_count = 0
        else:
            same_count = same_count+1
        if same_count >= CONVERGED_CNT:
            break
    return best_solution


def generate_random_configuration():
    return random.randint(5, size=(CITY_SIZE))


def generate_initial_population():
    pop = []
    for j in range(1, POP_SIZE):
        print("pop count:",j)
        i = generate_random_configuration()
        i = local_search(i)
        pop.append(i)
    return pop

def is_converged(pop):
    print()


def generate_new_population(pop):
    print(pop)


def update_population(pop, new_pop):
    print()


def restart_population(pop):
    print()


def mimetic(repetition):
    pop = generate_initial_population()
    print('inital pop', pop)
    for i in range(repetition):
        new_pop = generate_new_population(pop)
        pop = update_population(pop, new_pop)
        if is_converged(pop):
            pop = restart_population(pop)

    return pop


if __name__ == '__main__':
    repetition = 1
    mimetic(repetition)
