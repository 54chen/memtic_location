from operator import ne
from threading import local
import pandas as pd
from numpy import random
import numpy as np
import random as rd
import geemap, ee 

POP_SIZE = 100
CITY_SIZE = 73
WAREHOUSE_LEVEL = [0, 1, 2, 3, 4, 5]
WAREHOUSE_LEVEL_DISTANCE = [0,0.1,0.2,0.3,0.4,0.5] # cover distance is normalised already
WAREHOUSE_LEVEL_COST = [0,0.1,0.2,0.3,0.4,0.5] # cost of warehouse is normalised already
DISTANCE_DATA = pd.read_csv("distance_data.csv",index_col=0)
FINAL_DATA = pd.read_csv("final_data.csv",index_col=0)
CONVERGED_CNT=1

SELECT = 1
RECOMBINE = 2
MUTATE = 3
LS = 4
OP = [0,SELECT, RECOMBINE, MUTATE, LS] # Pipeline op

TOURNAMENT_SELECTION_K = 2

MUTATE_TIMES = 5

PRESERVE_RATE = 0.5
CONVERGED_RATE = 0.5

# print(FINAL_DATA.iloc[10])
# print(FINAL_DATA.loc[FINAL_DATA["name"] == 'Beijing Shi'])
# print(DISTANCE_DATA.loc[DISTANCE_DATA['city1'] == 'Beijing Shi'])
# print(DISTANCE_DATA.loc[(DISTANCE_DATA['city1'] == 'Beijing Shi') & (DISTANCE_DATA['distance'] <= 0.5)])

def fitness(solution):
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
    cover_population, cost_total = fitness(solution)
    best_cost = cost_total
    best_cover = cover_population
    same_count = 0
    for i in range(0, len(solutions)):
        p,c = fitness(solutions[i])
        # front decision
        if (c <= best_cost and p >= best_cover):
            best_cost = c
            best_cover = p
            best_solution = solutions[i]
            same_count = 0
        else:
            same_count = same_count+1
        if same_count >= CONVERGED_CNT:
            break
        print("best cover and cost for neighbour", best_cover, best_cost)

    return best_solution


def generate_random_configuration():
    return random.randint(5, size=(CITY_SIZE))


def generate_initial_population():
    pop = []
    for j in range(1, POP_SIZE):
        i = generate_random_configuration()
        i = local_search(i)
        pop.append(i)
    return pop

def is_converged(pop):
    same = 0
    total = 0
    for i in range(len(pop)):
        j = i+1
        if j >= len(pop):
            j = 0
        
        comp = np.isclose(pop[i], pop[j])
        same = same + np.count_nonzero(comp)
        total = total + len(comp)
    same_rate = same/total
    if same_rate >= CONVERGED_RATE:
        print('Converged checked!', same_rate)
        return True
    else:
        return False

def extract_from_buffer(buffer):
    return buffer

def apply_operator(op, buffer):
    if op == SELECT:
        temp = []
        while len(temp) < POP_SIZE:  
            r = rd.sample(buffer, TOURNAMENT_SELECTION_K)
            p0,c0 = fitness(r[0])
            p1,c1 = fitness(r[1])
            if c0 <= c1 and p0 >= p1:
                temp.append(r[0])
            else:
                temp.append(r[1])
        return temp

    if op == RECOMBINE: #transmission crossover
        for i in range(len(buffer)):
            point_1 = random.randint(0,CITY_SIZE-1)
            point_2 = random.randint(0,CITY_SIZE-1)
            if point_1 > point_2:
                point_1, point_2 = point_2, point_1
            r = rd.sample(buffer, 3)
            parent1 = r[0]
            parent2 = r[1]
            parent3 = r[2]
            for j in range(0, CITY_SIZE):
                buffer[i][j] = parent1[j]
                if i >= point_1:
                    buffer[i][j] = parent2[j]
                if i >= point_2:
                    buffer[i][j] = parent3[j]                        
        return buffer

    if op == MUTATE:
        for i in range(len(buffer)):
            for _ in range(MUTATE_TIMES):
                point = random.randint(0,CITY_SIZE-1)
                buffer[i][point] = random.randint(0,5) # random WAREHOUSE_LEVEL
        return buffer
    
    if op == LS:
        for i in range(len(buffer)):
            buffer[i] = local_search(buffer[i])
        return buffer


def generate_new_population(pop):
    buffer = {}
    buffer[0] = pop
    n_op = 4
    for j in range(1, n_op+1):
        buffer[j] = []

    s_par = {}
    s_desc = {}
    for j in range(1,n_op+1):
        print("OP & previous popsize:", OP[j], len(buffer[j-1]))
        s_par[j] = extract_from_buffer(buffer[j-1])
        s_desc[j] = apply_operator(OP[j], s_par[j])
        buffer[j] = s_desc[j]

    return buffer[n_op]

def update_population(pop, new_pop): # Plus strategy
    next_pop = []
    next_pop.extend(pop)
    next_pop.extend(new_pop) 
    next_pop.sort(key = lambda v: fitness(v))
    return next_pop[:POP_SIZE]

def restart_population(pop): # Random immigrant
    new_pop = []
    preserved = int(len(pop) * PRESERVE_RATE)
    for i in range(preserved):
        new_pop.append(pop[i])
    for j in range(preserved+1, len(pop)):
        i = generate_random_configuration()
        i = local_search(i)
        new_pop.append(i)
    return new_pop 


def mimetic(repetition):
    pop = generate_initial_population()
    print('>>>inital pop size:', len(pop))
    for i in range(repetition):
        print('================iteration:', i)
        new_pop = generate_new_population(pop)
        print("new_pop size:", len(new_pop))
        pop = update_population(pop, new_pop)
        if is_converged(pop):
            pop = restart_population(pop)

    return pop

def generate_visual_csv(solution):
    # latitude,longitude,radius
    visualisation = []
    for i in range(len(solution)):
        row = []
        row.append(FINAL_DATA.iloc[i]['latitude'])
        row.append(FINAL_DATA.iloc[i]['longitude'])
        row.append(solution[i])
        visualisation.append(row)
    df = pd.DataFrame(visualisation, columns=['latitude','longitude','radius'])
    df.to_csv("visualisation1.csv")


if __name__ == '__main__':
    repetition = 2
    pop = mimetic(repetition)
    best_cost = 99999999
    best_cover = 0
    best_solution = []
    for x in pop:
        p,c = fitness(x)
          # front decision
        if (c <= best_cost and p >= best_cover):
            best_cost = c
            best_cover = p
            best_solution = x
        print('fitness:', p, c)

    generate_visual_csv(best_solution)