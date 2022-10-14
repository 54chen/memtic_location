import multiprocessing
import random as rd
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random

# config
ITERATION = 3
POP_SIZE = 100
MUTATE_TIMES = 5
PRESERVE_RATE = 0.8
CONVERGED_RATE = 0.6
TOURNAMENT_SIZE = 2
CONVERGED_CNT = 10

# constant
CITY_SIZE = 73
WAREHOUSE_LEVEL = [0, 1, 2, 3, 4, 5]
# cover distance is normalised already
WAREHOUSE_LEVEL_DISTANCE = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# cost of warehouse is normalised already
WAREHOUSE_LEVEL_COST = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
DISTANCE_DATA = pd.read_csv("distance_data.csv", index_col=0)
FINAL_DATA = pd.read_csv("final_data.csv", index_col=0)

SELECT = 1
RECOMBINE = 2
MUTATE = 3
LS = 4
OP = [0, SELECT, RECOMBINE, MUTATE, LS]  # Pipeline op

# pareto front [population] = cost


def add_pareto_front(pareto_front, pareto_front_solution, population, cost, solution):
    if population not in pareto_front:
        pareto_front[population] = cost
        pareto_front_solution[population] = solution
        return True
    else:
        if pareto_front[population] > cost:
            pareto_front[population] = cost
            pareto_front_solution[population] = solution
            return True
    return False


def fitness(solution):
    cover_population = 0
    cost_total = 0
    i = 0
    to_cities = []
    for vector in solution:
        if vector != 0:
            cost_total = cost_total + WAREHOUSE_LEVEL_COST[vector]
            city_name = FINAL_DATA.iloc[i]['name']
            city2_list = DISTANCE_DATA.loc[(DISTANCE_DATA['city1'] == city_name) & (
                DISTANCE_DATA['distance'] <= WAREHOUSE_LEVEL_DISTANCE[vector])]
            temp = city2_list.merge(
                FINAL_DATA[['name', 'increase', 'population']], left_on='city2', right_on='name')
            temp['cost'] = (1-temp.increase) * temp.distance * temp.population
            cost_total = cost_total + temp['cost'].sum()
            to_cities.extend(city2_list['city2'].values.tolist())
        i = i + 1

    to_cities = list(set(to_cities))
    cover_population = FINAL_DATA.loc[FINAL_DATA['name'].isin(
        to_cities)]['population'].sum()
    # cost_total = warehouse cost + (1-increase(traget)) * distance*population(target).
    return cover_population, cost_total, solution


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
    return neighbors


def local_search(solution):
    solutions = generate_neighbourhoods(solution)
    cover_population, cost_total, _ = fitness(solution)
    same_count = 0
    best_pareto_front = {}
    best_pareto_front_solution = {}
    add_pareto_front(best_pareto_front, best_pareto_front_solution,
                     cover_population, cost_total, solution)

    for i in range(0, len(solutions)):
        cover_population, cost_total, _ = fitness(solutions[i])
        # compare in front map
        if (add_pareto_front(best_pareto_front, best_pareto_front_solution, cover_population, cost_total, solution)):
            same_count = 0
        else:
            same_count = same_count+1
        if same_count >= CONVERGED_CNT:
            break
    print("pareto front for local_search, x->key:", cover_population)
    s = rd.sample(list(best_pareto_front_solution.values()), 1)[
        0]  # random choose from the pareto front line
    return s


def generate_random_configuration(max):
    return random.randint(max+1, size=(CITY_SIZE))


def generate_initial_population():
    pop = []
    item = []
    for _ in range(0, int(POP_SIZE/2)):
        i = generate_random_configuration(5)
        item.append(i)
    with multiprocessing.get_context('spawn').Pool(processes=12) as pool:
        for result in pool.map(local_search, item):
            pop.append(result)

    # add diversity by random in half of solutions
    for j in range(int(POP_SIZE/2), POP_SIZE):
        i = generate_random_configuration(2)
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
        print('Converged checked! rate:%f , pop:%s' % (same_rate, pop))
        return True
    else:
        return False


def extract_from_buffer(buffer):
    return buffer


def multi_objective_tournament_select(pop, tournament_size=TOURNAMENT_SIZE):
    best = pop[random.randint(0, len(pop)-1)]
    b_p, b_c, _ = fitness(best)
    for i in range(tournament_size):
        next = pop[random.randint(0, len(pop)-1)]
        p1, c1, _ = fitness(next)
        if p1 > b_p:
            best = next
            b_p = p1
            print('population:%s, cost:%s' % (p1, c1))
        if p1 == b_p and c1 <= b_c:
            best = next
            b_c = c1
            print('population:%s, cost:%s' % (p1, c1))
    return best


def apply_operator(op, buffer):
    if op == SELECT:
        temp = {}
        for i in range(0, len(buffer)):
            temp[i] = multi_objective_tournament_select(buffer)
        return list(temp.values())

    if op == RECOMBINE:  # transmission crossover
        for i in range(len(buffer)):
            point_1 = random.randint(0, CITY_SIZE-1)
            point_2 = random.randint(0, CITY_SIZE-1)
            if point_1 > point_2:
                point_1, point_2 = point_2, point_1
            r = rd.sample(list(buffer), 3)
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
                point = random.randint(0, CITY_SIZE-1)
                buffer[i][point] = random.randint(
                    0, 5)  # random WAREHOUSE_LEVEL
        return buffer

    if op == LS:
        temp = []
        with multiprocessing.get_context('spawn').Pool(processes=12) as pool:
            for result in pool.map(local_search, buffer):
                temp.append(result)
        return temp


def generate_new_population(p):
    pop = np.copy(p)
    buffer = {}
    buffer[0] = pop
    n_op = 4
    for j in range(1, n_op+1):
        buffer[j] = []

    s_par = {}
    s_desc = {}
    for j in range(1, n_op+1):
        print("OP & previous popsize:", OP[j], len(buffer[j-1]))
        s_par[j] = extract_from_buffer(buffer[j-1])
        s_desc[j] = apply_operator(OP[j], s_par[j])
        buffer[j] = s_desc[j]
    return buffer[n_op]


def update_population(pop, new_pop):  # Plus strategy
    next_pop = []
    next_pop.extend(pop)
    next_pop.extend(new_pop)
    temp = []

    best_pareto_front = {}
    best_pareto_front_solution = {}

    for i in range(0, len(next_pop)):
        solution = next_pop[i]
        cover_population, cost_total, _ = fitness(solution)
        add_pareto_front(best_pareto_front, best_pareto_front_solution,
                         cover_population, cost_total, solution)

    return list(best_pareto_front_solution.values())


def restart_population(pop):  # Random immigrant
    new_pop = []
    preserved = int(len(pop) * PRESERVE_RATE)
    for i in range(preserved):
        new_pop.append(pop[i])
    for j in range(preserved, len(pop)):
        i = generate_random_configuration(2)
        i = local_search(i)
        new_pop.append(i)
    return new_pop


def mimetic(repetition):
    pop = generate_initial_population()
    print('>>>inital pop size:', len(pop))
    print('>>>inital pop:', pop)

    for i in range(repetition):
        print('================iteration:', i)
        new_pop = generate_new_population(pop)
        print("new_pop has generated new_pop:", np.array(new_pop).shape)
        pop = update_population(pop, new_pop)
        print("update pop X new_pop:", np.array(pop).shape)

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
    df = pd.DataFrame(visualisation, columns=[
                      'latitude', 'longitude', 'radius'])
    df.to_csv("visualisation1.csv")


if __name__ == '__main__':
    start_time = time.time()

    same_count = 0
    best_pareto_front = {}
    best_pareto_front_solution = {}
    pop = mimetic(ITERATION)
    print('mimetic run successfully, time cost:%f' %
          (time.time() - start_time))

    # store result by csv and picture of pareto front
    with multiprocessing.get_context('spawn').Pool(processes=12) as pool:
        for result in pool.map(fitness, pop):
            cover_population, cost_total, solution = result
            add_pareto_front(best_pareto_front, best_pareto_front_solution,
                             cover_population, cost_total, solution)

    best_solution = rd.sample(list(best_pareto_front_solution.values()), 1)[
        0]  # random choose one from pareto front
    print("iteration:%d, initial popsize:%d, pareto front size:%d " %
          (ITERATION, POP_SIZE, len(pop)))
    p, c, _ = fitness(best_solution)
    print("csv verion evaluation:p->%f,c->%f" % (p, c))
    generate_visual_csv(best_solution)
    plt.scatter(list(best_pareto_front), best_pareto_front.values())
    plt.show()
