import numpy as np
import pandas as pd
import random
import time

# Problem parameters
np.random.seed(42)  # For reproducibility

file_path='/input/'
def cal_t(i, k):
    i = int(i)
    k = int(k)
    return block[5][i] + (distance[int(block[0][i])][int(block[1][i])]) / transporter[2][k] * 3 / 2 + block[5][i]


def cal_e(i, j, k):
    i = int(i)
    j = int(j)
    k = int(k)
    return (distance[int(block[1][i])][int(block[0][j])]) / transporter[2][k]


def visibility(x, i, j, k):
    i = int(i)
    j = int(j)
    k = int(k)
    return 1 / (max(block[2][j], x + cal_e(i, j, k)) + cal_t(j, k))


def select_agent(event):
    argmin = np.where(event.min() == event)
    agent = random.choice(argmin[0])
    return agent


def select_target(unvisited, block, transporter, agent, valid, mode, pheromone):
    block_info = block.copy()
    q_m = 0.3
    q_z = 0.05
    alpha = 0.8
    beta = 0.2
    possible_selection = np.where(block_info[4] < transporter[1][agent])[0]
    possible_selection = np.intersect1d(possible_selection, unvisited)

    value = np.array([])
    if len(possible_selection) == 0:
        return 0, True
    for j in possible_selection:
        if transporter[3][agent] == -1:

            temp_value = (1 / (block[2][int(j)] + cal_t(j, agent))) ** beta + pheromone[0][int(j)] ** alpha
        else:
            temp_value = visibility(transporter[4][int(agent)], transporter[3][int(agent)], j, agent) ** beta
            +pheromone[int(transporter[3][agent])][int(j)] ** alpha
        value = np.append(value, temp_value)
    q = random.uniform(0, 1)
    if valid:
        q = 0
    if q > q_m:
        x = value
        f_x = x / np.sum(x)
        select = np.random.choice(possible_selection, p=f_x)
    elif (q > q_m - q_z) & (mode == 'ACO_RS'):
        select = np.random.choice(possible_selection)
    else:
        select = np.argmax(value)
        select = possible_selection[select]

    return select, False


def simulation(B, T, transporter, block, pheromone, valid, mode):
    unvisited_set = np.array(range(B))
    empty_time = 0
    waiting_time = 0
    tardy_time = 0
    transporter[4] = 0
    transporter[3] = -1
    update = np.zeros((B, B))

    while len(unvisited_set) > 0:
        agent = select_agent(transporter[4])

        select, done = select_target(unvisited_set, block, transporter, agent, valid, mode, pheromone)
        start_time = transporter[4][agent]

        if done:
            transporter[4][agent] = np.inf
            continue
        update[int(transporter[3][agent])][int(select)] = 1
        if transporter[3][int(agent)] == -1:
            empty_time += (distance[0][int(block[0][select])]) / transporter[2][agent]

            transporter[4][int(agent)] = block[2][int(select)] + cal_t(select, agent)

        else:
            empty_time += cal_e(transporter[3][int(agent)], select, agent)
            waiting_time += max(
                transporter[4][int(agent)] + cal_e(transporter[3][int(agent)], select, agent) - block[2][int(select)],
                0)
            transporter[4][int(agent)] = max(block[2][int(select)],
                                             transporter[4][int(agent)] + cal_e(transporter[3][int(agent)],
                                                                                select, agent)) + cal_t(select, agent)
        transporter[3][int(agent)] = select
        tardy_time += max(0, transporter[4][int(agent)] - block[3][int(select)])
        unvisited_set = np.delete(unvisited_set, np.where(unvisited_set == select)[0])

    return empty_time, waiting_time, tardy_time, update


def run(B, T, transporter, block, distance, iteration, We, Ww, Wd, mode, valid_step):
    history = []
    validation = []
    compute_time = []

    pheromone = np.ones((B, B)) / B
    next_pheromone = pheromone.copy()
    s_time = time.time()
    all_time_best = 0
    for ite in range(1, iteration + 1):
        best_z = 0
        for i in range(2 * B):

            empty_time, waiting_time, tardy_time, update = simulation(B, T, transporter, block, pheromone, False, mode)

            history.append(We * empty_time + Ww * waiting_time + Wd * tardy_time)

            z = 1 / (We * empty_time + Ww * waiting_time + Wd * tardy_time)
            if z > best_z:
                best_z = z
                best_update = update
            next_pheromone = update_pheromone(next_pheromone, update, 0.05, z)
        if best_z > all_time_best:
            all_time_best = best_z
        next_pheromone = update_pheromone(next_pheromone, best_update, 0.1, best_z)
        pheromone = next_pheromone

    return history, validation, compute_time, pheromone, 1 / all_time_best


def update_pheromone(pheromone, update, evaporate, z):
    pheromone = (1 - evaporate) * pheromone + update * z
    return pheromone

B=100
T=10

distance=pd.read_excel(file_path+'validation_big.xlsx',index_col=0,sheet_name='dis')
block_case=[]
for i in range(20):
    sname='block'+str(i)
    case_study=np.array(pd.read_excel(file_path+'validation_big.xlsx',index_col=0,sheet_name=sname)).T
    block=[]
    block.append(case_study[0])
    block.append(case_study[1])
    block.append(case_study[3]*600)
    block.append(case_study[4]*600)
    block.append(case_study[6]*50+25)
    block.append(case_study[6]*0)
    block=np.array(block)
    block_case.append(block)

total_validation = []
total_compute_time = []
for i in range(20):
    B = 100
    T = 10
    transporter = np.array([[1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
                            [50., 50., 50., 50, 50, 100, 100, 100, 100, 100],
                            [120, 120., 120., 120, 120, 120, 120, 120, 120, 120],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [0., 0., 0, 0, 0, 0, 0, 0, 0, 0]])
    block = block_case[i]
    history, validation, compute_time, pheromone, rpd_best = run(B, T, transporter, block, distance, 1000, 1, 0, 1,
                                                                 'ACO_RS', 100)
    total_validation.append(validation)
    total_compute_time.append(np.array(compute_time).mean())
    print(np.array(compute_time).mean())
    print(rpd_best)
    print('aco_rs_end')

total_validation = []
total_compute_time = []

for i in range(20):
    B = 100
    T = 10
    transporter = np.array([[1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
                            [50., 50., 50., 50, 50, 100, 100, 100, 100, 100],
                            [120, 120., 120., 120, 120, 120, 120, 120, 120, 120],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [0., 0., 0, 0, 0, 0, 0, 0, 0, 0]])
    block = block_case[i]
    history, validation, compute_time, pheromone, rpd_best = run(B, T, transporter, block, distance, 1000, 1, 0, 1,
                                                                 'ACO', 100)
    total_validation.append(validation)
    total_compute_time.append(np.array(compute_time).mean())

    print(np.array(compute_time).mean())
    print(rpd_best)
    print('aco_end')

def select_target_for_P2(unvisited, current, distance, pheromone, block, mode):
    alpha = 1
    beta = 1
    value = np.array([])
    current = int(current)
    q_m = 0.3
    q_z = 0.05

    for j in unvisited:
        j = int(j)
        temp_value = (1 / (distance[int(block[1][current])][int(block[0][j])] + 10)) ** beta + pheromone[current][
            int(j)] ** alpha
        value = np.append(value, temp_value)
    q = random.uniform(0, 1)
    if q > q_m + q_z:
        x = value

        f_x = x / np.sum(x)
        select = np.random.choice(unvisited, p=f_x)
    elif (q > q_m) & (mode == 'ACO_RS'):
        select = np.random.choice(unvisited)
    else:
        select = np.argmax(value)
        select = unvisited[select]

    return select, distance[int(block[1][current])][int(block[0][select])]


def ACO_for_P2(iteration, Q, distance, B, block):
    pheromone = np.ones((B, B)) / B
    next_pheromone = pheromone.copy()

    for ite in range(1, iteration + 1):
        best_z = 0
        for i in range(2 * B):
            current = int(i % B)
            update = np.zeros((B, B))
            sum = 0

            unvisited = np.array(range(B))
            while len(unvisited) > 1:
                unvisited = np.delete(unvisited, np.where(unvisited == current)[0])
                next, z = select_target_for_P2(unvisited, current, distance, pheromone, block, mode='ACO_RS')
                update[current][next] = 1
                sum += z
                current = next

            next_pheromone = update_pheromone(next_pheromone, update, 0.05, Q / sum)
            if Q / sum > best_z:
                best_z = Q / sum
                best_update = update
        next_pheromone = update_pheromone(next_pheromone, best_update, 0.1, best_z)
        pheromone = next_pheromone.copy()

    return pheromone


def cal_t(i, k):
    i = int(i)
    k = int(k)
    return block[5][i] + (distance[int(block[0][i])][int(block[1][i])]) / transporter[2][k] * 10 / 6 + block[5][i]


def cal_e(i, j, k):
    i = int(i)
    j = int(j)
    k = int(k)
    return (distance[int(block[1][i])][int(block[0][j])]) / transporter[2][k]


def select_first_job(distance, block, transporter_initial_position, B):
    list_of_first_job = []
    candidate = np.array(range(B))
    for tp_start in transporter_initial_position:
        d = distance[int(tp_start)][block[0][candidate].astype('int32')]
        min_d = np.where(d.min() == d)
        selected_first_job = random.choice(min_d[0])
        list_of_first_job.append(candidate[selected_first_job])
        candidate = np.delete(candidate, selected_first_job)
    return list_of_first_job


def update_pheromone(pheromone, update, evaporate, z):
    pheromone = (1 - evaporate) * pheromone + update * z
    return pheromone


def simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position, sequence, nojfet, penalty):
    empty_time = 0
    tardy_time = 0
    weight_constraint = 0
    # penalty: weight constraint, ready time
    transporter[4] = 0
    transporter[3] = -1
    sequence = sequence.astype('int32')
    step = 0
    for agent, j in enumerate(nojfet):
        current_time = 0
        tp_current_position = int(transporter_initial_position[agent])
        for i in range(int(j)):
            empty_move_time = (distance[int(tp_current_position)][int(block[0][sequence[step]])]) / transporter[2][
                int(agent)]
            empty_time += empty_move_time
            current_time += empty_move_time
            tp_current_postion = block[0][sequence[step]]
            if i == 0:
                current_time = 0
            start_time = max(block[2][sequence[step]], current_time)
            end_time = start_time + cal_t(sequence[step], agent)
            weight_constraint += max(block[4][sequence[step]] - transporter[1][agent], 0)
            tardy_time += max(0, end_time - block[3][sequence[step]])
            current_time = end_time
            tp_current_position = block[1][sequence[step]]
            step += 1
    fitness = empty_time + tardy_time + weight_constraint * penalty[0]

    return fitness, empty_time, tardy_time, weight_constraint * penalty[0]


def assign_policy(B, T, transporter, block, distance, transporter_initial_position, pheromone):
    list_of_current_job = select_first_job(distance, block, transporter_initial_position, B)

    job_list = []
    for i in list_of_current_job:
        job_list.append([i])
    unvisited = np.array(range(B))

    while len(unvisited) > 1:
        unvisited = np.setdiff1d(unvisited, list_of_current_job)

        maximum = 0
        next_action = [0, 0]
        for ind, i in enumerate(list_of_current_job):
            if maximum < np.max(pheromone[i][unvisited]):
                maximum = np.max(pheromone[i][unvisited])
                next_action[0] = ind  # index
                next_action[1] = unvisited[np.argmax(pheromone[i][unvisited])]  # next assignment
        job_list[next_action[0]].append(next_action[1])
        list_of_current_job[next_action[0]] = next_action[1]
    number_of_job_for_each_transporter = []
    initial_solution = np.array([])
    for i in job_list:
        number_of_job_for_each_transporter.append(len(i))
        initial_solution = np.append(initial_solution, np.array(i))
    return number_of_job_for_each_transporter, initial_solution


B = 100
T = 10
distance = pd.read_excel(file_path+'validation_big.xlsx', index_col=0, sheet_name='dis')
block_case = []
for i in range(20):
    sname = 'block' + str(i)
    case_study = np.array(pd.read_excel(file_path+'validation_big.xlsx', index_col=0, sheet_name=sname)).T
    block = []
    block.append(case_study[0])
    block.append(case_study[1])
    block.append(case_study[3] * 300)
    block.append(case_study[4] * 300)
    block.append(case_study[6] * 50 + 25)
    block.append(case_study[6] * 0)
    block = np.array(block)
    block_case.append(block)

for i in range(20):
    B = 100
    T = 10
    transporter = np.array([[1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
                            [50., 50., 50., 50, 50, 100, 100, 100, 100, 100],
                            [120, 120., 120., 120, 120, 120, 120, 120, 120, 120],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [0., 0., 0, 0, 0, 0, 0, 0, 0, 0]])
    block = block_case[i]

    transporter_initial_position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    Q = 5000.0
    iteration = 100
    pheromone_matric = ACO_for_P2(iteration, Q, distance, B, block)
    number_of_job_for_each_transporter, initial_solution = assign_policy(B, T, transporter, block, distance,
                                                                         transporter_initial_position, pheromone_matric)

    # 목적 함수 (최소화하려는 함수)
    # fitness = simulation_for_GA(B,T,transporter,block,distance,transporter_initial_position,sequence,nojfet,penalty=[20])

    # 초기화
    population_size = 50
    chromosome_length = B
    mutation_rate = 0.01
    generations = 10000

    # 초기 인구 생성
    population = np.zeros((population_size, chromosome_length))
    nojfet = np.zeros((population_size, T))
    for i in range(population_size):
        nojfet[i] = number_of_job_for_each_transporter
        sum = 0
        for j in range(T):
            temp = int(nojfet[i][j])

            if temp != 0:
                population[i][sum:sum + temp] = np.random.choice(initial_solution[sum:sum + temp], temp, replace=False)
            sum += temp
            # 메인 루프

    for generation in range(generations):
        # 적합도 평가
        fitness_list = np.zeros(population_size)
        for i in range(population_size):
            fitness, e, t, w = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
                                                 population[i], nojfet[i], penalty=[100])
            fitness_list[i] = 1 / fitness

        # 토너먼트 선택
        x = fitness_list

        f_x = x / np.sum(x)

        selected_indices = np.random.choice(range(population_size), 2, p=f_x)
        parent = population[selected_indices]

        # 교차 (크로스오버)
        new_nojfet = np.zeros((population_size, T))
        child = -np.ones((population_size, chromosome_length))
        for j in range(population_size):
            selected_indices = np.random.choice(range(population_size), 2, p=f_x)
            parent = population[selected_indices]

            ratio = fitness_list[selected_indices[0]] / (
                        fitness_list[selected_indices[0]] + fitness_list[selected_indices[1]])
            new_nojfet[j] = nojfet[selected_indices[0]].copy()
            if ratio < 0.5:
                parent = parent[[1, 0]]
                ratio = 1 - ratio
                new_nojfet[j] = nojfet[selected_indices[1]].copy()
            main_ind = np.random.choice(chromosome_length, int(chromosome_length * ratio) + 1)
            left_over = [x for x in parent[1] if x not in parent[0][main_ind]]
            child[j][main_ind] = parent[0][main_ind]
            step = 0
            for i in range(chromosome_length):
                if child[j][i] < 0:
                    child[j][i] = left_over[step]
                    step += 1
            mu = random.uniform(0, 1)
            if mu < mutation_rate:
                tp_c = np.random.choice(range(T), 2)
                while new_nojfet[j][tp_c[0]] == 0:
                    tp_c = np.random.choice(range(T), 2)
                sum = 0
                for num in range(tp_c[0]):
                    sum += new_nojfet[j][num]
                mutation = random.randint(sum, sum + new_nojfet[j][tp_c[0]] - 1)
                sum = 0
                for num in range(tp_c[1]):
                    sum += new_nojfet[j][num]

                insert_index = random.randint(sum, sum + new_nojfet[j][tp_c[1]]) - 1
                temp = child[j][mutation]
                new_child = child[j].copy()
                new_child = np.delete(new_child, mutation)
                new_child = np.insert(new_child, insert_index, temp)
                child[j] = new_child.copy()
                new_nojfet[j][tp_c[0]] -= 1
                new_nojfet[j][tp_c[1]] += 1

        nojfet = new_nojfet.copy()
        population = child.copy()

        # 현재 세대에서의 최적 해 출력
    for i in range(population_size):
        fitness, e, t, w = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
                                             population[i], nojfet[i], penalty=[100])
        fitness_list[i] = 1 / fitness

    best_solution = population[np.argmax(fitness_list)]
    print(1 / np.max(fitness_list))
    fitness, e, t, w = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
                                         population[np.argmax(fitness_list)], nojfet[np.argmax(fitness_list)],
                                         penalty=[100])
    print(fitness, e, t, w)
