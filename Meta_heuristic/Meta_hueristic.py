import numpy as np
import pandas as pd
import random
import time

# Problem parameters
np.random.seed(42)  # For reproducibility

problem_name='B1.xlsx'


info=pd.read_excel('/input/'+problem_name,sheet_name='info')
B=info.at[0,'B']
T=info.at[0,'T']
B_T=info.at[0,'B_T']
Ms=info.at[0,'Ms'].split(',')


print(B)
print(T)
print(B_T)

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
    elif (q > q_m - q_z) & (mode == 'ACO'):
        select = np.random.choice(possible_selection)
    else:
        select = np.argmax(value)
        select = possible_selection[select]

    return select, False


def simulation(B, T, transporter, block, pheromone, valid, mode):
    TP_load=np.zeros(T)
    TP_end_time=np.zeros(T)
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
            TP_end_time[agent]=start_time
            transporter[4][agent] = np.inf
            continue
        else:

            TP_load[agent]+=1

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
    return empty_time, waiting_time, tardy_time, update,TP_load,TP_end_time


def run(B, T, transporter, block, distance, iteration, We, Ww, Wd, mode, valid_step):
    time_limit=1800

    pheromone = np.ones((B, B)) / B
    next_pheromone = pheromone.copy()
    s_time = time.time()
    all_time_best = 0
    all_best_We=0
    all_best_Wd=0
    for ite in range(1, iteration + 1):
        best_z = 0
        best_We=0
        best_Wd=0
        step=0

        for i in range(2*B):

            empty_time, waiting_time, tardy_time, update,TP_load,TP_end_time = simulation(B, T, transporter, block, pheromone, False, mode)

            step+=1

            z = 1 / (We * empty_time + Ww * waiting_time + Wd * tardy_time)
            if z > best_z:
                best_z = z
                best_update = update
                best_We=empty_time
                best_Wd=tardy_time
                best_TP_load=TP_load
                best_TP_end_time=TP_end_time
            next_pheromone = update_pheromone(next_pheromone, update, 0.05, z)
        if best_z > all_time_best:
            all_time_best = best_z
            all_best_We=best_We
            all_best_Wd=best_Wd
            all_best_TP_load=best_TP_load
            all_best_TP_end_time=best_TP_end_time
        next_pheromone = update_pheromone(next_pheromone, best_update, 0.1, best_z)
        pheromone = next_pheromone
        f_time=time.time()
        if f_time-s_time>time_limit:
            break
    compute_time=f_time-s_time
    print(compute_time)
    return compute_time, pheromone, 1 / all_time_best,all_best_We,all_best_Wd,all_best_TP_load,all_best_TP_end_time

def update_pheromone(pheromone, update, evaporate, z):
    pheromone = (1 - evaporate) * pheromone + update * z
    return pheromone

distance=pd.read_excel(problem_name,index_col=0,sheet_name='dis')
block_case=[]
for i in range(20):
    sname='block'+str(i)
    case_study=np.array(pd.read_excel(problem_name,index_col=0,sheet_name=sname)).T
    block=[]
    block.append(case_study[0])
    block.append(case_study[1])
    block.append(case_study[3]*int(B_T*60))
    block.append(case_study[4]*int(B_T*60))
    block.append(case_study[6]*50+25)
    block.append(case_study[6]*0)
    block=np.array(block)
    block_case.append(block)

result_ACO=np.zeros((20,4))
time_balance_ACO=np.zeros((20,T))
load_balance_ACO=np.zeros((20,T))

result_ACO_RS=np.zeros((20,4))
time_balance_ACO_RS=np.zeros((20,T))
load_balance_ACO_RS=np.zeros((20,T))

result_GA=np.zeros((20,4))
time_balance_GA=np.zeros((20,T))
load_balance_GA=np.zeros((20,T))
'''
for i in range(20):
    transporter = np.array([[1+2*int(x/T*2) for x in range(T)],
                        [0 for x in range(T)],
                        [120 for x in range(T)],
                        [-1 for x in range(T)],
                        [0 for x in range(T)]])
    current = 0
    for e, j in enumerate(Ms):

        transporter[1,current:current + int(j)] = 50 + e * 50
        current += int(j)
    transporter=transporter.astype(np.float32)
    block = block_case[i]
    compute_time, pheromone, rpd_best,all_best_We,all_best_Wd,all_best_TP_load,all_best_TP_end_time = run(B, T, transporter, block, distance, 1000, 1, 0, 1,
                                                                 'ACO', 100)
    result_ACO[i,0]=round(rpd_best,3)
    result_ACO[i,1]=round(all_best_We,3)
    result_ACO[i,2]=round(all_best_Wd,3)
    result_ACO[i,3]=round(compute_time,3)
    time_balance_ACO[i]=all_best_TP_end_time
    load_balance_ACO[i]=all_best_TP_load
    print(i)
df=pd.DataFrame(result_ACO)
time_balances=pd.DataFrame(time_balance_ACO)
load_balances=pd.DataFrame(load_balance_ACO)

with pd.ExcelWriter( problem_name+ '_ACO.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Validation', index=False)
    time_balances.to_excel(writer, sheet_name='Time_balance', index=False)
    load_balances.to_excel(writer, sheet_name='Load_balance', index=False)
print('ACO end')

for i in range(20):
    transporter = np.array([[1 + 2 * int(x / T * 2) for x in range(T)],
                            [0 for x in range(T)],
                            [120 for x in range(T)],
                            [-1 for x in range(T)],
                            [0 for x in range(T)]])
    current = 0
    for e, j in enumerate(Ms):
        transporter[1, current:current + int(j)] = 50 + e * 50
        current += int(j)
    transporter=transporter.astype(np.float32)
    block = block_case[i]
    compute_time, pheromone,rpd_best,all_best_We,all_best_Wd,all_best_TP_load,all_best_TP_end_time = run(B, T, transporter, block, distance, 1000, 1, 0, 1,
                                                                 'ACO_RS', 100)
    result_ACO_RS[i,0]=round(rpd_best,3)
    result_ACO_RS[i,1]=round(all_best_We,3)
    result_ACO_RS[i,2]=round(all_best_Wd,3)
    result_ACO_RS[i,3]=round(compute_time,3)
    time_balance_ACO_RS[i]=all_best_TP_end_time
    load_balance_ACO_RS[i]=all_best_TP_load
    print(i)
df=pd.DataFrame(result_ACO_RS)
time_balances=pd.DataFrame(time_balance_ACO_RS)
load_balances=pd.DataFrame(load_balance_ACO_RS)

with pd.ExcelWriter( problem_name+ '_ACO_RS.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Validation', index=False)
    time_balances.to_excel(writer, sheet_name='Time_balance', index=False)
    load_balances.to_excel(writer, sheet_name='Load_balance', index=False)
print('ACO_RS end')
'''
result_GA=np.zeros((20,4))
time_balance_GA=np.zeros((20,T))
load_balance_GA=np.zeros((20,T))

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
        for i in range(2*B):
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
    TP_load=np.zeros(T)
    TP_end_time=np.zeros(T)
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
            TP_end_time[agent]=end_time
            TP_load[agent]+=1
            tp_current_position = block[1][sequence[step]]
            step += 1

    fitness = empty_time + tardy_time + weight_constraint * penalty[0]

    return fitness, empty_time, tardy_time, weight_constraint * penalty[0],TP_end_time,TP_load


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


distance = pd.read_excel(problem_name, index_col=0, sheet_name='dis')
block_case = []
for i in range(20):
    sname = 'block' + str(i)
    case_study = np.array(pd.read_excel(problem_name, index_col=0, sheet_name=sname)).T
    block = []
    block.append(case_study[0])
    block.append(case_study[1])
    block.append(case_study[3] *int(B_T*60))
    block.append(case_study[4] *int(B_T*60))
    block.append(case_study[6] * 50 + 25)
    block.append(case_study[6] * 0)
    block = np.array(block)
    block_case.append(block)

for b_case in range(20):

    transporter = np.array([[1 + 2 * int(x / T * 2) for x in range(T)],
                            [0 for x in range(T)],
                            [120 for x in range(T)],
                            [-1 for x in range(T)],
                            [0 for x in range(T)]])
    current = 0
    for e, j in enumerate(Ms):
        transporter[1, current:current + int(j)] = 50 + e * 50
        current += int(j)
    transporter = transporter.astype(np.float32)
    block = block_case[b_case]

    transporter_initial_position = [0 for x in range(T)]
    s_t = time.time()

    Q = 5000.0
    iteration = 100
    pheromone_matric = ACO_for_P2(iteration, Q, distance, B, block)
    number_of_job_for_each_transporter, initial_solution = assign_policy(B, T, transporter, block, distance,
                                                                         transporter_initial_position, pheromone_matric)

    # 목적 함수 (최소화하려는 함수)
    # fitness = simulation_for_GA(B,T,transporter,block,distance,transporter_initial_position,sequence,nojfet,penalty=[20])

    # 초기화
    population_size = 4*B
    chromosome_length = B
    mutation_rate = 0.01
    generations = 8000
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
    time_limit=1800
    for generation in range(generations):
        # 적합도 평가
        fitness_list = np.zeros(population_size)
        for i in range(population_size):
            fitness, e, t, w,TP_end_time,TP_load = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
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
        f_t = time.time()
        if f_t-s_t>time_limit:
            break
        # 현재 세대에서의 최적 해 출력
    for i in range(population_size):
        fitness, e, t, w,TP_end_time,TP_load  = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
                                             population[i], nojfet[i], penalty=[100])
        fitness_list[i] = 1 / fitness

    best_solution = population[np.argmax(fitness_list)]
    
    fitness, e, t, w,TP_end_time,TP_load  = simulation_for_GA(B, T, transporter, block, distance, transporter_initial_position,
                                         population[np.argmax(fitness_list)], nojfet[np.argmax(fitness_list)],
                                         penalty=[100])

    print(f_t-s_t)
    print(fitness)
    result_GA[b_case, 0] = round(fitness,3)
    result_GA[b_case, 1] = round(e,3)
    result_GA[b_case, 2] = round(t,3)
    result_GA[b_case, 3] = round(f_t-s_t,3)
    time_balance_GA=TP_end_time
    load_balance_GA=TP_load
    print(b_case)
df=pd.DataFrame(result_GA)
time_balances=pd.DataFrame(time_balance_GA)
load_balances=pd.DataFrame(load_balance_GA)
with pd.ExcelWriter( problem_name+ '_ACO_RS.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Validation', index=False)
    time_balances.to_excel(writer, sheet_name='Time_balance', index=False)
    load_balances.to_excel(writer, sheet_name='Load_balance', index=False)
