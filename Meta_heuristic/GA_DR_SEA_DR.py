import numpy as np
import pandas as pd
import vessl

file_path='/input/'

B = 60
T = 8

problem_name='validation_busy.xlsx'

transporter = np.array([[1 + 2 * int(x / B * 2) for x in range(B)],
                        [50 + 50 * int(x / B * 2) for x in range(B)],
                        [120 for x in range(B)],
                        [-1 for x in range(B)],
                        [0 for x in range(B)]])
transporter=transporter.astype(np.float32)

distance = pd.read_excel(file_path+problem_name, index_col=0, sheet_name='dis')

block_case = []
for i in range(20):
    sname = 'block' + str(i)
    case_study = np.array(pd.read_excel(file_path+problem_name, index_col=0, sheet_name=sname)).T
    block = []
    block.append(case_study[0])
    block.append(case_study[1])
    block.append(case_study[3] * 450)
    block.append(case_study[4] * 450)
    block.append(case_study[6] * 50 + 25)
    block.append(case_study[6] * 0)
    block = np.array(block)
    block_case.append(block)
# 초기화
population_size = B * 3
chromosome_length = B
P_R = 0.5
P_C = 0.8
P_M = 0.2
generations = 1000


def cal_t(i, k):
    i = int(i)
    k = int(k)
    return block[5][i] + (distance[int(block[0][i])][int(block[1][i])]) / transporter[2][k] * 3 / 2 + block[5][i]


# 블록 i를 transporeter k에 배정했을때

def cal_e(i, j, k):
    i = int(i)
    if i == -1:
        distance[0][int(block[0][j])] / transporter[2][k]
    j = int(j)
    k = int(k)
    return (distance[int(block[1][i])][int(block[0][j])]) / transporter[2][k]


# 블록 i를 운반했던 TP k가 블록 j를 다음 블록으로 운송할 때
def select_agent_GA(block, block_num, transporter):
    T = transporter.shape[1]
    event = np.zeros(T)
    for i in range(T):
        if transporter[1][i] < block[4][block_num]:
            event[i] = finish_time(transporter[4][i], transporter[3][i], block_num, i)
        else:
            event[i] = np.inf

    argmin = np.where(event.min() == event)
    agent = np.random.choice(argmin[0])
    return agent


def finish_time(x, i, j, k):
    i = int(i)
    j = int(j)
    k = int(k)
    return (max(block[2][j], x + cal_e(i, j, k)) + cal_t(j, k))


def simulation_for_GA(B, T, transporter, block, distance, sequence):
    empty_time = 0
    tardy_time = 0
    weight_constraint = 0
    # penalty: weight constraint, ready time
    transporter[4] = 0
    transporter[3] = -1
    sequence = sequence.astype('int32')
    step = 0
    for block_num in sequence:
        agent = select_agent_GA(block, block_num, transporter)
        empty_time += cal_e(transporter[3][agent], block_num, agent)
        transporter[4][agent] = finish_time(transporter[4][agent], transporter[3][agent], block_num, agent)

        transporter[3][agent] = block_num
        tardy = max(0, transporter[4][agent] - block[3][block_num])

        tardy_time += tardy

    fitness = empty_time + tardy_time
    return fitness, empty_time, tardy_time


# 초기 인구 생성
def generate_random_sequence():
    sequence = np.arange(B)
    np.random.shuffle(sequence)
    return sequence

    # 50개의 수열을 생성
population = np.array([generate_random_sequence() for _ in range(population_size)])


for b_case in range(20):
    for generation in range(generations):
        # 적합도 평가
        fitness_list = np.zeros(population_size)
        for i in range(population_size):
            fitness, e, t = simulation_for_GA(B, T, transporter, block_case[b_case], distance, population[i])
            fitness_list[i] = 1 / fitness

        sorted_indices = np.argsort(fitness_list)[::-1]

        # 상위 60개의 인덱스 추출
        top_indices = sorted_indices[:2 * B]

        # 상위 60개의 인덱스에서 30개를 복원 추출로 선택
        selected_indices = np.random.choice(top_indices, int(2 * B * P_R), replace=True)

        new_population = np.zeros((population_size, B))

        # 새로운 population 생성
        new_population[:int(2 * B * P_R)] = population[selected_indices]

        for i in range(int(B * P_C)):  # 24번의 교차로 48개의 자녀 생성
            # 두 부모 무작위 선택
            parents_indices = np.random.choice(top_indices, 2, replace=False)
            parent1 = population[parents_indices[0]]
            parent2 = population[parents_indices[1]]

            # 교차 지점 선택
            crossover_point = np.random.randint(1, B - 1)

            # 자녀 생성
            child1 = np.concatenate((parent1[:crossover_point],
                                     [gene for gene in parent2 if gene not in parent1[:crossover_point]]))
            child2 = np.concatenate((parent2[:crossover_point],
                                     [gene for gene in parent1 if gene not in parent2[:crossover_point]]))

            # 새로운 population에 자녀 추가
            new_population[int(2 * B * P_R) + 2 * i] = child1
            new_population[int(2 * B * P_R) + 2 * i + 1] = child2

        selected_indices = np.random.choice(top_indices, int(B * P_M * 2), replace=False)

        for num, idx in enumerate(selected_indices):
            # 변경할 위치 선택

            new_population[-num - 1] = population[idx]
            i, j = np.random.choice(B, 2, replace=False)

            new_population[-num - 1][i], new_population[-num - 1][j] = new_population[-num - 1][j], \
            new_population[-num - 1][i]

        population = new_population.copy()

        # 현재 세대에서의 최적 해 출력
    for i in range(population_size):
        fitness, e, t = simulation_for_GA(B, T, transporter, block_case[b_case], distance, population[i])
        fitness_list[i] = 1 / fitness

    best_solution = population[np.argmax(fitness_list)]
    fitness, e, t =simulation_for_GA(B, T, transporter, block_case[b_case], distance, best_solution)
    print(round(fitness, 3),round(e, 3),round(t, 3))




population_size = B
chromosome_length = B

generations = B * 1000


def pull_operator(chromosome):
    n = len(chromosome)
    point1, point2 = np.sort(np.random.choice(n, 2, replace=False))
    pulled_genes = chromosome[point2:].copy()
    remaining_genes = chromosome[point1:point2].copy()
    chromosome[point1:point1 + len(pulled_genes)] = pulled_genes
    chromosome[point1 + len(pulled_genes):] = remaining_genes
    return chromosome


def insert_operator(chromosome):
    n = len(chromosome)
    point1, point2 = np.random.choice(n - 1, 2, replace=False)
    gene_to_insert = chromosome[point2].copy()
    chromosome = np.delete(chromosome, point2)
    chromosome = np.insert(chromosome, point1 + 1, gene_to_insert)
    return chromosome


def swap_operator(chromosome):
    n = len(chromosome)
    point1, point2 = np.random.choice(n, 2, replace=False)
    chromosome[point1], chromosome[point2] = chromosome[point2], chromosome[point1]
    return chromosome


def inner_random_operator(chromosome):
    n = len(chromosome)
    point1, point2 = np.sort(np.random.choice(n, 2, replace=False))
    inner_genes = chromosome[point1:point2 + 1].copy()
    np.random.shuffle(inner_genes)
    chromosome[point1:point2 + 1] = inner_genes
    return chromosome


def outer_random_operator(chromosome):
    n = len(chromosome)
    point1, point2 = np.sort(np.random.choice(n, 2, replace=False))
    outer_genes = np.concatenate((chromosome[:point1], chromosome[point2 + 1:]))
    np.random.shuffle(outer_genes)
    chromosome[:point1] = outer_genes[:point1].copy()
    chromosome[point2 + 1:] = outer_genes[point1:].copy()
    return chromosome


# 연산자를 랜덤하게 선택하고 실행하는 함수
def apply_random_operator(chromosome):
    operators = [pull_operator, insert_operator, swap_operator, inner_random_operator, outer_random_operator]
    selected_operator = np.random.choice(operators)
    return selected_operator(chromosome)


# 초기 인구 생성
def generate_random_sequence():
    sequence = np.arange(B)
    np.random.shuffle(sequence)
    return sequence


# 50개의 수열을 생성
population = np.array([generate_random_sequence() for _ in range(population_size)])

for b_case in range(20):
    for it in range(generations):
        # 적합도 평가
        i = np.random.choice(population_size, 1, replace=False)[0]

        fitness, e, t = simulation_for_GA(B, T, transporter, block_case[b_case], distance, population[i])
        new_chromosome = apply_random_operator(population[i].copy())
        fitness_new, e, t = simulation_for_GA(B, T, transporter, block_case[b_case], distance, new_chromosome)
        if fitness_new < fitness:
            population[i] = new_chromosome.copy()

    fitness_list = np.zeros(population_size)
    # 현재 세대에서의 최적 해 출력
    for i in range(population_size):
        fitness, e, t = simulation_for_GA(B, T, transporter, block_case[b_case], distance, population[i])
        fitness_list[i] = 1 / fitness

    best_solution = population[np.argmax(fitness_list)]
    fitness, e, t =simulation_for_GA(B, T, transporter, block_case[b_case], distance, best_solution)
    print(round(fitness, 3),round(e, 3),round(t, 3))

