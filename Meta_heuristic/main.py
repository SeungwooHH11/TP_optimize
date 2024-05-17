import numpy as np

from Simulation_DAN import *
from Network_DAN import *
import torch

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

if __name__=="__main__":
    problem_dir='/output/problem_set/'
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    model_dir='/output/model/ppo/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    history_dir='/output/history/'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    device='cuda'
    block_number=100
    location_number=45
    transporter_type=2
    transporter_number=10
    dis_high=3000
    dis_low=500
    ready_high=200 # 이거 낮추자,
    tardy_high=600
    gap=200
    K_epoch=2
    Pr_sampler=Problem_sampling(block_number,location_number,transporter_type,transporter_number,dis_high,dis_low,ready_high,tardy_high,gap)
    temp_dis=dis_low/Pr_sampler.Dis
    indices = np.diag_indices(min(temp_dis.shape))
    temp_dis[indices] = 0
    
    dis=torch.tensor(temp_dis,dtype=torch.float32).to(device)
    ppo=PPO( learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, epsilon=0.2, discount_factor=1,location_num=location_number,dis=dis)
    number_of_validation=20
    number_of_validation_batch=50
    number_of_problem=10 # 한번에 몇개의 문제를
    number_of_batch=80 # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial=1  #1, 10, 100, 1000 #이를 몇번 반복할껀지
    number_of_iteration=int(1000/number_of_trial)  # 전체 iteration #iteration 단위로 문제 변화
    validation=[]
    validation_step = 10
    Control_result=np.zeros((20,7,6))
    history = np.zeros((number_of_iteration * number_of_trial,2))
    validation_history=np.zeros((int(5000/validation_step)+10,6))
    step = 0
    mode_list = ['Random', 'SPT', 'SET', 'SRT', 'ATC', 'EDD', 'COVERT']
    temp_step = 0
    past_time_step=0

    for j in range(number_of_validation):
        B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
        efi = efi.astype('int')
        validation.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])

        for nu, mod in enumerate(mode_list):
            rs = np.zeros(20)
            es = np.zeros(20)
            ts = np.zeros(20)
            for k in range(20):
                reward_sum, tardy_sum, ett_sum, event, episode, actions, probs, rewards, dones = simulation(
                    validation[j][0], validation[j][1], validation[j][2], validation[j][3], validation[j][4], validation[j][5],
                    validation[j][6], validation[j][7], validation[j][8], validation[j][9], mod, ppo)
                rs[k] = reward_sum
                es[k] = ett_sum
                ts[k] = tardy_sum
            Control_result[temp_step, nu, 0] = rs.mean()
            Control_result[temp_step, nu, 1] = rs.var()
            Control_result[temp_step, nu, 2] = es.mean()
            Control_result[temp_step, nu, 3] = es.var()
            Control_result[temp_step, nu, 4] = ts.mean()
            Control_result[temp_step, nu, 5] = ts.var()
        temp_step += 1
    for nu, mod in enumerate(mode_list):
        print(mod, Control_result[past_time_step:temp_step, nu, 0].mean(),
              Control_result[past_time_step:temp_step, nu, 2].mean(),
              Control_result[past_time_step:temp_step, nu, 4].mean())

    writer = pd.ExcelWriter('validation.xlsx', engine='openpyxl')
    dis=pd.DataFrame(validation[0][7])
    dis.to_excel(writer, sheet_name='dis')

    for j in range(number_of_validation):
        dis = pd.DataFrame(validation[j][3])
        dis.to_excel(writer, sheet_name='block'+str(j))

    writer.close()
