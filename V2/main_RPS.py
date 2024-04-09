from Simulation_RPS import *
from Network_RPS import *
import torch
import vessl
np.random.seed(0)
random.seed(0)

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
    block_number=18
    location_number=8
    transporter_type=2
    transporter_number=6
    dis_high=3000
    dis_low=500
    ready_high=60 # 이거 낮추자,
    tardy_high=180
    gap=60
    K_epoch=2
    Pr_sampler=Problem_sampling(block_number,location_number,transporter_type,transporter_number,dis_high,dis_low,ready_high,tardy_high,gap)
    temp_dis=dis_low/Pr_sampler.Dis
    indices = np.diag_indices(min(temp_dis.shape))
    temp_dis[indices] = 0
    
    dis=torch.tensor(temp_dis,dtype=torch.float32).to(device)
    ppo=PPO( learning_rate=0.0005, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, epsilon=0.2, discount_factor=1,location_num=location_number,dis=dis)

    number_of_problem=10 # 한번에 몇개의 문제를
    number_of_batch=50 # 문제당 몇 episode씩 한번에 학습할껀지
    number_of_trial=2000  # #이를 몇번 반복할껀지
    number_of_iteration=10  # 전체 iteration #iteration 단위로 문제 변화
    problem = []
    Control_result=np.zeros((number_of_iteration*number_of_problem,7,6))
    history = np.zeros((number_of_iteration * number_of_trial,4))
    step = 0
    mode_list = ['Random', 'SPT', 'SET', 'SRT', 'ATC', 'EDD', 'COVERT']
    temp_step = 0
    past_time_step=0
    for i in range(number_of_iteration):

        for j in range(number_of_problem):
            B, T, b, tp, efi, nf, ef, dis, step_to_ij = Pr_sampler.sample()
            efi = efi.astype('int')
            problem.append([B, T, tp, b, efi, nf, ef, dis, step_to_ij, tardy_high])


            for nu,mod in enumerate(mode_list):
                rs=np.zeros(20)
                es=np.zeros(20)
                ts=np.zeros(20)
                for k in range(20):
                    reward_sum, tardy_sum, ett_sum, event, nf_list, ef_list, efi_list, distance_list, type_list, mask_list, actions, probs, rewards, dones,starts = simulation(
                        problem[j][0], problem[j][1], problem[j][2], problem[j][3], problem[j][4], problem[j][5],
                        problem[j][6], problem[j][7], problem[j][8], problem[j][9], mod, ppo)
                    rs[k]=reward_sum
                    es[k]=ett_sum
                    ts[k]=tardy_sum
                Control_result[temp_step,nu,0]= rs.mean()
                Control_result[temp_step,nu,1] =rs.var()
                Control_result[temp_step, nu, 2] = es.mean()
                Control_result[temp_step, nu, 3] = es.var()
                Control_result[temp_step, nu, 4] = ts.mean()
                Control_result[temp_step, nu, 5] = ts.var()
            temp_step+=1
        with pd.ExcelWriter(problem_dir+'problem_set'+str(i)+'.xlsx') as writer:
            dis = pd.DataFrame(problem[j][7])
            dis.to_excel(writer, sheet_name='Sheet_Dis', index=False)
            for j in range(number_of_problem):
                block_s=pd.DataFrame(problem[j][3])
                
                block_s.to_excel(writer, sheet_name='Sheet_block'+str(j), index=False)
                

        for nu,mod in enumerate(mode_list):
            print(mod,Control_result[past_time_step:temp_step,nu,0].mean(),Control_result[past_time_step:temp_step,nu,2].mean(),Control_result[past_time_step:temp_step,nu,4].mean())
        past_time_step = temp_step

        
        for k in range(number_of_trial):
            ave_reward = 0
            ave_tardy = 0
            ave_ett = 0
            loss_temp = 0
            data = [] #batch
            nf_lists=[]
            ef_lists=[]
            efi_lists=[]
            distance_lists=[]
            type_lists=[]
            mask_lists=[]
            action_list = np.array([])
            prob_list = np.array([])
            reward_list = np.array([])
            done_list = np.array([])
            start_list=np.array([])
            for j in range(number_of_problem):
        
                for l in range(number_of_batch):
                    reward_sum, tardy_sum, ett_sum, event, nf_list, ef_list, efi_list, distance_list, type_list, mask_list, actions, probs, rewards, dones,starts = simulation(
                        problem[j][0], problem[j][1], problem[j][2], problem[j][3], problem[j][4], problem[j][5],
                        problem[j][6], problem[j][7], problem[j][8], problem[j][9], 'RL', ppo)
                    ave_reward += reward_sum.item()
                    ave_ett += ett_sum
                    ave_tardy += tardy_sum
                    nf_lists.append(nf_list)
                    ef_lists.append(ef_list)
                    efi_lists.append(efi_list)
                    distance_lists.append(distance_list)
                    type_lists.append(type_list)
                    mask_lists.append(mask_list)
                    action_list = np.concatenate((action_list, actions))
                    prob_list = np.concatenate((prob_list, probs))
                    reward_list = np.concatenate((reward_list, rewards))
                    done_list = np.concatenate((done_list, dones))
                    start_list=np.concatenate((start_list,starts))
            nf_lists=torch.cat(nf_lists,dim=0)
            ef_lists=torch.cat(ef_lists,dim=0)
            efi_lists=torch.cat(efi_lists,dim=0)
            distance_lists=torch.cat(distance_lists,dim=0)
            type_lists=torch.cat(type_lists,dim=0)
            mask_lists=torch.cat(mask_lists,dim=0)
            for m in range(K_epoch):
                ave_loss, v_loss, p_loss = ppo.update(nf_lists, ef_lists, efi_lists, distance_lists, type_lists, mask_lists, prob_list, reward_list,done_list, start_list, action_list, number_of_problem*number_of_batch, step,model_dir)
                loss_temp += ave_loss
            ave_reward = float(ave_reward) / number_of_problem / number_of_batch
            ave_ett = float(ave_ett) / number_of_problem /number_of_batch
            ave_tardy = float(ave_tardy) / number_of_problem / number_of_batch
            
            history[step,0]=ave_reward
            vessl.log(step=step, payload={'average_reward': ave_reward})
            history[step,1]=loss_temp/K_epoch
            vessl.log(step=step, payload={'loss': loss_temp/K_epoch})
            history[step,2]=ave_ett
            vessl.log(step=step, payload={'average_ett': ave_ett})
            history[step,3]=ave_tardy
            vessl.log(step=step, payload={'average_tardy': ave_tardy})
            step += 1

    history=pd.DataFrame(history)
    history.to_excel(history_dir+'history.xlsx', sheet_name='Sheet', index=False)
