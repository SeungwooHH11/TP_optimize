import numpy as np
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = 'cuda'


class ConvLayer(nn.Module):
    def __init__(self, node_fea_len, edge_fea_len):
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len  # 여기서 fc layer 하나더 추가해도 될듯
        
        self.fc_full = nn.Linear(2 * self.node_fea_len + self.edge_fea_len,
                                 2 * self.node_fea_len).to(device)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.aplha = nn.Parameter(torch.tensor(0.7,dtype=torch.float32))  # Learnable weight parameter
        self.initialize_weights()

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        B,N, M = edge_fea_idx.shape # 2,9,3
        # convolution
        tensor_list=[]
        for i in range(B):
            tensor_list.append(node_in_fea[i][edge_fea_idx[i],:].unsqueeze(0))
        
        node_edge_fea = torch.cat(tensor_list,dim=0)  # edge fea idx -> 시작 노드 -도착 노드 2,9,3,16     2,9,16

        total_nbr_fea = torch.cat([node_in_fea.unsqueeze(2).expand(B, N, M, self.node_fea_len), node_edge_fea, edge_fea],
                                  dim=3)  # 2,9,3,32    2,9,3,32,  2,9,3,32

        total_gated_fea = self.fc_full(total_nbr_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=3)  #2,9,3,32  2,9,3,32
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        mask = torch.where(edge_fea_idx < 0, torch.tensor(0), torch.tensor(1)) #2,9,3,1
       
        nbr_filter = nbr_filter * mask.unsqueeze(3)
        
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=2) #2,9,3,32
        out = self.softplus(self.aplha*node_in_fea + nbr_sumed) #2,9,32

        return out  #2,9,32


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_node_fea_len, orig_edge_fea_len, edge_fea_len, node_fea_len,
                 final_node_len, dis):
        super(CrystalGraphConvNet, self).__init__()
        self.embedding_n = nn.Linear(orig_node_fea_len, node_fea_len).to(device)
        self.embedding_e = nn.Linear(orig_edge_fea_len, edge_fea_len).to(device)
        self.dis=dis
        N=dis.shape[0]
        self.convs1 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs2 = ConvLayer(node_fea_len, edge_fea_len)
        self.convs3 = ConvLayer(node_fea_len, edge_fea_len)
        self.final_layer = nn.Linear(node_fea_len,int(final_node_len/2)).to(device)
        self.concat2fc = nn.Linear(final_node_len*N,256).to(device)
        self.readout1 = nn.Linear(256, 128).to(device)
        self.readout2 = nn.Linear(128, 64).to(device)
        self.fc_out = nn.Linear(64, 1).to(device)
        self.DA_weight = nn.Parameter(torch.tensor(48/5,dtype=torch.float32))  # Learnable weight parameter
        self.DA_bias = nn.Parameter(torch.tensor(-28/5,dtype=torch.float32))
        self.DA_act=nn.Sigmoid()

        self.act_fun = nn.ELU()


        self.initialize_weights()
                     
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, node_fea, edge_fea, edge_fea_idx):
        node_fea = self.embedding_n(node_fea)  # N,fea
        edge_fea = self.embedding_e(edge_fea)
        node_fea = self.convs1(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs2(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs3(node_fea, edge_fea, edge_fea_idx)
        
        node_fea=self.final_layer(node_fea)
        node_clone=node_fea.clone()
        DA_matrix=self.DA_act(self.DA_weight*self.dis+self.DA_bias) #9,9
        node1=torch.matmul(DA_matrix,node_clone) #2,9,16
        node_final=torch.cat([node_fea,node1],dim=2)  
        return node_final #2,9,32

    def readout(self, node_fea):
        B,N,F=node_fea.shape
        node_fea = self.concat2fc(node_fea.reshape(B,-1))
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout1(node_fea)
        node_fea = self.act_fun(node_fea)
        node_fea = self.readout2(node_fea)
        node_fea = self.act_fun(node_fea)
        out = self.fc_out(node_fea)
        return out


class MLP(nn.Module):
    def __init__(self, state_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화를 사용하여 가중치를 초기화합니다.
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # 편향을 0으로 초기화합니다.
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 배치 정규화 레이어의 가중치를 초기화합니다.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PPO(nn.Module):
    def __init__(self, learning_rate, lmbda, gamma, alpha, beta, epsilon, discount_factor,
                 location_num,dis):
        super(PPO, self).__init__()
        self.node_fea_len = 32
        self.final_node_len=32
        self.edge_fea_len = 32
        self.gnn = CrystalGraphConvNet(orig_node_fea_len=4, orig_edge_fea_len=5, edge_fea_len=self.edge_fea_len, node_fea_len=self.node_fea_len, final_node_len=32, dis=dis)
        self.pi = MLP(self.final_node_len + 10 + 5 + 5, 1).to(device)
        self.temperature = nn.Parameter(torch.tensor(1.5,dtype=torch.float32))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.beta = beta
        self.epsilon = epsilon

    def calculate_GNN(self, node_fea, edge_fea, edge_fea_idx):
        return self.gnn(node_fea, edge_fea, edge_fea_idx)

    def calculate_pi(self, state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        # state gnn 9,32
        # node_fea 9,13
        # edge_fea 9,3,5
        # edge_fea_idx 9, 3
        # distance 9,3
        # tp_type float=> 9 3 5
        B,N,E=edge_fea_idx.shape
        tensor_list=[]
        for i in range(B):
            tensor_list.append(state_gnn[i][edge_fea_idx[i],:].unsqueeze(0)) #N,E,32
        
        action_variable = torch.cat(tensor_list,dim=0) # 2, 9, 3 32
        
        #action_variable = state_gnn[edge_fea_idx, :] #9,3,32
        
        edge_fea_tensor = edge_fea.repeat(1, 1, 1, 2) #2. 9,3,10

        distance_tensor = distance.unsqueeze(3).repeat(1, 1, 1, 5) #2 9,3 5

        action_variable = torch.cat([action_variable, edge_fea_tensor], dim=3)

        action_variable = torch.cat([action_variable, distance_tensor], dim=3)

        action_variable = torch.cat(
            (action_variable, tp_type.view(B, 1, 1, 1).expand(B, N, E, 5)), dim=3) # 2 9,3,52
        action_probability = self.pi(action_variable) #2 9,3,1
        
        return action_probability

    def get_action(self, node_fea, edge_fea, edge_fea_idx, mask, distance, tp_type):
        with torch.no_grad():
            B, N, M = edge_fea_idx.shape
            # print(edge_fea_idx) 1,9,3
            # print(edge_fea) 1,9,3,5
            # print(node_fea) 1,9,4
            state = self.calculate_GNN(node_fea, edge_fea, edge_fea_idx)  #state 1,9, 32
            # print(state)
            probs = self.calculate_pi(state, node_fea, edge_fea, edge_fea_idx, distance, tp_type) #2 9,3,1
            # print(probs) # type0 weight 작다
            logits_masked = probs - 1e8 * mask #1 9,3,1
            # print(logits_masked)
            prob=torch.softmax((logits_masked.flatten()-torch.max(logits_masked.flatten()))/self.temperature,dim=-1)
            m = Categorical(prob)
            action = m.sample()
            i = int(action / M)
            j = int(action % M)
            return action, i, j, prob[action]

    def calculate_v(self, x):
        return self.gnn.readout(x)

    def update(self, nf_list,ef_list,efi_list,distance_list,type_list,mask_list, probs, rewards, dones, starts, actions, ep_num,step1,model_dir):
        num = 0
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        # data-> episode-> state [30,16,5]  node_fea (9,13),edge_fea (9,3,5),edge_fea_idx(9,3), distance (9,3) type (1), mask(9,3), 
        # nf_list 1000 9 4
        # ef_list 1000 9 3 5
        # efi_list 1000 9 3
        # distance_list 1000,9,3
        #type_list 1000,1
        # mask_list 1000, 9 , 3.1


        
        # probs [30*15] numpy
        # rewards [30*15]
        # action [30*15]
        # dones [30*15]
        # starts []
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        actions= torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        starts = torch.tensor(starts, dtype=torch.float32).unsqueeze(1).to(device)

       
        B,N,M=efi_list.shape
        state_gnn = self.calculate_GNN(nf_list, ef_list, efi_list) #1050,9,32
        
        # "done" 텐서에서 값이 1인 배치 인덱스 찾기
        selected_indices_ex_done = (dones.squeeze() == 1).nonzero().squeeze()
        selected_indices_ex_start = (starts.squeeze() == 1).nonzero().squeeze()

        
        prob_a=self.calculate_pi(state_gnn,nf_list,ef_list,efi_list,distance_list,type_list) #1000,9,3,1
        logits_masked = prob_a - 1e8 * mask_list # 1050, 9, 3,1
        
        logits_masked =logits_masked.reshape(B,-1) #1050 27
        logits_masked=logits_masked[selected_indices_ex_done] #1000 27  
        
        new_B,new_N=logits_masked.shape
        prob=torch.softmax((logits_masked-torch.max(logits_masked,dim=1)[0].reshape(new_B,-1))/self.temperature,dim=1)
        pr_a=torch.gather(prob,1,actions) #1000 1
        v_value = self.calculate_v(state_gnn) #1050 1
        
        v_value=v_value*dones
        state_v=v_value[selected_indices_ex_done] #1000 1
        state_next_v = v_value[selected_indices_ex_start]#1000 1
        
        td_target = rewards + self.gamma * state_next_v 
        delta = td_target - state_v
        
        advantage_lst = torch.zeros(new_B,1)
        
        
        ep_len=int(new_B/ep_num)
        j=0
        for i in range(ep_num):
            advantage = 0.0
            for t in reversed(range(j, j + ep_len)):
                advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                advantage_lst[t][0] = advantage
            j += ep_len
        
        ratio = torch.exp(torch.log(pr_a) - torch.log(probs))  # a/b == exp(log(a)-log(b))
        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1%100==0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    print(f"{name} gradient mean: {param.grad.abs().mean().item()}")

        if step1 % 1000 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step) + '.pth')

        return ave_loss, v_loss, p_loss
