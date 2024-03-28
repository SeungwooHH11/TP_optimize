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
    def __init__(self, node_fea_len, edge_fea_len, out_fea_len):
        super(ConvLayer, self).__init__()
        self.node_fea_len = node_fea_len
        self.edge_fea_len = edge_fea_len  # 여기서 fc layer 하나더 추가해도 될듯
        self.out_fea_len = out_fea_len
        self.fc_full = nn.Linear(2 * self.node_fea_len + self.edge_fea_len,
                                 2 * self.out_fea_len).to(device)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.out_fea_len).to(device)
        self.bn2 = nn.BatchNorm1d(self.out_fea_len).to(device)
        self.softplus2 = nn.Softplus()

    def forward(self, node_in_fea, edge_fea, edge_fea_idx):
        N, M = edge_fea_idx.shape
        # convolution
        node_edge_fea = node_in_fea[edge_fea_idx, :]  # edge fea idx -> 시작 노드 -도착 노드

        total_nbr_fea = torch.cat([node_in_fea.unsqueeze(1).expand(N, M, self.node_fea_len), node_edge_fea, edge_fea],
                                  dim=2)

        total_gated_fea = self.fc_full(total_nbr_fea)

        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.out_fea_len * 2)).view(N, M, self.out_fea_len * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        mask = torch.where(edge_fea_idx < 0, torch.tensor(0), torch.tensor(1))
        nbr_filter = nbr_filter * mask.unsqueeze(2)
        nbr_core = nbr_filter * mask.unsqueeze(2)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(node_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_node_fea_len, edge_fea_len, h_fea_len,
                 node_fea_len, n_conv=3, n_h=2):
        super(CrystalGraphConvNet, self).__init__()
        semi_fea_len = int(node_fea_len)
        self.embedding = nn.Linear(orig_node_fea_len, semi_fea_len).to(device)

        self.convs1 = ConvLayer(node_fea_len, edge_fea_len, node_fea_len)
        self.convs2 = ConvLayer(node_fea_len, edge_fea_len, node_fea_len)
        self.convs3 = ConvLayer(node_fea_len, edge_fea_len, node_fea_len)
        self.readout1 = nn.Linear(h_fea_len, h_fea_len).to(device)
        self.readout2 = nn.Linear(h_fea_len, h_fea_len).to(device)

        self.conv_to_fc = nn.Linear(semi_fea_len, h_fea_len).to(device)
        self.act_fun = nn.ELU()

        self.fc_out = nn.Linear(h_fea_len, 1).to(device)

    def forward(self, node_fea, edge_fea, edge_fea_idx):
        node_fea = self.embedding(node_fea)  # N,fea
        node_fea = self.convs1(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs2(node_fea, edge_fea, edge_fea_idx)
        node_fea = self.convs3(node_fea, edge_fea, edge_fea_idx)
        # node1=torch.matmul(distance_rev,node_fea)
        # node_final=torch.cat([node_fea,node1],dim=1)
        return node_fea

    def readout(self, node_fea):
        node_fea = self.conv_to_fc(node_fea)
        node_fea = torch.sum(node_fea, 1)  # batch
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
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PPO(nn.Module):
    def __init__(self, learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, epsilon=0.2, discount_factor=1,
                 location_num=15):
        super(PPO, self).__init__()
        self.node_fea_len = 128
        self.gnn = CrystalGraphConvNet(4 + location_num, 5, self.node_fea_len, self.node_fea_len, 3, 2)
        self.pi = MLP(self.node_fea_len + 10 + 5 + 5, 1).to(device)
        self.optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.beta = beta
        self.epsilon = epsilon

    def calculate_GNN(self, node_fea, edge_fea, edge_fea_idx):
        return self.gnn(node_fea, edge_fea, edge_fea_idx)

    def calculate_pi(self, state_gnn, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        # node_fea 9,13
        # edge_fea 9,3,5
        # edge_fea_idx 9, 3
        # distance 9,3
        # tp_type float=> 9 3 5
        action_variable = state_gnn[edge_fea_idx, :]
        edge_fea_tensor = edge_fea.repeat(1, 1, 2)

        distance_tensor = distance.unsqueeze(2).repeat(1, 1, 5)

        action_variable = torch.cat([action_variable, edge_fea_tensor], 2)

        action_variable = torch.cat([action_variable, distance_tensor], 2)

        action_variable = torch.cat(
            (action_variable, torch.full((edge_fea_idx.shape[0], edge_fea_idx.shape[1], 5), tp_type).to(device)), dim=2)
        action_probability = self.pi(action_variable)
        return action_probability

    def get_action(self, node_fea, edge_fea, edge_fea_idx, distance, tp_type):
        with torch.no_grad():
            N, M = edge_fea_idx.shape
            # print(edge_fea_idx)
            # print(edge_fea)
            # print(node_fea)
            state = self.calculate_GNN(node_fea, edge_fea, edge_fea_idx)
            # print(state)
            probs = self.calculate_pi(state, node_fea, edge_fea, edge_fea_idx, distance, tp_type)
            # print(probs) # type0 weight 작다
            mask = torch.where((edge_fea_idx >= 0) & (edge_fea[:, :, 4] <= tp_type), torch.tensor(0),
                               torch.tensor(1)).unsqueeze(2)

            logits_masked = probs - 1e8 * mask
            # print(logits_masked)
            prob = torch.softmax(logits_masked.flatten() - torch.max(logits_masked.flatten()), dim=-1)
            m = Categorical(prob)
            action = m.sample().item()
            i = int(action / M)
            j = int(action % M)
            while edge_fea_idx[i][j] < 0:
                action = m.sample().item()
                i = int(action / M)
                j = int(action % M)
            return action, i, j, prob[action]

    def calculate_v(self, x):
        return self.gnn.readout(x)

    def update(self, data, probs, rewards, action, dones,step1,model_dir):
        num = 0
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        p_loss = 0
        # data-> episode-> state [30,16,4]  node_fea (9,13),edge_fea (9,3,5),edge_fea_idx(9,3),distance (9,3) type (1)
        # probs [30*15] numpy
        # rewards [30*15]
        # action [30*15]
        # dones [30*15]
        probs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        tr = 0
        step = 0
        sw = 0
        for episode in data:
            for state in episode:

                # node_fea=torch.tensor(state[0],dtype=torch.float32).to(device)
                # edge_fea=torch.tensor(state[1],dtype=torch.float32).to(device)
                # edge_fea_idx=torch.tensor(state[2],dtype=torch.int32).to(device)
                # distance=torch.tensor(state[3],dtype=torch.float32).to(device)
                # type
                state_gnn = self.calculate_GNN(state[0], state[1], state[2])

                if step < len(episode) - 1:
                    prob_a = self.calculate_pi(state_gnn, state[0], state[1], state[2], state[3], state[4])
                    mask = torch.where((state[2] >= 0) & (state[1][:, :, 4] <= state[4]), torch.tensor(0),
                                       torch.tensor(1)).unsqueeze(2).to(device)
                    logits_maksed = prob_a - 1e8 * mask
                    prob = torch.softmax(logits_maksed.flatten() - torch.max(logits_maksed.flatten()), dim=-1)
                    pi_a = prob[int(action[sw])]
                    sw += 1
                    if tr == 0:
                        pi_a_total = pi_a.unsqueeze(0)
                    else:
                        pi_a_total = torch.cat([pi_a_total, pi_a.unsqueeze(0)])
                state_gnn = state_gnn.unsqueeze(0)
                if tr == 0:
                    state_GNN = state_gnn
                elif tr == 1:
                    next_state_GNN = state_gnn
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == 0:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                elif step == len(episode) - 1:
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                else:
                    state_GNN = torch.cat([state_GNN, state_gnn])
                    next_state_GNN = torch.cat([next_state_GNN, state_gnn])
                tr += 1
                step += 1
            step = 0

        total_time_step = sw

        state_v = self.calculate_v(state_GNN)
        state_next_v = self.calculate_v(next_state_GNN)
        td_target = rewards + self.gamma * state_next_v * dones
        delta = td_target - state_v

        advantage_lst = np.zeros(total_time_step)
        advantage_lst = torch.tensor(advantage_lst, dtype=torch.float32).unsqueeze(1).to(device)
        for episode in data:
            advantage = 0.0
            i = 0
            for t in reversed(range(i, i + len(episode))):
                advantage = self.gamma * self.lmbda * advantage + delta[t][0]
                advantage_lst[t][0] = advantage
            i += len(episode)
        ratio = torch.exp(torch.log(pi_a_total.unsqueeze(1)) - torch.log(probs))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage_lst
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_lst
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_v, td_target.detach()) * self.alpha

        ave_loss = loss.mean().item()
        v_loss = (self.alpha * F.smooth_l1_loss(state_v, td_target.detach())).item()
        p_loss = -torch.min(surr1, surr2).mean().item()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        if step1 % 200 == 0:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),

            }, model_dir+'trained_model' + str(step) + '.pth')

        return ave_loss, v_loss, p_loss




