# Automated Feature Selection: A Reinforcement Learning Perspective
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
import torch.utils.data as Data

from sklearn.metrics import normalized_mutual_info_score

from feature_env import FeatureEvaluator

import warnings

from utils.logger import info

warnings.filterwarnings("ignore")

GMM = GaussianMixture(n_components=2)
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100  # After how much time you refresh target network
MEMORY_CAPACITY = 200  # The size of experience replay buffer
EXPLORE_STEPS = 3000  # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY, feature


def Feature_DB(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix

def _feature_state_generation_des(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
            describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix

def Feature_GCN(X):
    corr_matrix = X.corr().abs()
    corr_matrix[np.isnan(corr_matrix)] = 0
    corr_matrix_ = corr_matrix - np.eye(len(corr_matrix), k=0)
    sum_vec = corr_matrix_.sum()

    for i in range(len(corr_matrix_)):
        corr_matrix_.iloc[:, i] = corr_matrix_.iloc[:, i] / sum_vec[i]
        corr_matrix_.iloc[i, :] = corr_matrix_.iloc[i, :] / sum_vec[i]
    W = corr_matrix_ + np.eye(len(corr_matrix), k=0)
    Feature = np.mean(np.dot(X.values, W.values), axis=1)

    return Feature


class AutoEncoder(nn.Module):
    def __init__(self, N_feature, N_HIDDEN=4):
        self.N_feature = N_feature
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.N_feature, N_HIDDEN * 4),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 4, N_HIDDEN * 2),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 2, N_HIDDEN)
        )
        self.decoder = nn.Sequential(
            nn.Linear(N_HIDDEN, N_HIDDEN * 2),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 2, N_HIDDEN * 4),
            nn.Tanh(),
            nn.Linear(N_HIDDEN * 4, self.N_feature)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def Feature_AE(X, N_HIDDEN=4):
    N_feature = X.shape[1]
    autoencoder = AutoEncoder(N_feature, N_HIDDEN=N_HIDDEN).cuda(0)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()

    X_tensor = torch.Tensor(X.values).cuda(0)
    train_loader = Data.DataLoader(dataset=X_tensor, batch_size=104, shuffle=True)
    # X_variable = Variable(X)
    for epoch in range(10):
        for x in train_loader:
            b_x = Variable(x.view(-1, N_feature)).float()
            # b_y = Variable(x.view(-1,N_feature)).float() #same as b_x in auto-encoder
            encoded, decoded = autoencoder.forward(b_x)
            optimizer.zero_grad()
            loss = loss_func(decoded, b_x)
            loss.backward()
            optimizer.step()
            # print(loss.item())
    #     X_encoded = autoencoder(X_tensor)[0].view(-1).detach().numpy()
    X_encoded = autoencoder.forward(X_tensor)[0][0].cpu().detach().numpy()
    return X_encoded


def cal_relevance(X, y):
    if len(X.shape) == 1:
        return normalized_mutual_info_score(X, y)
    else:
        N_col = X.shape[1]
        _sum = 0
        for i in range(N_col):
            _sum += normalized_mutual_info_score(X.iloc[:, i], y)
        return _sum / X.shape[1]


def cal_redundancy(X):
    if len(X.shape) == 1:
        return 1
    else:
        N_col = X.shape[1]
        _sum = 0
        for i in range(N_col):
            for j in range(N_col):
                _sum += normalized_mutual_info_score(X.iloc[:, i], X.iloc[:, j])
        return _sum / X.shape[1] ** 2


# %%
class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


# %%
class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def gen_marlfs(feature_env, N_STATES=64, N_ACTIONS=2, EPISODE=-1, EXPLORE_STEPS=30):
    # info(f'begin the task {feature_env.task_name} on EPISODE {EPISODE}')
    np.random.seed(0)
    N_feature = feature_env.ds_size
    dqn_list = []
    for agent in range(N_feature):
        dqn_list.append(DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS))
    results = []
    action_list = np.random.randint(2, size=N_feature)
    i = 0
    while sum(action_list) < 2:
        np.random.seed(i)
        action_list = np.random.randint(2, size=N_feature)
        # i += 1

    result = feature_env.report_performance(action_list, flag='train', rp=False)
    # info(f'current selection is {action_list}')
    results.append([result, action_list])
    state = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
    for i in tqdm.tqdm(range(EXPLORE_STEPS)):
        # info(f'begin the task {feature_env.task_name} on EPISODE {i}')
        action_list = np.zeros(N_feature)
        for agent, dqn in enumerate(dqn_list):
            action_list[agent] = dqn.choose_action(state)
        while sum(action_list) < 2:
            np.random.seed(i)
            action_list = np.random.randint(2, size=N_feature)
            # i += 1

        result = feature_env.report_performance(action_list, flag='train', rp=False)
        info(f'store train set {feature_env.__len__()} in step {i}')
        # info(f'current selection is {action_list}')
        state_ = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
        # gen_dataset = feature_env.generate_data(action_list, flag='train')
        # tmp_X = gen_dataset.iloc[:, :-1]
        # tmp_Y = gen_dataset.iloc[:, -1]

        r_list = result * action_list
            # (result + cal_relevance(tmp_X, tmp_Y) - cal_redundancy(tmp_X)) / sum(
            #     action_list) * action_list

        for agent, dqn in enumerate(dqn_list):
            dqn.store_transition(state, action_list[agent], r_list[agent], state_)

        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
            # info('DQN learning!')
            for dqn in dqn_list:
                dqn.learn()
        state = state_
        results.append([result, action_list])
    max_accuracy = 0
    optimal_set = []
    for i in range(len(results)):
        if results[i][0] > max_accuracy:
            max_accuracy = results[i][0]
            optimal_set = results[i][1]
    
    test_max_accuracy = feature_env.report_performance(optimal_set, flag='test', store=False)
    info("The optimal accuracy is: {}, the optimal selection for MARLFS is:{}".format(test_max_accuracy, optimal_set))
    return max_accuracy, optimal_set, int(optimal_set.sum())


if __name__ == '__main__':
    task_name = 'openml_586'
    fe = FeatureEvaluator(task_name)
    print(gen_marlfs(fe, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=2))
    info(len(fe))

# %%
