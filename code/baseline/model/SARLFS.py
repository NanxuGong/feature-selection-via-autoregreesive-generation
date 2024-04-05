# Single-Agent Reinforcement Learning Feature Selection
# 10.1109/ICMLA.2009.71
# Automatic Feature Selection for Model-Based Reinforcement Learning in Factored MDPs
# In SARLFS [15], the agent learns a KWIK (Knows What It Knows) model,
# which is represented by a dynamic Bayesian network,
# deduces a minimal feature set from this network,
# and computes a policy on this feature subset using dynamic programming methods.
# In the experiments, the two accuracy thresholds in the KWIK are set to  = 0.15, δ = 0.10.
import tqdm

from MARLFS import Feature_AE, cal_relevance, cal_redundancy, EPSILON, LR, \
    TARGET_REPLACE_ITER, BATCH_SIZE, GAMMA, Feature_DB
from feature_env import FeatureEvaluator
from utils.logger import info
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEMORY_CAPACITY = 10


class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 2 * N_STATES)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(2 * N_STATES, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return torch.sigmoid(action_value)


# %%
class SADQN(object):

    def __init__(self, N_STATES, N_ACTIONS, lim=None):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.limit = lim
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS * 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action = np.zeros(self.N_ACTIONS)
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x).squeeze(0).detach().numpy()
            if self.limit is not None:
                action_value[np.argsort(action_value)[:self.limit]] = 0
            action[action_value > 0.5] = 1.
        else:
            score = np.random.random(self.N_ACTIONS)
            if self.limit is not None:
                score[np.argsort(score)[:self.limit]] = 0
            action[score > 0.5] = 1.
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
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
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + self.N_ACTIONS])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + self.N_ACTIONS:self.N_STATES + 2 * self.N_ACTIONS])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def gen_sarlfs(feature_env, k=None, N_STATES=64, EPISODE=-1, EXPLORE_STEPS=50):
    info(f'begin the task {feature_env.task_name} on EPISODE {EPISODE}')
    np.random.seed(0)
    N_feature = feature_env.ds_size

    dqn = SADQN(N_STATES=N_STATES, N_ACTIONS=N_feature, lim=k)
    results = []
    action_list = np.random.randint(2, size=N_feature)
    i = 0
    while sum(action_list) < 2:
        np.random.seed(i)
        action_list = np.random.randint(2, size=N_feature)
        i += 1
    result = feature_env.report_performance(action_list, flag='train', rp=False)
    # info(f'current selection is {action_list}')
    results.append([result, action_list])

    state = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
    for i in tqdm.tqdm(range(EXPLORE_STEPS)):
        action_list = dqn.choose_action(state)
        while sum(action_list) < 2:
            np.random.seed(i)
            action_list = np.random.randint(2, size=N_feature)
            i += 1
        result = feature_env.report_performance(action_list, flag='train', rp=False)
        # info(f'current selection is {action_list}')
        state_ = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
        gen_dataset = feature_env.generate_data(action_list, flag='train')
        tmp_X = gen_dataset.iloc[:, :-1]
        tmp_Y = gen_dataset.iloc[:, -1]

        r = result * action_list
            #  + cal_relevance(tmp_X, tmp_Y) - cal_redundancy(tmp_X)) / sum(
            # action_list) * action_list

        dqn.store_transition(state, action_list, r, state_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            # info('DQN learning!')
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
    info("The optimal accuracy is: {}, the optimal selection for SARLFS is:{}".format(test_max_accuracy,
                                                                                            optimal_set))
    return max_accuracy, optimal_set


def gen_sarlfs_init(feature_env, N_STATES=64, EPISODE=-1, EXPLORE_STEPS=300):
    info(f'begin the task {feature_env.task_name} on EPISODE {EPISODE}')
    np.random.seed(0)
    N_feature = feature_env.ds_size
    info(f'the feature number is {N_feature}')
    dqn = SADQN(N_STATES=N_STATES, N_ACTIONS=N_feature)
    info(f'the state dim is {N_STATES}')
    results = []
    action_list = np.random.randint(2, size=N_feature)
    i = 0
    while sum(action_list) < 2:
        np.random.seed(i)
        action_list = np.random.randint(2, size=N_feature)
        i += 1
    result = feature_env.report_performance(action_list, flag='train')
    info(f'current selection is {action_list}')
    results.append([result, action_list])

    state = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
    for i in tqdm.tqdm(range(EXPLORE_STEPS)):
        action_list = dqn.choose_action(state)
        while sum(action_list) < 2:
            np.random.seed(i)
            action_list = np.random.randint(2, size=N_feature)
            i += 1
        result = feature_env.report_performance(action_list, flag='train', rp=False)
        info(f'current selection is {action_list}')
        state_ = Feature_AE(feature_env.train.iloc[:, :-1].iloc[:, action_list == 1], N_HIDDEN=N_STATES)
        gen_dataset = feature_env.generate_data(action_list, flag='train')
        tmp_X = gen_dataset.iloc[:, :-1]
        tmp_Y = gen_dataset.iloc[:, -1]

        r = result * action_list
            # (result + cal_relevance(tmp_X, tmp_Y) - cal_redundancy(tmp_X)) / sum(
            # action_list) * action_list

        dqn.store_transition(state, action_list, r, state_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            info('DQN learning!')
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
    info("The optimal accuracy is: {}, the optimal selection for SARLFS is:{}".format(test_max_accuracy,
                                                                                            optimal_set))
    return max_accuracy, optimal_set, int(optimal_set.sum())

if __name__ == '__main__':
    task_name = 'ap_omentum_ovary'
    fe = FeatureEvaluator(task_name)
    gen_sarlfs_init(fe)
