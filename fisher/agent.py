import torch
from torch import nn
from copy import deepcopy
import numpy as np

class DQN:
    def __init__(self, base_net, batch_size, n_states, n_actions, memory_capacity=2000, epsilon=0.9, gamma=0.9, rep_frep=100, lr=0.01):
        self.eval_net = base_net
        self.target_net = deepcopy(base_net)

        self.batch_size=batch_size
        self.epsilon=epsilon
        self.gamma=gamma
        self.n_states=n_states
        self.n_actions=n_actions
        self.memory_capacity=memory_capacity
        self.rep_frep=rep_frep

        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        x = torch.FloatTensor(x).unsqueeze(0)  # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.argmax(actions_value, dim=1).numpy()  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train_step(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % self.rep_frep == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        # convert long int type to tensor
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step