import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        # multi fully connected layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) #(input, output)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) #Adam optimizer apply to the network
        self.loss = nn.MSELoss() # Mean Square Loss applied
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # if cuda enabled, gpu activated
        self.to(self.device)

    def save_check_point(self, name):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), os.path.join("models", name))

    def load_check_point(self, name):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(os.path.join("models", name)))

    def forward(self, state):
        # relu activation function applied to layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x) # dont use activation, want to have raw number
        return actions

   

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size= 100_000, eps_end = 0.01, eps_dec=5e-4, name="LunarLander-v2"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)] # [1,2,...]
        self.mem_size = max_mem_size # max memory size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.name = name
        self.mem_cntr = 0 # memory counter, track of position of first available memory

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # rechange into replay buffer later
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def save_model(self):
        self.Q_eval.save_check_point(self.name)
    
    def load_model(self):
        self.Q_eval.load_check_point(self.name)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size # over 100,000, go back to position 0 and rewirte
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation], dtype=np.float32), dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # replcae False means cannot be selected multiple times
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) 
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] 
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # max return value, index, zeroth element value. greedy action
        # using bellman equation Q*(s,a) = E[R(t+1) + gamma * max(Q*(s', a'))]
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] 
      

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min # epsilon greedy





