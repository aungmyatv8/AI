import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,n_actions, n_agents, batch_size):
        self.memory_size = max_size
        self.mem_counter = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.memory_size, critic_dims)) # for critic
        self.new_state_memory = np.zeros((self.memory_size, critic_dims)) # to keep track of actor stuff
        self.reward_memory = np.zeros((self.memory_size, n_agents)) 

      
        self.terminal_memory = np.zeros((self.memory_size, n_agents), dtype=bool) #critic value for terminal state is always zero
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_action_memory.append(
                np.zeros((self.memory_size, self.n_actions))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.memory_size, self.actor_dims[i]))
            )
            self.actor_state_memory.append(
                np.zeros((self.memory_size, self.actor_dims[i])))


    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_counter % self.memory_size # position

        # store to appropriate position
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self):
        max_memory = min(self.mem_counter, self.memory_size) #highest position filled in memory
        batch = np.random.choice(max_memory, self.batch_size, replace=False) #  replace false sure that don't get the same memory twice

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # handle states and actions for the actors
        actor_states = []
        actor_new_states = [] 
        actions = []
   
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch]) 
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    """
    determine whether or not we're allowed to sample a memory based on
    whether or not we have filled up the batch size of our memory
    """
    def ready(self):
        if self.mem_counter >= self.batch_size:
            return True
        else:
            return False


class CriticNetwork(nn.Module):
    """
    learing rate called beta, reason being that actor_critic can have separate learning rate
    although in their implementation as well as mine we just use the same learning for both
    actor and critic but it is entirely possible to have separate ones   

    each agent has four diferent networks, actor-critic, target actor-critic

    checkpoints are saving the correct model to correct file and more importantly when we're loading those models
    that we load the correct model it from the correct file 
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpk_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpk_dir, name)
        self.fc1 = nn.Linear(input_dims+ n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_check_point(self):
        T.save(self.state_dict(), self.chkpt_file)
    
    def load_check_point(self):
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpk_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpk_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions) # policy

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1) #

        return pi

    def save_check_point(self):
        T.save(self.state_dict(), self.chkpt_file)
    
    def load_check_point(self):
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpk_dir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' %agent_idx

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions=n_actions, name=self.agent_name+"_actor", chkpk_dir=chkpk_dir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, name=self.agent_name+'_critic', chkpk_dir=chkpk_dir)

        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions=n_actions, name=self.agent_name+"_target_actor", chkpk_dir=chkpk_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, name=self.agent_name+'_target_critic', chkpk_dir=chkpk_dir)
        
        self.update_network_parameters(tau=1) # when we start our simulation, we want to directly copy the weight of our network to target network


    def update_network_parameters(self, tau=None): 
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        """
        iterate over both the actor and target actor, nam_param performs the application 
        with tau and sum them up and upload the dict to the target actor
         """
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1-tau) *target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
        
        # for critic network
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

 
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1-tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device) # take our conversation and convert it to tensor and add a batch dimension as pytorch expect
        actions = self.actor.forward(state) 
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0] # we can't pass a pytorch tensor into the env. expect function actually get out our actual numpy array and return it as a tuple of the numpy array so we have to zeroth index  

    def save_models(self):
        self.actor.save_check_point()
        self.target_actor.save_check_point()
        self.critic.save_check_point()
        self.target_critic.save_check_point()

    def load_models(self):
        self.actor.load_check_point()
        self.target_actor.load_check_point()
        self.critic.load_check_point()
        self.target_critic.load_check_point()


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, scenairo="simple", alpha=0.01, beta=0.01, 
                fc1=64, fc2=64, gamma=0.99, tau=0.01, chpt_dir="model/"):
                self.agents = [] # to track adversary and two cooperting agent
                self.n_agents = n_agents
                self.n_actions = n_actions
                chpt_dir += scenairo # keep the different agents train for different scenaiors or environment 

                for agent_idx in range(self.n_agents):
                    # self.agents.append(Agent())
                    self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, chpt_dir, alpha=alpha, beta=beta,
                            ))
    
    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)

        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten() # for new states
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_ # critic update equation
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

            

    def save_check_point(self):
        print("... saving all agents checkpoint ...")
        for agent in self.agents:
            agent.save_models()
    
    def load_check_point(self):
        print('... loading all agents checkpoint')
        for agent in self.agents:
            agent.load_models()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state
    
