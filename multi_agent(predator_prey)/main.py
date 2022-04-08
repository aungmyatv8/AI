import numpy as np
from make_env import make_env
from maddpg_torch import MADDPG, MultiAgentReplayBuffer, obs_list_to_state_vector
from utils import plot_learning_curve

if __name__ == "__main__":
    scenairo = "simple_tag"
    env = make_env(scenairo)
    n_agents = env.n  # 4 agents, good agents(green) and 3 adversaries(red)

    #env.observation_space has [Box(16,), Box(16,), Box(16,), Box(14,)]
    #env.action_space has [Discrete(5), Discrete(5), Discrete(5), Discrete(5)] for each agent
   
    actor_dims = [] # actor_dims will be # [16, 16, 16, 14]
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims) # 62 = 16 + 16+ 16 + 14

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n # 5
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, scenairo=scenairo,)
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)
    Print_Interval = 100
    n_games = 5000
    max_steps = 25
    total_stpes = 0
    score_history = []
    evalute = True # train or test
    best_score = 0
    each_agent_reward_history = {
            "agent0": [],
            "agent1": [],
            "agent2": [],
            "agent3": [],

    }

    if evalute:
        maddpg_agents.load_check_point()

    
    
    for i in range(n_games):
        obs = env.reset()
        total_score = 0
        each_agent_score = np.zeros(4)
        done = [False] * n_agents
        episode_step = 0 
        while not any(done):
            if evalute:
                env.render()
            # env.render()
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step > max_steps:
                done = [True] * n_agents 
            
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_stpes % 100 == 0 and not evalute:
                maddpg_agents.learn(memory)

            obs = obs_
            total_score += sum(reward)
            each_agent_score = each_agent_score + np.array(reward)
           
            total_stpes += 1
            episode_step += 1
        score_history.append(total_score)
        each_agent_reward_history["agent0"].append(each_agent_score[0])
        each_agent_reward_history["agent1"].append(each_agent_score[0])
        each_agent_reward_history["agent2"].append(each_agent_score[0])
        each_agent_reward_history["agent3"].append(each_agent_score[-1])
        avg_score = np.mean(score_history[-100: ])
        
        if not evalute:
            if avg_score > best_score:
                maddpg_agents.save_check_point()
                best_score = avg_score
        if i % Print_Interval == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score)) # print every 100 steps
    plot_learning_curve(n_games, each_agent_reward_history)