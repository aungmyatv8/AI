from fileinput import filename
from unicodedata import name
import gym
from network import Agent
import numpy as np
import sys
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = "LunarLander-v2" if sys.argv[-1] == "lunar" else "CartPole-v0"

    print("Environment {}".format(env_name))
    
    env = gym.make(env_name)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, eps_end=0.01, input_dims=[env.observation_space.shape[0]], lr=0.003, name=env_name)
    scores, eps_history = [],[]
    n_games = 500
        
    best_score = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            # print("saved checkpoint", avg_score, best_score)
            agent.save_model()
            best_score = avg_score

        print('episode', i, 'score %2.f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon )

    x = [i+1 for i in range(n_games)]
    filename = '{}.png'.format(env_name)
    plot_learning_curve(x, scores, eps_history, filename)
