#!/usr/bin/env python
# coding: utf-8

import gym
import time
import numpy as np
import office_control.envs as office_env
#import MC.EpsilonGreedy as MCE
import TD.QLearning as QL
import NN.DQN as DQN
#import FA.QLearning_FA as LQL
from lib import plotting
#import envTest
import argparse
import os



def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  
    parser = argparse.ArgumentParser(description='Run Reinforcment Learning on Chamber')
    parser.add_argument('--env', default='office_control-v2', help='Office env name')
    parser.add_argument('-o', '--output', default='chamber-v2', help='Directory to save data to')
    parser.add_argument('--num', default=2000, help='Number of Episodes')
    parser.add_argument('--df', default=0.95, help='Discount Factor')
    parser.add_argument('--alpha', default=0.5, help='Constant step-size parameter')
    parser.add_argument('--epsilon', default=0.9, help='Epsilon greedy policy')
    parser.add_argument('--epsilon_min', default=0.1, help='Smallest Epsilon that can get')
    parser.add_argument('--epsilon_decay', default=0.93, help='Epsilon decay after the number of episodes')
    parser.add_argument('--batch_size', default=32, help='Sampling batch size')
    parser.add_argument('--lr', default=0.01, help='Learning rate')


    args = parser.parse_args()

    output = get_output_folder(args.output, args.env)

    print(output)

    #create environment
    env = gym.make(args.env)

    #Q, policy = MCE.mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
    Q, stats = QL.q_learning(env, int(args.num), float(args.df), float(args.alpha), float(args.epsilon), 
        float(args.epsilon_min),  float(args.epsilon_decay), output)
    plotting.plot_episode_stats(stats, smoothing_window=1)
    print(Q)
    # # estimator = LQL.Estimator(env)
    # stats = LQL.q_learning(env, estimator, int(args.num),  float(args.df),  float(args.epsilon),
    #     float(args.epsilon_decay))

    #envTest.run_random_policy(env)
    # state_size = env.nS
    # action_size = env.nA
    # agent = DQN.DQNAgent(state_size, action_size, float(args.df), float(args.lr))
    # #DQN.test_model(env, agent)
    # stats, model = DQN.q_learning(env, agent, int(args.num), int(args.batch_size),
    #     float(args.epsilon), float(args.epsilon_min), float(args.epsilon_decay), output)


    #plotting.plot_episode_stats(stats, smoothing_window=1)




if __name__ == '__main__':
    main()
