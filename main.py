#!/usr/bin/env python
# coding: utf-8

import gym
import time
import numpy as np
import MC.EpsilonGreedy as MCE
import TD.QLearning as QL
import FA.QLearning_FA as LQL
from lib import plotting
import envTest
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
    parser.add_argument('--env', default='office_control-v1', help='Office env name')
    parser.add_argument('-o', '--output', default='chamber-v1', help='Directory to save data to')
    parser.add_argument('--num', default=500, help='Number of Episodes')
    parser.add_argument('--df', default=1.0, help='Discount Factor')
    parser.add_argument('--alpha', default=0.5, help='Constant step-size parameter')
    parser.add_argument('--epsilon', default=0.9, help='Epsilon greedy policy')
    parser.add_argument('--epsilon_decay', default=0.9, help='Epsilon decay after the number of episodes')


    args = parser.parse_args()

    output = get_output_folder(args.output, args.env)

    print(output)

    #create environment
    env = gym.make(args.env)

    #Q, policy = MCE.mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
    # Q, stats = QL.q_learning(env, int(args.num), float(args.df), float(args.alpha), float(args.epsilon),  
    #     float(args.epsilon_decay), output)
    # plotting.plot_episode_stats(stats)
    # print(Q)
    estimator = LQL.Estimator(env)
    stats = LQL.q_learning(env, estimator, int(args.num),  float(args.df),  float(args.epsilon),
        float(args.epsilon_decay))
    plotting.plot_episode_stats(stats)


if __name__ == '__main__':
    main()
