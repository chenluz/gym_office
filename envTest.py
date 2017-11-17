#!/usr/bin/env python
# coding: utf-8
import office_control.envs as office_env
import gym
import time
import numpy as np



def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    total_reward = 0
    num_steps = 0
    while True:
        current_step = np.random.choice(env.nA)
        nextstate, reward, done, debug_info = env.step(current_step)
        print(nextstate, reward, done)

        total_reward += reward
        num_steps += 1

        time.sleep(60)

    return total_reward, num_steps




def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nR, env.nA))


