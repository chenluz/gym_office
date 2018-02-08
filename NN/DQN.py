import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import itertools
from lib import plotting
from gym import wrappers
import pydot
from keras.utils import plot_model


# ref: https://keon.io/deep-q-learning/
class DQNAgent:
    def __init__(self, state_size, action_size, discount_factor, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = discount_factor   # discount rate
        self.learning_rate = learning_rate#0.001
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #ref: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='sgd')
 
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))
        #(1) mse + adam : saturation, same value for all input, reward increasing
        #(2) mse + sgd : best 41, reward increasing
        #(3) categorical_crossentropy + sdg: reward dereasing
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

  
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        

    def get(self):
        return self.model


    def load(self, name):
        self.model.load_weights(name)
        return self.model


    def save(self, name):
        self.model.save_weights(name)


def make_epsilon_greedy_policy(agent, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        epsilon: The probability to select a random action . float between 0 and 1.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = agent.get().predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, agent, num_episodes, batch_size, epsilon, epsilon_min, epsilon_decay, folder):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
  
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # agent.load("./save/cartpole-dqn.h5")
    done = False


    for i_episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.nS])

        policy = make_epsilon_greedy_policy(agent, epsilon, env.nA)
       
        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            env.my_render()
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t
            next_state = np.reshape(next_state, [1, env.nS])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(i_episode, num_episodes,  stats.episode_rewards[i_episode], epsilon))
            if done:
                
                # if(i_episode%2 == 0):
                #     if epsilon > epsilon_min:
                #         epsilon *= epsilon_decay
                break
            if len(agent.memory) > batch_size:
                    agent.replay(batch_size)    
    agent.save("office_mmch409-dqn.h5")           

    return stats, agent.get()



def test_model(env, agent):
    model = agent.load("office_simulator-dqn.h5")
    ob, state = env.get_state(25, 25, 8, 50)   
    state = np.reshape(state, [1, env.nS])
    print(ob)
    target_f = model.predict(state)
    print(target_f)
    print(np.argmax(target_f))