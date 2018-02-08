import gym
import numpy as np
import json
import requests
import time
import itertools
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler




# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeSampleEnv(gym.Env):

	def __init__(self, state_type="physical", csv_file='office_control\envs\csv\environment_sample-skin-skin-human.csv'):
		self.state_type = state_type
		self.sample_env = self._load_sample(csv_file)
		scaler = MinMaxScaler()
		self.sample_env[['state-skin1', 'state-skin2','next_state-skin1', 'next_state-skin2']] = scaler.fit_transform(
			self.sample_env[['state-skin1', 'state-skin2','next_state-skin1', 'next_state-skin2']])
		self.nS = 2
		self.nA = 3
		self.cur_state = [0,0]
		self.next_state = [0,0] 
		self.is_terminal = False
		self.step_count = 0
	

	def _load_sample(self, csv_file):
		df = pd.read_csv(csv_file)
		return df


	def _step(self, action):
		"""after take action, waiting for the response_time to make sure 
		 the action has effect on the building and occupant"""
		#self._take_action(action)
		if self.state_type == "subjective":
			ob = self._get_subjective_state(action)
		elif self.state_type == "physical":
			ob = self._get_physical_state(action)
		reward = self._get_reward(action)
		self.cur_state = self.next_state
		self.step_count += 1
		if self.step_count > 250:
			self.is_terminal = True
			self.step_count = 0
		return ob, reward, self.is_terminal, {}


	def _take_action(self, action):
		""" Converts the action space into an real actuation to devices. 
		Parameters
		----------
		action: int value from 0 to self.nA-1
		"""
		a = 0
		print("action:" + str(action)) 

		if action < 3:
			# choose one device on ,other off.
			for i in range(0, len(DEVICE)):
				if a == action:
					# one device on  
					print (DEVICE_NAME[DEVICE[i]], ACTION[1])
				else:
					# other device off
					print (DEVICE_NAME[DEVICE[i]], ACTION[0])
				a += 1
		elif action < 6:
			a = a + 3
			# choose one device off, other on
			for i in range(0, len(DEVICE)):
				if a == action:
					# one device off  
					print (DEVICE_NAME[DEVICE[i]], ACTION[0])
				else:
					# other device off
					print (DEVICE_NAME[DEVICE[i]], ACTION[1])
				a += 1
		# three on or three off  
		elif action == 6:
			for i in range(0, len(DEVICE)):
				print (DEVICE_NAME[DEVICE[i]], ACTION[1])
		else: #7
			for i in range(0, len(DEVICE)):
				print (DEVICE_NAME[DEVICE[i]], ACTION[0])

					
	def _get_physical_state(self,action):
		""" Get next state based on current state and action """
		possible_next_state = self.sample_env[((self.sample_env["state-skin1"] ==  self.cur_state[0] ) 
			& (self.sample_env["state-skin2"] == self.cur_state[1])
			& (self.sample_env['action'] == action))]
		self.next_state = possible_next_state.loc[possible_next_state.index[0],['next_state-skin1', 'next_state-skin2']].tolist()
		return self.next_state


	def _get_subjective_state(self, action):
		""" Get next state based on current state and action """
		possible_next_state = self.sample_env[((self.sample_env['state'] ==  self.cur_state ) 
			& (self.sample_env['action'] == action))]
		length = len(possible_next_state.index)
		index = np.random.choice(length)
		self.next_state = possible_next_state['next state'].tolist()[index]
		return self.next_state


	def _get_reward(self, action):
		""" Get reward based on current state, action, and next state  """
		possible_reward= self.sample_env[((self.sample_env["state-skin1"] ==  self.cur_state[0] ) 
			& (self.sample_env["state-skin2"] == self.cur_state[1]) 
			& (self.sample_env['action'] == action)
			& (self.sample_env['next_state-skin1'] == self.next_state[0])
			& (self.sample_env['next_state-skin2'] == self.next_state[1]))]
		row_reward = possible_reward[['reward1','reward2']]
		reward = (row_reward['reward1'] + row_reward['reward2'])/2
		return reward


	def _reset(self):
		if self.state_type == "subjective":
			ob = np.random.choice(self.sample_env["state"])
		elif self.state_type == "physical":
			ob_index = np.random.choice(self.sample_env.index)
		ob = self.sample_env.loc[ob_index,["state-skin1", "state-skin2"]].tolist()
		self.cur_state = ob
		self.is_terminal = False
		return ob

	def _render(self, mode='human', close=False):
		pass


DEVICE = {
    0 : 499, # plugwise id of Fan Heater75A67F
    1 : 462, # plugwise id of Fan D3688D
    2 : 548, # plugwise id of Air Conditioner D359E6
}

DEVICE_NAME = {
	499 : "Fan Heater",
	462 : "Fan",
	548 : "Air Conditioner"
}

ACTION = {
    0 : "switchoff",
    1 : "switchon",
}
