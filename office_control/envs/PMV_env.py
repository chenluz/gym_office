import gym
import numpy as np
import json
import requests
import time
import itertools
import pandas as pd
import os

from .simulator import feedback

from sklearn.preprocessing import MinMaxScaler
import csv
import datetime


Ta_min = 17
Ta_max = 29
Rh_min = 35
Rh_max = 36



# ref: https://github.com/openai/gym/tree/master/gym/envs

class PMVEnv(gym.Env):

	def __init__(self):
		self.nS = 2
		self.nA = 3
		self.is_terminal = False
		self.step_count = 0
		self.cur_Ta = 0
		self.cur_Rh = 0
		self.action = 0
		self.reward = 0
		self.vote = feedback()
			

	def _step(self, action):
		""" take action and return next state and reward
		Parameters
		----------
		action: int value from 0 to self.nA-1, it is fan speed setting

		Return 
		----------
		ob:  array of state
		reward: float , PMV value

		"""
		# get air temperature and air humidity after action
		incr_Ta = action - 1 
		self.action = action
		pre_Ta = self.cur_Ta
		pre_Rh = self.cur_Rh
		self.cur_Ta  = self.cur_Ta + incr_Ta
		self.cur_Rh  = np.random.choice(np.arange(Rh_min, Rh_max, 1))


		if self.cur_Ta > Ta_max:
			self.is_terminal = True
		elif self.cur_Ta < Ta_min:
			self.is_terminal = True

		# get PPD after action
		self.reward = -1*self.vote.comfPMV(self.cur_Ta, self.cur_Ta, self.cur_Rh, 1.0)[1]

		state = self.process_state([self.cur_Ta, self.cur_Rh])
	
		return state, self.reward, self.is_terminal, {}


	def _reset(self):
		self.is_terminal = False
		self.cur_Ta = np.random.choice(np.arange(Ta_min, Ta_max, 1))
		self.cur_Rh  = np.random.choice(np.arange(Rh_min, Rh_max, 1))
		state = self.process_state([self.cur_Ta, self.cur_Rh])
		return state

	def _print(self):
		print(self.cur_Ta, self.cur_Rh)

	def process_state(self, state):
		# process state 
		state[0] = (state[0] - Ta_min)*1.0/(Ta_max - Ta_min) # air temperature
		state[1] = (state[1] - Rh_min)*1.0/(Rh_max - Rh_min) # relative humidity
		return state


	def _render(self, mode='human', close=False):
		pass

	def my_render(self, model='human', close=False):
	    with open("render_PMV_5action.csv", 'a', newline='') as csvfile:
	        fieldnames = ['time', 'action', 'air_temp','air_humid', 'reward']
	        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	        writer.writerow({fieldnames[0]: datetime.datetime.utcnow(), 
	        	fieldnames[1]:self.action, 
				fieldnames[2]:self.cur_Ta, 
				fieldnames[3]:self.cur_Rh, 
				fieldnames[4]:self.reward})


