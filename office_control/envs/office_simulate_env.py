import gym
import numpy as np
import json
import requests
import time
import itertools
import pandas as pd
import os
from .simulator import airVelocity
from .simulator import airEnviroment
from .simulator import skinTemperature
from .simulator import feedback
from sklearn.preprocessing import MinMaxScaler
import csv
import datetime

Clo_initial = 1.2
Rh_initial = 20
Ta_out_initial = -1
Rh_out_initial = 60


# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeSimulateEnv(gym.Env):

	def __init__(self):
		self.nS = 3
		self.nA = 7
		self.is_terminal = False
		self.step_count = 0
		self.cur_Ta = 0
		self.cur_Rh = 0
		self.cur_Tskin = 0
		self.action = 0
		self.reward = 0
	

	def get_state(self, cur_Ta, cur_Ha):
		Tskin_obj = skinTemperature()
		Tskin = Tskin_obj.skin_SVR(cur_Ta, cur_Ha)

		ob = self.process_state([Tskin, cur_Ta, cur_Ha])
		return [Tskin, cur_Ta, cur_Ha], ob


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
		self.action = action
		pre_Ta = self.cur_Ta
		pre_Rh = self.cur_Rh
		pre_Tskin = self.cur_Tskin
		air = airEnviroment()
		self.cur_Ta  = air.get_air_temp(action, self.cur_Ta).flatten()[0]
		self.cur_Rh  = air.get_air_humidity(action, self.cur_Rh).flatten()[0]

		# get old skin temperature after action
		Tskin_obj = skinTemperature()
		self.cur_Tskin = Tskin_obj.skin_SVR(self.cur_Ta, self.cur_Rh)[0]

		# get thermal satisfaction after action
		feedback_obj = feedback()
		self.reward = feedback_obj.Satisfaction_neural(self.cur_Tskin, self.cur_Ta, self.cur_Rh,
			pre_Tskin, pre_Ta, pre_Rh)

		state = self.process_state([self.cur_Tskin, self.cur_Ta, self.cur_Rh])

		self.step_count += 1
		if self.step_count > 100:
			self.is_terminal = True
			self.step_count = 0
		#return ob, reward, self.is_terminal, {}
		return state, self.reward, self.is_terminal, {}


	def _reset(self):
		self.is_terminal = False
		self.cur_Ta = 21
		self.cur_Rh = 25
		Tskin_obj = skinTemperature()
		self.cur_Tskin = Tskin_obj.skin_SVR(self.cur_Ta, self.cur_Rh)
		state = self.process_state([self.cur_Ta, self.cur_Rh, self.cur_Tskin])
		return state


	def process_state(self, state):
		# process state 
		state[0] = (state[0] - 21)*1.0/(30 - 21) # air temperature
		state[1] = (state[1] - 20)/(80 - 20) # relative humidity
		state[2] = (state[2] - 29)/(36 - 29) # skin temperature
		return state


	def _render(self, mode='human', close=False):
		pass

	def my_render(self, model='human', close=False):
	    with open("render_simulator.csv", 'a', newline='') as csvfile:
	        fieldnames = ['time', 'action', 'skin_temp_mean', 'skin_temp_deriv', 'air_temp_mean',
	                    'air_temp_deriv', 'air_humi_mean', 
	                    'thermal_sensation', 'reward']
	        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	        writer.writerow({fieldnames[0]: datetime.datetime.utcnow(), 
	        	fieldnames[1]:self.action, 
				fieldnames[2]:self.cur_Tskin, 
				fieldnames[3]:self.cur_Ta, 
				fieldnames[4]:self.cur_Rh, 
				fieldnames[5]:self.reward})


