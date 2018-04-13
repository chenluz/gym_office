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
from .simulator import outdoorSet
from sklearn.preprocessing import MinMaxScaler
import csv
import datetime


Ts_min = 25
Ts_max = 37
Ta_min = 16.5
Ta_max = 32
Rh_min = 12
Rh_max = 44
Rh_out_min = 24
Rh_out_max = 85
Ta_out_min = 5.77
Ta_out_max = 22

# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeSimulateEnv(gym.Env):

	def __init__(self):
		self.nS = 5
		self.nA = 4
		self.is_terminal = False
		self.step_count = 0
		self.cur_Ta = 0
		self.cur_Rh = 0
		self.cur_Tskin = 0
		self.out_Ta = 0
		self.out_Rh = 0
		self.action = 0
		self.reward = 0
		self.outdoor = outdoorSet()
		self.air = airEnviroment()
		self.skin = skinTemperature()
		self.vote = feedback()
			

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
		self.step_count += 1
		# get outdoor temperature and humidity 
		self.out_Ta, self.out_Rh = self.outdoor.get_out(self.step_count)

		# get air temperature and air humidity after action
		self.action = action
		pre_Ta = self.cur_Ta
		pre_Rh = self.cur_Rh
		pre_Tskin = self.cur_Tskin
		self.cur_Ta  = self.air.get_air_temp(action, pre_Ta, self.out_Ta)
		self.cur_Rh  = self.air.get_air_humidity(self.cur_Rh, self.out_Rh)
		# get old skin temperature after action
		self.cur_Tskin = self.skin.skin_SVR(self.cur_Ta, self.cur_Rh)

		if self.cur_Ta > Ta_max:
			self.is_terminal = True
			self.reward = -3
		elif self.cur_Ta < Ta_min:
			self.is_terminal = True
			self.reward = -3
		else:
			# get thermal satisfaction after action
			self.reward = self.vote.Satisfaction_neural(self.cur_Tskin, self.cur_Ta, self.cur_Rh)

		state = self.process_state([self.cur_Tskin, self.cur_Ta, self.cur_Rh, self.out_Ta, self.out_Rh])
		self.reward = (self.reward + 3)/6

		if self.step_count > 82:
			self.is_terminal = True
			self.step_count = 0
		return state, self.reward, self.is_terminal, {}


	def _reset(self):
		self.is_terminal = False
		self.out_Ta, self.out_Rh = self.outdoor.get_out(self.step_count)
		self.cur_Ta = np.random.choice(np.arange(Ta_min, Ta_max, 1))
		self.cur_Rh  = self.air.get_air_humidity(self.cur_Rh, self.out_Rh)
		self.cur_Tskin = self.skin.skin_SVR(self.cur_Ta, self.cur_Rh)
		#print(self.cur_Tskin, self.cur_Ta, self.cur_Rh,self.out_Ta, self.out_Rh)
		state = self.process_state([self.cur_Tskin, self.cur_Ta, self.cur_Rh,self.out_Ta, self.out_Rh])
		return state


	def process_state(self, state):
		# process state 
		state[0] = (state[0] - Ts_min)*1.0/(Ts_max - Ts_min) # skin temperature
		state[1] = (state[1] - Ta_min)*1.0/(Ta_max - Ta_min) # air temperature
		state[2] = (state[2] - Rh_min)*1.0/(Rh_max - Rh_min) # relative humidity
		state[3] = (state[3] - Ta_out_min)*1.0/(Ta_out_max - Ta_out_min) # outdoor air temperature
		state[4] = (state[4] - Rh_out_min)*1.0/(Rh_out_max - Rh_out_min) # outdoor relative humidity
		return state


	def _render(self, mode='human', close=False):
		pass

	def my_render(self, model='human', close=False):
	    with open("render_simulator_4action.csv", 'a', newline='') as csvfile:
	        fieldnames = ['time', 'action', 'skin_temp', 'air_temp','air_humid', 'out_temp', 'out_humid', 'reward']
	        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	        writer.writerow({fieldnames[0]: datetime.datetime.utcnow(), 
	        	fieldnames[1]:self.action, 
				fieldnames[2]:self.cur_Tskin, 
				fieldnames[3]:self.cur_Ta, 
				fieldnames[4]:self.cur_Rh, 
				fieldnames[5]:self.out_Ta, 
				fieldnames[6]:self.out_Rh,
				fieldnames[7]:self.reward})


