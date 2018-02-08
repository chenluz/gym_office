import gym
import numpy as np
import json
import requests
import time
import itertools
import pandas as pd
import os
from .simulator import airVelocity
from .simulator import skinTemperature
from .simulator import feedback
from sklearn.preprocessing import MinMaxScaler


Clo_initial = 1.2
Rh_initial = 20
Ta_out_initial = -1
Rh_out_initial = 60
Action_Dict = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3:[0, 1, 1], 
				4: [1, 0, 0], 5: [1, 0, 1], 6: [1, 1, 0], 7: [1, 1, 1]}


# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeSimulateEnv(gym.Env):

	def __init__(self):
		self.nS = 6
		self.nA = 8
		self.is_terminal = False
		self.step_count = 0
		self.cur_Ta = 0
	


	def get_state(self, Ta, Tr, Rh):
		Tskin_obj = skinTemperature()
		Tskin = Tskin_obj.comfPierceSET(Ta, Tr, Rh, Clo_initial)
		# get PMV and PPD based on environmental parameter
		feedback_obj = feedback()
		[pmv, ppd] = feedback_obj.comfPMV(Ta, Tr, Rh, Clo_initial)

		ob = self.process_state([Ta, Rh, Tskin, pmv,Ta_out_initial, Rh_out_initial])
		return [Ta, Rh, Tskin, pmv,Ta_out_initial, Rh_out_initial], ob


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
		# # get fan speed and temperature set point from action 
		# fan = Action_Dict[action][1]
		# T_setpoint = Action_Dict[action][0]
		# # get air velocity after action
		# action_obj = airVelocity()
		# Vel = action_obj.get_air_velocity(action)

		# get air temperature after action

		# if first heater on, increase 2
		if Action_Dict[action][0] == 1:
			Ta = self.cur_Ta + 0.5
		else:
			T_increase = sum(Action_Dict[action])*0.25
			Ta = self.cur_Ta + T_increase

		self.cur_Ta = Ta

		#get other envrionmental parameter:
		# air temperature, mean radiant temperature, humidity, cloth 
		Tr = Ta - 2
		Rh = Rh_initial
		Clo = Clo_initial 
		# get skin temperature based on envrionmental parameter and individual parameter 
		Tskin_obj = skinTemperature()
		Tskin = Tskin_obj.comfPierceSET(Ta, Tr, Rh, Clo)
		# get PMV and PPD based on environmental parameter
		feedback_obj = feedback()
		[pmv, ppd] = feedback_obj.comfPMV(Ta, Tr, Rh, Clo)
	
		ob = self.process_state([Ta, Rh,Tskin, pmv, Ta_out_initial,Rh_out_initial])
		
		reward = -ppd/100

		self.step_count += 1
		if self.step_count > 100:
			self.is_terminal = True
			self.step_count = 0
		return ob, reward, self.is_terminal, {}



	def _reset(self):
		self.is_terminal = False
		# get initial velocity randomly
		Ta_initial = np.random.uniform(18,30)
		self.cur_Ta = Ta_initial
		Tr_initial = Ta_initial - 2
		Rh_initial = np.random.choice(np.arange(20,80))

		action = np.random.choice(self.nA)
		#action_obj = airVelocity()
		#Vel_initial = action_obj.get_air_velocity(action)
		Tskin_obj = skinTemperature()
		Tskin_initial = Tskin_obj.comfPierceSET(Ta_initial, Tr_initial, 
			Rh_initial, Clo_initial)
		# get PMV and PPD based on environmental parameter
		feedback_obj = feedback()
		[pmv_initial, ppd_initial] = feedback_obj.comfPMV(Ta_initial, Tr_initial, 
			Rh_initial, Clo_initial)
		pmv_initial= pmv_initial

		ob = self.process_state([Ta_initial,Rh_initial, Tskin_initial,
		 pmv_initial, Ta_out_initial,Rh_out_initial])

		return ob


	def process_state(self, state):
		# process state 
		state[0] = (state[0] - 18)*1.0/(30 - 18) # air temperature
		state[1] = (state[2] - 20)/(80 - 20) # relative humidity
		state[2] = (state[3] - 29)/(36 - 29) # skin temperature
		state[3] = (state[4] + 3.86)/(3.86 + 1.38) # pmv
		# assume outdoor does not change for now
		state[4] = 0 # outdoor air temperture
		state[5] = 0 # outdoor relative humidity
		return state


	def _render(self, mode='human', close=False):
		pass


# Action_Dict = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3:[0, 1 , 0], 
# 				4: [0, 1, 1], 5: [0, 1, 2], 6: [0, 2, 0], 7: [0, 2, 1], 8: [0, 2, 2],
# 				9: [1, 0, 0], 10: [1, 0, 1], 11: [1, 0, 2], 12: [1, 1, 0], 13: [1, 1, 1], 
# 				14: [1, 1, 2], 15: [1, 2, 0], 16: [1,2, 1], 17: [1, 2, 2], 18: [2, 0, 0],
# 				19: [2, 0, 1], 20:[2, 0, 2], 21:[2, 1, 0], 22:[2, 1, 1], 23:[2, 1, 2], 
# 				24:[2, 2, 0],25:[2, 2, 1],26:[2, 2, 2]}
				
# Action_Dict = {0: [21, 0], 1: [21, 1], 2: [21, 2], 3:[21, 3], 
# 				4: [22, 0], 5: [22, 1], 6: [22, 2], 7: [22, 3], 8: [22, 4],
# 				 9: [23, 0], 10: [23, 1], 11: [23, 2], 12: [23, 3], 13: [23, 4], 
# 				14: [24, 0], 15: [24, 1], 16: [24, 2], 17: [24, 3], 18: [24, 4], 19: [24, 5],
# 				20: [25, 0], 21: [25, 1], 22: [25, 2], 23: [25, 3], 24: [25, 4], 25: [25, 5],
# 				26: [26, 1], 27: [26, 2], 28: [26, 3], 29: [26, 4], 30: [26, 5], 31: [26, 6],
# 				32: [27, 1], 33: [27, 2], 34: [27, 3], 35: [27, 4], 36: [27, 5], 37: [27, 6], 38: [27, 7],
# 				39: [28, 2], 40: [28, 3], 41: [28, 4], 42: [28, 5], 43: [28, 6], 44: [28, 7]}