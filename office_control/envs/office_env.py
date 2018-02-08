import gym
from .action import openHab
from .action import plugWise
from .observation import PIServer
from .observation import InfluxDB
import numpy as np
import json
import requests
import time
import itertools
import csv


# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeEnv(gym.Env):

	def __init__(self, state_type="physical", step_in_episode = 60, response_time = 300):
		self.plugwise = plugWise("128.2.108.76", 8080)
		self.db = InfluxDB(host='localhost', port=8086, username='chenlu',
            password='research', database='CMUMM409office')
		self.nA = len(Action_Dict)
		self.nS = 6
		self.step_counter = 20
		self.action = 0 
		self.reward = 0
		self.observation = []
		self.response_time = response_time


	def _step(self, action):
		"""after take action, waiting for the response_time to make sure 
		 the action has effect on the building and occupant"""
		self.action = action
		print("step: " + str(self.step_counter))
		self._take_local_action(action)
		interval = 1
		for i in range(self.response_time,0,-1):
			time.sleep(interval)
			print("next action time:" + str(i*interval) + "second later")
		self.observation = self.db.get_observation('10m', '3m')
		print(self.observation)
		state,reward = self._process_observation(self.observation)
		self.reward = reward
		self.step_counter = self.step_counter - 1
		if self.step_counter < 1:
			is_terminal = True
		else:
			is_terminal = False
		return state, reward, is_terminal, {}


	def _take_central_action(self,acition):
		pass
		


	def _take_local_action(self, action):
		""" Converts the action space into an real actuation to devices. 
		Parameters
		----------
		action: int value from 0 to self.nA-1
		"""
		a = 0
		print("action:" + str(action)) 
		[heater_1, heater_2, heater_3] = Action_Dict[action] 
		print (DEVICE_NAME[DEVICE[0]], ACTION_Local[heater_1])
		self.plugwise.send_command(DEVICE[0], ACTION_Local[heater_1]) 
		print (DEVICE_NAME[DEVICE[1]], ACTION_Local[heater_2])
		self.plugwise.send_command(DEVICE[1], ACTION_Local[heater_2]) 
		print (DEVICE_NAME[DEVICE[2]], ACTION_Local[heater_3])
		self.plugwise.send_command(DEVICE[2], ACTION_Local[heater_3]) 


	def _process_observation(self, observation):
		""""
		 process observation:
		(observation - low)/(high - low)

		return
		-------
		state: array
		reward: float

		"""
		pro_obser = [None]*len(observation)
		for i in range(len(observation)):
			if len(low_high[Obser_Order[i]]) != 0:
				pro_obser[i] = (observation[Obser_Order[i]] - 
						low_high[Obser_Order[i]][0])*1.0/(low_high[Obser_Order[i]][1] 
						- low_high[Obser_Order[i]][0]) 
			else:
				pro_obser[i] = observation[Obser_Order[i]]

		reward = -(abs(observation[Obser_Order[-2]])/3)

		return pro_obser, reward


	def _reset(self):
		self.observation = self.db.get_observation('10m', '3m')
		print(self.observation)
		state, reward = self._process_observation(self.observation)
		print(state)
		# set all the device to off
		for i in DEVICE:
			self.plugwise.send_command(DEVICE[i], ACTION_Local[0])
		time.sleep(5)
		return state

	def _render(self, mode='human', close=False):
		pass

	def my_render(self, model='human', close=False):
	    with open("render.csv", 'a', newline='') as csvfile:
	        fieldnames = ['skin_temp_mean', 'skin_temp_deriv', 'air_temp_mean',
	                    'air_temp_deriv', 'air_humi_mean', 
	                    'thermal_sensation', 'action', 'reward']
	        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	        writer.writerow({fieldnames[0]: self.observation[fieldnames[0]], fieldnames[1]: 
	        	self.observation[fieldnames[1]], fieldnames[2]:self.observation[fieldnames[2]],
	                fieldnames[3]:self.observation[fieldnames[3]], fieldnames[4]:self.observation[fieldnames[4]], 
	                fieldnames[5]:self.observation[fieldnames[5]],
	                fieldnames[6]:self.action, fieldnames[7]:self.reward})


DEVICE = {
	0 : 548, # plugwise name of Fan Heater 1 D359E6
    1 : 462, # plugwise name of Fan Heater 2 D3688D
    2 : 467, # plugwise name of Fan Heater 3 D36928
}

DEVICE_NAME = {
    548 : "Fan Heater Right",
    462 : "Fan Heater Left ",
    467 : "Fan Heater Front"
}

ACTION_Local = {
    0 : "switchoff",
    1 : "switchon",
}

Action_Dict = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3:[0, 1, 1], 
				4: [1, 0, 0], 5: [1, 0, 1], 6: [1, 1, 0], 7: [1, 1, 1]}

Obser_Order = ['skin_temp_mean', 'skin_temp_deriv', 'air_temp_mean',
				'air_temp_deriv', 'air_humi_mean', 
				'thermal_sensation', 'thermal_satisfaction']

low_high = {'skin_temp_mean':[25, 35], 'skin_temp_deriv':[], 'air_temp_mean':[18, 30],
				'air_temp_deriv':[], 'air_humi_mean':[5, 70], 
				'thermal_sensation':[-3, 3], 'thermal_satisfaction':[-3, 3]}


