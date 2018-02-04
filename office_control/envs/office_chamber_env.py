import gym
from .action import openHab
from .action import plugWise
from .observation import PIServer
import numpy as np
import json
import requests
import time
import itertools


# ref: https://github.com/openai/gym/tree/master/gym/envs

class OfficeEnv(gym.Env):

	def __init__(self, state_type="physical", step_in_episode = 60, response_time = 18):
		self.openhab = openHab()
		self.plugwise = plugWise()
		self.piserver = PIServer()
		self.nA = len(ACTION)**len(DEVICE)
		self.nR = 7
		self.begining = True
		self.pre_timestamp = ""
		self.state_type = state_type
		self.step_counter = 0
		self.step_in_episode = step_in_episode
		self.response_time = response_time


	def _step(self, action):
		"""after take action, waiting for the response_time to make sure 
		 the action has effect on the building and occupant"""
		self._take_action(action)
		for i in range(self.response_time,0,-1):
		    time.sleep(10)
		    print("next action time:" + str(i*10) + "second later")
		if self.state_type == "subjective":
			ob = self._get_subjective_state()
		elif self.state_type == "physical":
			ob = self._get_physical_state()
		reward = self._get_reward()
		is_terminal = self._is_episode_over_timer()
		return ob, reward, is_terminal, {}


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

		if action < 3:
			# choose one device on ,other off.
			for i in range(0, len(DEVICE)):
				if a == action:
					# one device on  
					print (DEVICE_NAME[DEVICE[i]], ACTION[1])
					self.plugwise.send_command(DEVICE[i], ACTION[1]) 
				else:
					# other device off
					print (DEVICE_NAME[DEVICE[i]], ACTION[0])
					self.plugwise.send_command(DEVICE[i], ACTION[0]) 
				a += 1
		elif action < 6:
			a = a + 3
			# choose one device off, other on
			for i in range(0, len(DEVICE)):
				if a == action:
					# one device off  
					print (DEVICE_NAME[DEVICE[i]], ACTION[0])
					self.plugwise.send_command(DEVICE[i], ACTION[0]) 
				else:
					# other device off
					print (DEVICE_NAME[DEVICE[i]], ACTION[1])
					self.plugwise.send_command(DEVICE[i], ACTION[1]) 
				a += 1
		# three on or three off  
		elif action == 6:
			for i in range(0, len(DEVICE)):
				self.plugwise.send_command(DEVICE[i], ACTION[1])
		else: #7
			for i in range(0, len(DEVICE)):
				self.plugwise.send_command(DEVICE[i], ACTION[0]) 

					

	def _get_physical_state(self):
		""" state is occupant's skin temperature and heart rate. """
		hrate = self.piserver.get_command(OBSERVATION['hrate'])[0]
		rr = self.piserver.get_command(OBSERVATION['rr'])[0]
		skin_temp = self.piserver.get_command(OBSERVATION['temp'])[0]	
		return  np.array([skin_temp, hrate, rr])


	def _get_subjective_state(self):
		""" state is occupant's senstaion """
		senstaion = int(self.piserver.get_command(OBSERVATION['sensation'])[0])
		return senstaion


	def _get_reward(self):
		""" Reward is given by the sensation """
		sensation = int(self.piserver.get_command(OBSERVATION['sensation'])[0])
		reward =  self._process_reward(sensation)
		return reward



	def _process_reward(self, sensation):
	    """ convert the sensation to a meaningful reward """
	    if sensation > 0:
	    	return -sensation
	    return sensation



	def _is_episode_over(self):
		""" episode is over when user give new sensation feedback,
		which means the timestamp for the sensation in the databased is changed """
		timestamp = self.piserver.get_command(OBSERVATION['sensation'])[1]
		if self.begining:
			self.pre_timestamp = timestamp
			self.begining = False

		if self.pre_timestamp  == timestamp:
			return False 
		return True


	def _is_episode_over_timer(self):
		self.step_counter += 1
		if self.step_counter > self.step_in_episode:
			return True
		return False


	def _reset(self):
		if self.state_type == "subjective":
			ob = 0
		elif self.state_type == "physical":
			ob = self._get_physical_state()
		self.step_counter = 0
		# set all the device to off
		for i in range(0, len(DEVICE)):
			self.plugwise.send_command(DEVICE[i], ACTION[0])
		time.sleep(5)
		return ob

	def _render(self, mode='human', close=False):
		pass


DEVICE = {
    0 : 499, # plugwise name of Fan Heater 75A67F
    1 : 462, # plugwise name of Fan D3688D
    2 : 548, # plugwise name of Air Condiitoner D359E6
    3 : 464, # plugwise name of D35E98
}

DEVICE_NAME = {
	499 : "Fan Heater",
	462 : "Fan",
	548 : "Air Conditioner"
}

ACTION_Local = {
    0 : "switchoff",
    1 : "switchon",
}

OBSERVATION = {
	"hrate": "P0-MYhSMORGkyGTe9bdohw0ATvECAAV0lOLTYyTlBVMkJWTDIwXElXL0NIRU5MVS9DSEVOTFUvSEVBUlRfUkFURQ", 
    "temp":"P0-MYhSMORGkyGTe9bdohw0AWywCAAV0lOLTYyTlBVMkJWTDIwXElXL0NIRU5MVS9DSEVOTFUvU0tJTl9URU1QRVJBVFVSRQ", 
    "rr":"P0-MYhSMORGkyGTe9bdohw0AVPECAAV0lOLTYyTlBVMkJWTDIwXElXL0NIRU5MVS9DSEVOTFUvUlJfUkFURQ", 
    "sensation": "P0-MYhSMORGkyGTe9bdohw0AUfECAAV0lOLTYyTlBVMkJWTDIwXElXL0NIRU5MVS9DSEVOTFUvVEhFUk1BTF9QUkVGRVJFTkNF"
}

ACTION_Central = {
	"setpoint": "P0-MYhSMORGkyGTe9bdohw0AVzMCAAV0lOLTYyTlBVMkJWTDIwXElXX05PREUwMl9JVy5DT05UUk9MLlNFVFBPSU5U/value", 
}