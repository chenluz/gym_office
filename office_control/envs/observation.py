import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import pytz, datetime
import time
from influxdb import InfluxDBClient
import numpy as np


class PIServer():
    """
    Data from PI server

    Parameters
    ----------
    host: str
    port: int
    username: str
    password: str
    """

    def __init__(self, host, username, password):  
        self.piserver_host = host
        self.headers = {'Content-type': 'application/json'} 
        self.auth = HTTPBasicAuth(username, password)
    

    def get_command(self, key):
        """ get data from piserver - 
        Parameters
        ----------
        key: str, piserver 
        """

        url = 'https://%s/piwebapi/streams/%s/value'%(self.piserver_host, key)
        payload = {'type': 'json'}
        req = json.loads(requests.get(url, headers=self.headers, auth=self.auth).text)
        return req["Value"], str(req["Timestamp"]) 


class InfluxDB():
    """
    Data from InfluxDB

    Parameters
    ----------
    host: str
    port: int
    username: str
    password: str
    """

    def __init__(self, host, port, username, password, database):  
        self.client = InfluxDBClient(host = host, port = port, username = username,
         password = password, database = database)


    ############### begin of help function #################
    def get_topValue_db(self, column, table):
        _query = 'select top('+ column +', 1) from ' + table + ';'
        #print(_query)
        data_db = self.client.query(_query)
        top_value = list(data_db.get_points())[0]["top"]
        return top_value

    def get_mean_pastTime_db(self, column, table, past_time):
        ## the mean funciton in influxdb is : 
        ##(1)if current time is 1:20, the latest mean over 3 mininutes will be 1:18
        ##(2) and the mean value for 1:18 is the averaged from [1:18, 1:21)
        ##So here we don't use mean in influxdb
        _query = 'select '+ column +' from ' + table +' where time > now() - '+ past_time + ';'
        #print(_query)
        data_db = self.client.query(_query)
        value_list = []
        for value in list(data_db.get_points()):
            #print(value)
            value_list.append(value[column])
        return np.mean(np.array(value_list))
        

    def get_derivative_db(self, column, table, interval):
        _query = 'select derivative(mean('+ column +')) from '+ table +' where time > now() - '+ interval + ' group by time('+ interval +') ;'
        #print(_query)
        data_db = self.client.query(_query)
        value_re = 0
        if len(list((data_db.get_points()))) == 0:
            value_re = 0
        else: 
            value_re = list(data_db.get_points())[-1]['derivative']
        #print(value_re)
        return value_re

    def get_vote_db(self, column, table, past_time):
        ## if there is no voting in past time, set voting as 0 
        _query = 'select '+ column +' from '+ table +' where time > now() - '+ past_time + ';'
        #print(_query)
        data_db = self.client.query(_query)
        value_list = []
        value_re = 0
        for value in list(data_db.get_points()):
            #print(value)
            value_list.append(value["value"])
        # There is no voting in last past_time minutes, set satisfaction as 0
        if len(value_list) == 0:
            value_re = 0
        else:
            value_re = value_list[-1]
        return value_re
    ############### End of help function #################


    def get_observation(self, interval, past_time):
        """ get data from InfluxDB - 

        Parameters
        ----------
        interval: str  time between two actions
        past_time: str only select data from relative past_time

        Return
        ----------
        A dictionary of observation including:
        skin_temp_mean, skin_temp_deriv, air_temp_mean, air_temp_deriv,
        air_humi_mean, heart_rate_mean, thermal_sat}

        The structure of different table in the influx database:
        Measurement(Table)       field_key1       field_key2   
        skin_temperature         value
        heart_rate               value
        thermal_satisfaction     value
        thermal_sensation        value
        environment              temperature      humidity 


        Data: Timestamp field_value

        """

        ###feature selection#####
        #1. mean of past 'past_time' data
        # if there is no value in 'past_time' in db, get the latest value in db
        obs_dict = {}
        for key in occupant:
            value = self.get_mean_pastTime_db('value', key, past_time) 
            if np.isnan(value):
                obs_dict[occupant[key]] = self.get_topValue_db('value', key)
            else:
                obs_dict[occupant[key]] = value
        for key in environment:
            value = self.get_mean_pastTime_db(key, 'environment', past_time) 
            if np.isnan(value):
                obs_dict[environment[key]] = self.get_topValue_db(key, 'environment')
            else:
                obs_dict[environment[key]] = value  

        #2. derivative value of past 'past_time' data
        obs_dict[derivative['temperature']] = self.get_derivative_db('temperature', 'environment', interval)
        obs_dict[derivative['skin_temperature']] = self.get_derivative_db('value', 'skin_temperature', interval)

        #3. Get lastest voting in last past_time minutes
        #obs_dict["thermal_satisfaction"] = self.get_vote_db('value', 'thermal_satisfaction', past_time)
        obs_dict["thermal_sensation"] = self.get_vote_db('value', 'thermal_sensation', past_time)

        return obs_dict

        def save_action_db(self, action):
            json_body = [
                {
                    "measurement": "action",
                     "tags": {
                        "name": heater,
                    },
                    "time": utcnow,
                    "fields": {
                        "value": action,
                    }
                }
            ]

            self.client.write_points(json_body)



occupant = {"skin_temperature": 'skin_temp_mean'}
             #"heart_rate" : 'heart_rate_mean'}
environment = {"temperature": 'air_temp_mean', 
                "humidity": 'air_humi_mean'}
derivative = {"skin_temperature": "skin_temp_deriv",
              "temperature": "air_temp_deriv"}




