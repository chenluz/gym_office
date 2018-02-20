#!/usr/bin/env python
# coding: utf-8
import office_control.envs as office_env
import gym
import time
import numpy as np
from influxdb import InfluxDBClient
import requests
import json
import jwt
import time
import datetime
from hyper import HTTPConnection


def calibrate_sensor():
    client = InfluxDBClient(host='localhost', port=8086, username='chenlu',
            password='research', database='CMUMM409office')
    _query = 'select * from environment;'

    while True: 
        rs = client.query(_query)
        print(list(rs.get_points(tags={'name':'user1'}))[-1])
        print(list(rs.get_points(tags={'name':'user2'}))[-1])
        print(list(rs.get_points(tags={'name':'user3'}))[-1])
        print(list(rs.get_points(tags={'name':'user4'}))[-1])
        print(list(rs.get_points(tags={'name':'user5'}))[-1])
        print(list(rs.get_points(tags={'name':'chenlu'}))[-1])
        print("   ")
        time.sleep(10)


def run_controller_policy(initial_temp):
    """Run a policy to stablize temperature for the given environment.

    """
    client = InfluxDBClient(host='localhost', port=8086, username='chenlu',
            password='research', database='CMUMM409office')
    _query = 'select * from environment;'
    highest_temp = 30
    lowest_temp = 18
    rise_count = highest_temp - initial_temp 
    curr_temp = initial_temp + 1
    stable_counter = 0
    rise_step = 0
    title1 = "Thermal Sensation:"
    title2 = "Thermal Satisfaction:"
    message = "Tell me your current feeling!" 
    while True:
        rs = client.query(_query)
        value_list = []
        value_re = 0
        value1 = list(rs.get_points(tags={'name':'user5'}))[-1]['temperature']
        value2 = list(rs.get_points(tags={'name':'user5'}))[-2]['temperature']
        value3 = list(rs.get_points(tags={'name':'user5'}))[-3]['temperature']
        print(value1, value2, value3)
        print(curr_temp, stable_counter)
        big = (value1 >= curr_temp) & (value2 >= curr_temp) & (value3 >= curr_temp)
        small =  (value1 <= curr_temp - 0.1) & (value2 <= curr_temp - 0.1) & (value3 <= curr_temp - 0.1) 
        if(big):
            print("off")
            #take_local_action(plugwise, 0)
        if(small):
            #take_local_action(plugwise, 7)
            print("on")

        withBig = (value1 <= curr_temp + 0.2) & (value2 <= curr_temp + 0.2) & (value3 <= curr_temp + 0.2)
        withSmall =  (value1 >= curr_temp - 0.2) & (value2 >= curr_temp - 0.2) & (value3 >= curr_temp - 0.2)
        if withBig & withSmall: 
            stable_counter += 1
        else: 
            stable_counter = 0
        # if stable_counter == 10:
        #     send_notification(title1, message)
        # after 5 minutes stable
        if stable_counter == 10:
            if rise_step < rise_count:
                curr_temp += 1
                rise_step += 1
            else: 
                curr_temp -=1 
            stable_counter =0
        time.sleep(30)
    return 


def take_local_action(plugwise, action):
        """ Converts the action space into an real actuation to devices. 
        Parameters
        ----------
        action: int value from 0 to self.nA-1
        """
        [heater_1, heater_2, heater_3] = Action_Dict[action] 
        print (DEVICE_NAME[DEVICE[0][0]], ACTION_Local[heater_1])
        plugwise.send_command(DEVICE[0][0], ACTION_Local[heater_1]) 
        print (DEVICE_NAME[DEVICE[0][1]], ACTION_Local[heater_1])
        plugwise.send_command(DEVICE[0][1], ACTION_Local[heater_1]) 
        print (DEVICE_NAME[DEVICE[1]], ACTION_Local[heater_2])
        plugwise.send_command(DEVICE[1], ACTION_Local[heater_2]) 
        print (DEVICE_NAME[DEVICE[2]], ACTION_Local[heater_3])
        plugwise.send_command(DEVICE[2], ACTION_Local[heater_3]) 


class plugWise():
    # http://128.2.108.76:8080/api/data.xml?type=appliances  ##check id and name mapping
    """
    Actuation through plugWise

    Parameters
    ----------
    host: str
    port: int

    """

    def __init__(self, host, port):  
        self.plugwise_host = host
        self.plugwise_port = port
    

    def send_command(self, key, value):
        """ send a command/action to plugwise - 
        Parameters
        ----------
        key: int, plugwiseid 
        value: str, action send to plugwise, e.g.,"switchoff" """

        url = 'http://%s:%s/api/actions.html?option=%s&id=%s'%(self.plugwise_host,
                                self.plugwise_port, value, key)
        payload = {'type': 'json'}
        req = requests.get(url, params=payload)

        if req.status_code != requests.codes.ok:
            req.raise_for_status()



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


def send_notification(title, message):
    ALGORITHM = 'ES256'

    APNS_KEY_ID = '27LKLT6JA6'
    APNS_AUTH_KEY = 'AuthKey_27LKLT6JA6.p8'
    TEAM_ID = 'N87XX7JNU3'
    BUNDLE_ID = 'edu.cmu.iw.chenli.comfort'

    #REGISTRATION_ID = '48e6081edca89c945ea15abe9669da1ebad165e2ac348bfb478e143154685012'
    REGISTRATION_ID = '1dba1ad3032432fc99db9c49a23a2ce206fed105aa40a8ae99b6480adad8557d'

    f = open(APNS_AUTH_KEY)
    secret = f.read()

    key_time = time.time()
    print(
        datetime.datetime.fromtimestamp(
            key_time
        ).strftime('%Y-%m-%d %H:%M:%S')
    )

    token = jwt.encode(
        {
            'iss': TEAM_ID,
            'iat': key_time 
        },
        secret,
        algorithm= ALGORITHM,
        headers={
            'alg': ALGORITHM,
            'kid': APNS_KEY_ID,
        }
    )

    path = '/3/device/{0}'.format(REGISTRATION_ID)

    request_headers = {
        'apns-expiration': '0',
        'apns-priority': '10',
        'apns-topic': BUNDLE_ID,
        'authorization': 'bearer {0}'.format(token.decode('ascii'))
    }


    # Open a connection the APNS server
    conn = HTTPConnection('api.development.push.apple.com:443')

    payload_data = { 
        "aps": {
            "badge": 0,
            "content-available": 1,
        },
        "custom" : {
            "title": title,
            'message': message
        }
    }
    payload = json.dumps(payload_data).encode('utf-8')

    # Send our request
    conn.request(
        'POST', 
        path, 
        payload, 
        headers=request_headers
    )
    resp = conn.get_response()
    print(resp.status)
    return


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nR, env.nA))


DEVICE = {
    0 : [464, 507], # plugwise name of Fan Heater 1, 4 D35E98, 75AA21
    1 : 467, # plugwise name of Fan Heater 2, 3 D36928 
    2 : 548, # plugwise name of Fan Heater 5, 6, D359E6 
    4 : 462, # plugwise name of Air conditioner D3688D 
}

DEVICE_NAME = {
    464 : "Fan Heater Front Left",
    507 : "Fan Heater Front Right",
    467 : "Fan Heater Left",
    548 : "Fan Heater Right ",
    462 : "Air Conditioner ",
    
}

ACTION_Local = {
    0 : "switchoff",
    1 : "switchon",
}

Action_Dict = {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3:[1, 0, 0], 
                4:[0, 1, 1], 5: [1, 0, 1], 6: [1, 1, 0], 7: [1, 1, 1]}



plugwise = plugWise("128.2.108.76", 8080)
run_controller_policy(21)


