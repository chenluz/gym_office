from flask import Flask, request    
import csv
import pytz, datetime
import requests
from requests.auth import HTTPBasicAuth
import json
from actuation_openhab import openHab
from actuation_openhab import plugWise
import time
import os
import copy
import datetime
import ast
from influxdb import InfluxDBClient
import calendar

data_keyword = ["hrate", "temp", "gsr", "rr", "activity"]

dict_keyword = {"hrate": "heart_rate", "temp": "skin_temperature", "gsr": "galvanic_skin_response", 
"rr": "rr_interval", "activity": "activity_level"}

app = Flask(__name__)

@app.route("/environment", methods=["POST"])

def getData_env():
    writeToCSV_Environment()
    writeToDB_Environment()
    return ""

def writeToCSV_Environment():
    """
    write data getting from environmental sensor to CSV file
    The format for data is
    {username: xxx, datastreams:{time:xxx, temperature:xxx, humidity:xxx}}
    """
    data = request.data
    json_data = json.loads(data)
    username = json_data["username"]
    now = datetime.datetime.now()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    with open('data/environment' + '-' + username + '-' + str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '.csv', 'ab') as csvfile:
        fieldnames = ['time', 'temperature', 'humidity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data =  json_data["datastreams"]
        writer.writerow({'time': st, 'temperature': data['temperature'], 'humidity': data['humidity']})


def writeToDB_Environment():
    """
    write data getting from environmental sensor to influxdb database
    The format for data is
    {username: xxx, datastreams:{time:xxx, temperature:xxx, humidity:xxx}}
    """
    data = request.data
    json_data = json.loads(data)
    username = json_data["username"]
    utcnow = datetime.datetime.utcnow()
    data =  json_data["datastreams"]
                
    json_body = [
        {
            "measurement": "environment",
             "tags": {
                "name": username,
            },
            "time": utcnow,
            "fields": {
                "temperature": data['temperature'],
                "humidity": data['humidity']
            }
        }
    ]

    client.write_points(json_body)


@app.route("/data", methods=["POST"])

def getData_Occupant():
    writeToCSV_Occupant()
    writeToDB_Occupant()

    return ""

def writeToCSV_Occupant():
    """
    write data getting from Microsoft band to CSV file 
    
    """
    now = datetime.datetime.now()
    for keyword in data_keyword:
        for option in request.json:
            username = option["name"]
            with open('data/' + keyword + '-' + username + '-' + str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '.csv', 'ab') as csvfile:
                fieldnames = ['time', keyword]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                item =  option["value"]
                if item["type"] == keyword:
                        writer.writerow({'time': datetime.datetime.fromtimestamp(int(item["date"])).strftime('%Y-%m-%d %H:%M:%S'), 
                            keyword:item[keyword]})
                   
    with open('data/voting' + '-' + username + '-' + str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '.csv', 'ab') as csvfile:
        fieldnames = ['time', 'voting']
        votingwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for option in request.json:
            item =  option["value"]
            if item["type"] == "notification":
                print item["notification"]
                if "button" in item["notification"]:
                    voting = int(item["notification"][-1]) 
                    if voting == 2: # codest
                        voting = -3
                    elif voting == 3: 
                        voting = -2
                    elif voting == 4:   
                        voting = -1
                    elif voting == 5:   
                        voting = 0 
                    elif voting == 6:   
                        voting = 1 
                    elif voting == 7:   
                        voting = 2
                    else:
                        voting = 3 
                    votingwriter.writerow({'time': datetime.datetime.fromtimestamp(int(item["date"])).strftime('%Y-%m-%d %H:%M:%S'), 'voting':voting})

    return ""


# ref:https://influxdb-python.readthedocs.io/en/latest/include-readme.html#examples

# E:\research\influxdb\influxdb\influxdb-1.4.2-1>influx
#create database mydb
#show databases
#drop database mydb
def writeToDB_Occupant():
    """
    write data getting from Microsoft band to influxdb database

    """
    for key_word in data_keyword:
        for option in request.json:
            item =  option["value"]
            username = option["name"]
            if item["type"] == key_word:
                utcnow = datetime.datetime.utcnow()
                value = item[key_word]
                # for concept of database, ref:https://docs.influxdata.com/influxdb/v0.9/concepts/key_concepts/
                json_body = [
                    {
                        "measurement": dict_keyword[key_word],
                         "tags": {
                            "name": username,
                        },
                        "time": utcnow,
                        "fields": {
                            "value": value
                        }
                    }
                ]
   
                client.write_points(json_body)


    for option in request.json:
            item =  option["value"]
            username = option["name"]   
            if item["type"] == "notification":
                utcnow = datetime.datetime.utcnow()
                if "button" in item["notification"]:
                    voting = int(item["notification"][-1]) 
                    if voting == 2: # codest
                        voting = -3
                    elif voting == 3: 
                        voting = -2
                    elif voting == 4:   
                        voting = -1
                    elif voting == 5:   
                        voting = 0 
                    elif voting == 6:   
                        voting = 1 
                    elif voting == 7:   
                        voting = 2
                    else:
                        voting = 3 
                    value = voting
                    json_body = [
                        {
                            "measurement": "thermal_sensation",
                             "tags": {
                                "name": username,
                            },
                            "time": utcnow,
                            "fields": {
                                "value": value
                            }
                        }
                    ]
                    
                    client.write_points(json_body)
                    #result = client.query('select value from thermal_sensation;')
                    #print("Result: {0}".format(result))  

    return ""

def test():
    username = 'Chenlu'
    utcnow = datetime.datetime.utcnow()

    data =  {'temperature': 22, 'humidity': 38}
                
    json_body = [
        {
            "measurement": "environment",
             "tags": {
                "name": username,
            },
            "time": utcnow,
            "fields": {
                "temperature": data['temperature'],
                "humidity": data['humidity']
            }
        }
    ]

    client.write_points(json_body)
    json_body = [
            {
                "measurement": "heart_rate",
                 "tags": {
                    "name": 'Chenlu',
                },
                "time": utcnow,
                "fields": {
                    "value": 70
                }
            }
        ]
    client.write_points(json_body)
 

@app.route("/token", methods=['POST'])
def get_token(): 
    token = request.json['token']
    print token 
    return "" 

if __name__ == "__main__":
    client = InfluxDBClient(host='localhost', port=8086, username='chenlu', password='research', database='CMUMM409office')
    #test()
    app.run(host='0.0.0.0')

