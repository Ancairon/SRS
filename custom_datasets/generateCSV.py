from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os


def getDataFromAPI(whom, ip, chart, dimension, timeStepsBack=(60 * 60)*24*7):
    points = timeStepsBack

    query = 'http://{}:19999/api/v2/data?chart={}&dimension={}&after=-{}&before={}&points={}&group=average&gtime=0&tier=0&format=json&options=seconds&options=jsonwrap'.format(ip, chart, dimension, timeStepsBack, 0, points)
    print(query)

    r = requests.get(query, timeout=60*60)

    a = r.json()['result']['data']

    a.reverse()

    json_dump(a, "a.json")

    pdObj = pd.read_json("a.json")
    pdObj.to_csv("{0}.csv".format(whom + "_" + chart + "_" + dimension))

    os.remove("a.json")

name = "idle3_rpi"
ip = "192.168.1.27"

# getDataFromAPI(name, ip, "system.cpu", "user")
getDataFromAPI(name, ip, "system.cpu", "user")
getDataFromAPI(name, ip, "system.ram", "free")
getDataFromAPI(name, ip, "system.ram", "used")
getDataFromAPI(name, ip, "system.ram", "cached")
getDataFromAPI(name, ip, "system.ram", "buffers")
getDataFromAPI(name, ip, "system.processes", "running")
getDataFromAPI(name, ip, "system.active_processes", "active")
getDataFromAPI(name, ip, "system.file_nr_used", "used")
getDataFromAPI(name, ip, "mem.committed", "Committed_AS")
