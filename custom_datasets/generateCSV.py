from matplotlib.font_manager import json_dump
import pandas as pd
import requests


def getDataFromAPI(whom, ip, chart, dimension, timeStepsBack=60 * 60):
    points = timeStepsBack

    r = requests.get(
        'http://{}:19999/api/v1/data?chart={}&dim'
        'ension={}&after=-{}&before={}&points={}&group=average&gtime=0&format=json&options=seconds&options'
        '=jsonwrap'.format(ip, chart, dimension, timeStepsBack, 0, points))

    a = r.json()['result']['data']

    a.reverse()

    json_dump(a, "a.json")

    pdObj = pd.read_json("a.json")
    pdObj.to_csv("{0}.csv".format(whom + "_" + chart + "_" + dimension))


name = "face_detection_rpi"
ip = "192.168.1.27"

getDataFromAPI(name, ip, "system.cpu", "user")
getDataFromAPI(name, ip, "system.cpu", "user")
getDataFromAPI(name, ip, "system.ram", "free")
getDataFromAPI(name, ip, "system.ram", "used")
getDataFromAPI(name, ip, "system.ram", "cached")
getDataFromAPI(name, ip, "system.ram", "buffers")
getDataFromAPI(name, ip, "system.processes", "running")
getDataFromAPI(name, ip, "system.active_processes", "active")
getDataFromAPI(name, ip, "system.file_nr_used", "used")
getDataFromAPI(name, ip, "mem.committed", "Committed_AS")
