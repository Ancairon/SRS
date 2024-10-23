from matplotlib.font_manager import json_dump
import pandas as pd
import requests


def getDataFromAPI(whom, ip, chart, dimension, timeStepsBack= 100 * 60):
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


getDataFromAPI("audio_classification", "192.168.1.5", "system.cpu", "user")
getDataFromAPI("audio_classification", "192.168.1.5", "system.cpu", "user")
getDataFromAPI("audio_classification", "192.168.1.5", "system.cpu_some_pressure_stall_time", "time")
getDataFromAPI("audio_classification", "192.168.1.5", "system.load", "load1")
getDataFromAPI("audio_classification", "192.168.1.5", "system.load", "load5")
getDataFromAPI("audio_classification", "192.168.1.5", "system.load", "load15")
getDataFromAPI("audio_classification", "192.168.1.5", "system.ram", "free")
getDataFromAPI("audio_classification", "192.168.1.5", "system.ram", "used")
getDataFromAPI("audio_classification", "192.168.1.5", "system.ram", "cached")
getDataFromAPI("audio_classification", "192.168.1.5", "system.ram", "buffers")
getDataFromAPI("audio_classification", "192.168.1.5", "system.processes", "running")
getDataFromAPI("audio_classification", "192.168.1.5", "system.active_processes", "active")
getDataFromAPI("audio_classification", "192.168.1.5", "system.file_nr_used", "used")
getDataFromAPI("audio_classification", "192.168.1.5", "mem.thp", "anonymous")
getDataFromAPI("audio_classification", "192.168.1.5", "ip.sockstat_sockets", "used")
getDataFromAPI("audio_classification", "192.168.1.5", "mem.committed", "Committed_AS")

###

# getDataFromAPI("server", "192.168.1.60", "system.cpu", "user")
# getDataFromAPI("server", "192.168.1.60", "net.eth0", "received")
# getDataFromAPI("server", "192.168.1.60", "net.eth0", "sent")

# getDataFromAPI("server", "192.168.1.60", "mysql_local.queries", "queries")
# getDataFromAPI("server", "192.168.1.60", "mysql_local.net", "out")