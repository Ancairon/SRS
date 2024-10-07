from matplotlib.font_manager import json_dump
import pandas as pd
import requests


def getDataFromAPI(whom, ip, chart, dimension, timeStepsBack= 14 * 24 * 60 * 60):
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


getDataFromAPI("ran", "localhost", "system.cpu", "user")
getDataFromAPI("ran", "localhost", "system.cpu", "user")
getDataFromAPI("ran", "localhost", "system.cpu_some_pressure_stall_time", "time")
getDataFromAPI("ran", "localhost", "system.load", "load1")
getDataFromAPI("ran", "localhost", "system.load", "load5")
getDataFromAPI("ran", "localhost", "system.load", "load15")
getDataFromAPI("ran", "localhost", "system.ram", "free")
getDataFromAPI("ran", "localhost", "system.ram", "used")
getDataFromAPI("ran", "localhost", "system.ram", "cached")
getDataFromAPI("ran", "localhost", "system.ram", "buffers")
getDataFromAPI("ran", "localhost", "system.processes", "running")
getDataFromAPI("ran", "localhost", "system.active_processes", "active")
getDataFromAPI("ran", "localhost", "system.file_nr_used", "used")
getDataFromAPI("ran", "localhost", "mem.thp", "anonymous")
getDataFromAPI("ran", "localhost", "ip.sockstat_sockets", "used")
getDataFromAPI("ran", "localhost", "mem.committed", "Committed_AS")

###

# getDataFromAPI("server", "192.168.1.60", "system.cpu", "user")
# getDataFromAPI("server", "192.168.1.60", "net.eth0", "received")
# getDataFromAPI("server", "192.168.1.60", "net.eth0", "sent")

# getDataFromAPI("server", "192.168.1.60", "mysql_local.queries", "queries")
# getDataFromAPI("server", "192.168.1.60", "mysql_local.net", "out")