import time
import pandas as pd
import requests
import os


def getDataFromAPI(whom, ip, chart, dimension, points=60):
    """
    Fetches a time series from Netdata v3 API and saves a CSV with columns
    [timestamp,value] (timestamps in milliseconds, ascending).

    :param whom: prefix (dataset name)
    :param ip: host (e.g. 'localhost')
    :param chart: chart/context id (contexts parameter)
    :param dimension: dimension name
    :param points: number of points to request (default 60)
    """
    now = int(time.time())

    url = (
        f"http://{ip}:19999/api/v1/data?chart={chart}"
        f"&dimension={dimension}&after=-{points}&before={now}&points={points}"
        "&group=average&format=json&options=seconds,jsonwrap"
    )

    print('[DEBUG] Fetching Netdata URL:', url)
    print('[DEBUG] Request timestamp (before):', now, '=', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)))

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    data = r.json()
    # Expect v3 format: { result: { labels: [...], data: [[timestamp, val], ...] } }
    if not data or 'result' not in data or 'data' not in data['result']:
        raise RuntimeError('Unexpected Netdata response format: no result.data')

    raw = data['result']['data']

    # Build series of (timestamp_ms, value)
    series = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        try:
            ts_s = int(row[0])
            val = row[1]
        except Exception:
            continue
        series.append({'timestamp': ts_s * 1000, 'value': val})

    if not series:
        print(f"Warning: no data points returned for {chart} - {dimension}")
        return

    # sort ascending by timestamp (older -> newer)
    series.sort(key=lambda x: x['timestamp'])

    df = pd.DataFrame(series)

    # Save CSV with two columns so downstream combiner can pick the last column (value)
    os.makedirs(f"custom_datasets/{whom}", exist_ok=True)
    filename = f"custom_datasets/{whom}/{whom}_{chart}_{dimension}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved {filename} ({len(df)} rows)")


name = "nginx"
# ip = "192.168.1.123"
ip = "localhost"

# fetch a handful of common charts/dimensions
getDataFromAPI(name, ip, "system.cpu", "user")
getDataFromAPI(name, ip, "system.ram", "free")
getDataFromAPI(name, ip, "system.ram", "used")
getDataFromAPI(name, ip, "system.ram", "cached")
getDataFromAPI(name, ip, "system.ram", "buffers")
getDataFromAPI(name, ip, "system.processes", "running")
getDataFromAPI(name, ip, "system.active_processes", "active")
getDataFromAPI(name, ip, "system.file_nr_used", "used")
getDataFromAPI(name, ip, "mem.committed", "Committed_AS")
