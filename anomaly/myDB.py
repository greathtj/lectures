from influxdb_client import InfluxDBClient

# InfluxDB connection details
url = "http://218.148.55.186:32037"
token = "1sUlWTDrQFWVREAL7u1lpwUyeX70tvWsXk7IOGXC3wI_JupPGKh_qzU_4dP-csvprL2cZlRhBqRfwuXCDqfYKA=="
org = "kitech"
bucket = "mqtt"

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

query = '''
from(bucket: "mqtt")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "sensor_kitech")
  |> filter(fn: (r) => r["_field"] == "amplitude")
  |> filter(fn: (r) => r["deviceId"] == "todd3367")
'''

# Execute query and get CSV iterator
result = query_api.query_csv(query)

# Save to CSV
with open("output.csv", "w") as f:
    for line in result:
        f.write(",".join(line) + "\n")  # Convert list to CSV row