import urllib.request
import json

try:
    response = urllib.request.urlopen("http://localhost:8000/analytics")
    data = json.loads(response.read().decode())
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
