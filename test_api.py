import requests
r = requests.post("http://localhost:8000/podcast", json={"query": "Benefits of yoga and scientific facts", "max_generations": 3, "user_id": "Healthcare"})
print(r.status_code)
print(r.json())
