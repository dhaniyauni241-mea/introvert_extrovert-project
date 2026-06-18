import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "Time_spent_Alone": 8,
    "Stage_fear": 1,
    "Social_event_attendance": 2,
    "Going_outside": 2,
    "Drained_after_socializing": 1,
    "Friends_circle_size": 3,
    "Post_frequency": 1
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())