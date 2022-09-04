import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = json.dumps({
  "text" : "@VirginAmerica plus you've added commercials to the experience... tacky."
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)