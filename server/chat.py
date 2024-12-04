import requests
import json


if __name__ == "__main__":
    api_url = "http://127.0.0.1:5000/chat"
    payload = {
        "prompt": "What is topic modeling? How many documents have you used for your answer?",
        "history": [],
        # "docs": []
    }
    response = requests.post(api_url, json=payload)
    parsed = json.loads(response.text)
    print(json.dumps(parsed, indent=4))
    print(len(response.json()["documents"]))