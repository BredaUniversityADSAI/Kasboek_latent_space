import requests
import json

API_URL = "https://ai-assistants.buas.nl/aioda-api" 
ASSISTANT_KEY = "8c1cdd40-14e7-460b-acaf-0f874f50703e"
USER_KEY = "b371ec4e-b0eb-4e3a-8657-5b3739afddb6"

init_endpoint_url = f"{API_URL}/actions/init_assistant"
ask_endpoint_url = f"https://ai-assistants.buas.nl/aioda-api/actions/ask_assistant" 
headers = {"Content-Type": "application/json"}

# INITIALIZE THE AI ASSISTANT (CREATE A NEW CONVERSATION)
data = {"user_key": USER_KEY,
        "assistant_key": ASSISTANT_KEY}
response = requests.post(init_endpoint_url, data=json.dumps(data), headers=headers)
assistant = response.json()

if assistant['status'] == "Failed":
    print(f"ERROR: {assistant['error']}")
    exit(-1)

# GET THE KEY THAT IDENTIFIES THE CONVERSATION
data = {'assistant_key': ASSISTANT_KEY,
        'chat_key': assistant["chat_key"],
        'message': input("What is your message?: "),
        'user_key': USER_KEY}

response = requests.post(ask_endpoint_url, data=json.dumps(data), headers=headers)

print(response.text)