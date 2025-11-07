import requests
import json
import logging

def credentials(ASSISTANT_KEY: str):
    '''
    Create valid credentials to use custom created models on the BUas AI Assistants website

    Params:
        ASSISTANT_KEY (str): unique ID of the assistant, obtained from the chat URL of the assistant
    
    Returns:
        ASSISTANT_KEY: same key as the parameter
        USER_KEY: ID of the user that uses the models
        init_endpoint_url: endpoint URL used to initialize the model
        ask_endpoint_url: endpoint URL used to post prompts and query results
        headers: HTTP header
    '''

    API_URL = "https://ai-assistants.buas.nl/aioda-api"
    USER_KEY = "b371ec4e-b0eb-4e3a-8657-5b3739afddb6"

    init_endpoint_url = f"{API_URL}/actions/init_assistant"
    ask_endpoint_url = f"{API_URL}/actions/ask_assistant" 
    headers = {"Content-Type": "application/json"}

    return ASSISTANT_KEY, USER_KEY, init_endpoint_url, ask_endpoint_url, headers

def initialize_ai_assistant(ASSISTANT_KEY: str, USER_KEY: str, init_endpoint_url: str, headers: dict):
    '''
    Initialize AI assistant
    
    Params:
        ASSISTANT_KEY: ID of the model to initialize
        USER_KEY: ID of the user that uses the model
        init_endpoint_url: initialization endpoint used to initialize the model
        headers: HTTP header

    '''
    # Package credentials in a dictionary to be converted to json
    data = {"user_key": USER_KEY,
            "assistant_key": ASSISTANT_KEY}
    
    # Send data to the website and get the response
    response = requests.post(init_endpoint_url, data=json.dumps(data), headers=headers)

    # Convert json response to a dictionary
    response = response.json()

    # Return error if one occurred
    if response['status'] == "Failed":
        print(f"ERROR: {response['error']}")
        exit(-1)
    
    return response
    

def chat_with_ai_assistant(response: dict, message: str, ASSISTANT_KEY: str, USER_KEY: str, ask_endpoint_url: str, headers: dict):
    '''
    Send prompt to the model and obtain the response

    Params:
        response: response of the initaliziation
        message: prompt to be sent to the model
        ASSISTANT_KEY: the model the prompt is directed to
        USER_KEY: ID of the user that uses the model
        ask_endpoint_url: post URL used to send and receive data
        headers: HTTP header

    Returns:
        reponse.text: the response of the model without metadata
    '''

    # Package credentials in a dictionary to be converted to json
    data = {'assistant_key': ASSISTANT_KEY,
            'chat_key': response["chat_key"], # All prompts and responses in a session are saved to the same chat
            'message': message, # prompt
            'user_key': USER_KEY}

    # Convert json response to a dictionary
    response = requests.post(ask_endpoint_url, data=json.dumps(data), headers=headers)

    return response.text