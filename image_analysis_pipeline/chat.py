import requests
import json

class LLMModel:

    _attributes = ["name", "description", "purpose", "expertise",
        "capabilities", "restrictions", "additional",
        "llm", "creativity"]
    
    def __init__(self, key):
        with open('.env', 'r') as file:
            access_token = file.readlines()[1].split('=')[1]
        with open('.env', 'r') as file:
            user_key = file.readlines()[2].split('=')[1]

        self.assistant_key = key
        self._user_key = user_key
        self._access_token = access_token
        self.__url_api = "https://ai-assistants.buas.nl/aioda-api"
        self.__headers = {"Content-Type": "application/json"}
        self.__init_endpoint_url = f"{self.__url_api}/actions/init_assistant"
        self.__assistant_configs_endpoint_url = f"{self.__url_api}/actions/get_assistant"
        self.__assistant_update_endpoint_url = f"{self.__url_api}/actions/update_assistant"
        self.__ask_endpoint_url = f"{self.__url_api}/actions/ask_assistant"
        self._name = None
        self._description = None
        self._purpose = None
        self._expertise = None
        self._capabilities = None
        self._restrictions = None
        self._additional = None
        self._llm = None
        self._creativity = None
        self.__chat_key = None

        self._attr_storage = {attr: None for attr in self._attributes}

    def initialize(self):
        init_data = {"user_key": self._user_key,
                    "assistant_key": self.assistant_key}
        
        init_response = requests.post(self.__init_endpoint_url, data=json.dumps(init_data), headers=self.__headers)
        init_response = init_response.json()

        # Return error if one occurred
        if init_response['status'] == "Failed":
            print(f"ERROR: {init_response['error']}")
            exit(-1)

        self.__chat_key = init_response["chat_key"]

        assistant_params_data = {
            "assistant_key": self.assistant_key,
            "user_key": self._user_key,
            "access_token": self._access_token,
            "chat_key": self.__chat_key
        }

        assistant_params_response = requests.post(self.__assistant_configs_endpoint_url, data=json.dumps(assistant_params_data), headers=self.__headers)
        assistant_params_response = assistant_params_response.json()

        self._name = assistant_params_response["name"]
        self._description = assistant_params_response["description"]
        self._purpose = assistant_params_response["purpose"]
        self._expertise = assistant_params_response["expertise"]
        self._capabilities = assistant_params_response["capabilities"]
        self._restrictions = assistant_params_response["restrictions"]
        self._additional = assistant_params_response["additional"]
        self._llm = assistant_params_response["llm"]
        self._creativity = assistant_params_response["temperature"]
        
        return init_response
    

    def chat(self, message):
        chat_data = {'assistant_key': self.assistant_key,
                     'chat_key': self.__chat_key, # All prompts and responses in a session are saved to the same chat
                     'message': message, # prompt
                     'user_key': self._user_key}
        
        chat_response = requests.post(self.__ask_endpoint_url, data=json.dumps(chat_data), headers=self.__headers)
        chat_response = chat_response.text

        return chat_response


    def update_assistant(self, name: str = None, description: str = None, purpose: str = None, expertise: str = None, capabilities: str = None, restrictions: str = None, additional: str = None, llm: str = None, temperature: float = 1.0, image_generation=False):
        update_assistant_data = {
            "access_token": self._access_token,
            "assistant_key": self.assistant_key,
            "name": name if name != None else self._name,
            "description": description if description != None else self._description,
            "purpose": purpose if purpose != None else self._purpose,
            "expertise": expertise if expertise != None else self._expertise,
            "capabilities": capabilities if capabilities != None else self._capabilities,
            "restrictions": restrictions if restrictions != None else self._restrictions,
            "additional": additional if additional != None else self._additional,
            "llm": llm if llm != None else self._llm,
            "api_key": "",
            "temperature": temperature if temperature != None else 1,
            "image_generation": image_generation if image_generation != None else False,
            "visibility": 0
        }

        assistant_params_response = requests.post(self.__assistant_update_endpoint_url, data=json.dumps(update_assistant_data), headers=self.__headers)
        assistant_params_response = assistant_params_response.json()

        reinit_response = self.initialize()

        return reinit_response
