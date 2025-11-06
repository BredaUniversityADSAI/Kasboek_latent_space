"""
LLM API client for poem generation
"""

import json
import requests
from config import API_CONFIG, POEM_SETTINGS, DEBUG


class LLMClient:
    def __init__(self):
        self.api_url = API_CONFIG["api_url"]
        self.assistant_key = API_CONFIG["assistant_key"]
        self.user_key = API_CONFIG["user_key"]
        self.chat_key = None
        self.poem_count = 0

    def init_conversation(self) -> bool:
        """Initialize new conversation"""
        init_endpoint = f"{self.api_url}/actions/init_assistant"
        headers = {"Content-Type": "application/json"}

        data = {
            "user_key": self.user_key,
            "assistant_key": self.assistant_key
        }

        try:
            response = requests.post(init_endpoint,
                                     data=json.dumps(data),
                                     headers=headers,
                                     timeout=10)
            assistant = response.json()

            if assistant.get('status') == "Failed":
                print(f"   API Error: {assistant.get('error')}")
                return False

            self.chat_key = assistant.get("chat_key")

            if not self.chat_key:
                return False

            if DEBUG["verbose"]:
                print(f"   Chat initialized")

            return True

        except Exception as e:
            if DEBUG["verbose"]:
                print(f"   Init error: {e}")
            return False

    def generate_poem(self, prompt: str) -> str:
        """
        Generate poem from prompt with streaming support
        Returns poem text or fallback
        """
        if self.poem_count % POEM_SETTINGS["reinit_conversation_every"] == 0 or not self.chat_key:
            if DEBUG["verbose"]:
                print("   Initializing conversation...")
            if not self.init_conversation():
                return self._get_fallback_poem()

        self.poem_count += 1

        ask_endpoint = f"{self.api_url}/actions/ask_assistant"
        headers = {"Content-Type": "application/json"}

        data = {
            'assistant_key': self.assistant_key,
            'chat_key': self.chat_key,
            'message': prompt,
            'user_key': self.user_key
        }

        try:
            if DEBUG["verbose"]:
                print(f"   Sending request...")

            response = requests.post(ask_endpoint,
                                     data=json.dumps(data),
                                     headers=headers,
                                     timeout=POEM_SETTINGS["api_timeout"],
                                     stream=True)

            full_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded = chunk.decode('utf-8')
                    full_response += decoded
                    if DEBUG["verbose"]:
                        print(".", end='', flush=True)

            if DEBUG["verbose"]:
                print()
            try:
                result = json.loads(full_response)
                if isinstance(result, dict):
                    poem = result.get('response') or result.get('message') or result.get('text')
                else:
                    poem = full_response
            except json.JSONDecodeError:
                poem = full_response

            if not poem:
                if DEBUG["verbose"]:
                    print(f"   No poem in response")
                return self._get_fallback_poem()

            poem = poem.strip()

            if len(poem) < POEM_SETTINGS["min_poem_length"]:
                if DEBUG["verbose"]:
                    print(f"   Poem too short ({len(poem)} chars)")
                return self._get_fallback_poem()

            if DEBUG["verbose"]:
                print(f"   ✓ Poem received ({len(poem)} chars)")

            return poem

        except requests.Timeout:
            print(f"   Timeout after {POEM_SETTINGS['api_timeout']}s")
            return self._get_fallback_poem()

        except Exception as e:
            if DEBUG["verbose"]:
                print(f"   Generation error: {e}")
            return self._get_fallback_poem()

    def _get_fallback_poem(self) -> str:
        """Fallback poem when API fails"""
        return """In the gaze lies a story
Of what cannot be spoken
A pattern that unfolds

In the silence of looking"""
