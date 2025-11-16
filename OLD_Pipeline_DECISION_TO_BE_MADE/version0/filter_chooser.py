import cv2
import numpy as np
from typing import List, Dict
import time
import base64
import os
import requests
import json

# Configuration - Change this to adjust how often filters are chosen
FILTER_CHOOSE_TIME = 15  # Seconds between AI filter changes

class FilterChooser:
    """
    LLM-based filter chooser that analyzes the video frame
    and decides which filters to apply using a local Ollama model.
    """
    
    def __init__(self, model_name="llava-phi3:latest"):
        """
        Initialize the filter chooser with a local Ollama model.
        
        Args:
            model_name: Ollama model to use (default: llava-phi3:latest)
                       Options: llava-phi3:latest, llava:7b, bakllava
        """
        self.model_name = model_name
        self.last_analysis_time = 0
        self.analysis_interval = FILTER_CHOOSE_TIME  # Use configurable time
        self.current_filters = []
        self.ollama_url = "http://localhost:11434"
        self.first_run = True  # Flag to force immediate first analysis
        self.initialize_model()
        
    def initialize_model(self):
        """Check if Ollama is running and the model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name in model_names:
                    print(f"✓ Ollama model '{self.model_name}' is ready!")
                else:
                    print(f"⚠ Model '{self.model_name}' not found.")
                    print(f"Available models: {', '.join(model_names)}")
                    print(f"\nTo install, run: ollama pull {self.model_name}")
            else:
                print("⚠ Ollama is running but returned an error")
        except requests.exceptions.ConnectionError:
            print("⚠ Ollama is not running!")
            print("Start it with: ollama serve")
            print(f"Then pull the model: ollama pull {self.model_name}")
        except Exception as e:
            print(f"ERROR checking Ollama: {e}")
    
    def analyze_frame(self, frame: np.ndarray) -> str:
        """
        Analyze the video frame using local Ollama vision model.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Description of the frame content
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for faster processing (smaller = faster)
            h, w = rgb_frame.shape[:2]
            max_size = 384  # Smaller size for faster local inference
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
            
            # Encode frame to JPEG with lower quality for speed
            _, buffer = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Call Ollama API
            payload = {
                "model": self.model_name,
                "prompt": "Describe this webcam image in one sentence. Focus on: the subject, activity, and mood.",
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '').strip()
                print(f"\n🎨 Frame Analysis: {description}")
                return description
            else:
                print(f"ERROR: Ollama returned status {response.status_code}")
                return "Error analyzing frame"
            
        except requests.exceptions.Timeout:
            print("ERROR: Ollama request timed out")
            return "Timeout"
        except Exception as e:
            print(f"ERROR analyzing frame: {e}")
            return "Error analyzing frame"
    
    def choose_filters(self, frame_description: str, available_filters: Dict[int, str]) -> List[int]:
        """
        Use local LLM to decide which filters to apply based on frame content.
        
        Args:
            frame_description: Text description of the frame
            available_filters: Dictionary of filter_num -> filter_name
            
        Returns:
            List of filter numbers to apply
        """
        try:
            # Create a concise filter list
            filter_list = ", ".join([f"{num}={name}" for num, name in available_filters.items()])
            
            prompt = f"""Scene: {frame_description}

Filters: {filter_list}

Pick 1-3 filter numbers that match the scene. Reply with ONLY numbers separated by commas (e.g., "3,7,13")."""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 20  # Limit output tokens for speed
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=8
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                print(f"🤖 AI recommends: {response_text}")
                
                # Parse the response to extract filter numbers
                filter_nums = []
                # Remove any text, just get numbers and commas
                cleaned = ''.join(c for c in response_text if c.isdigit() or c == ',')
                
                for part in cleaned.split(','):
                    try:
                        num = int(part.strip())
                        if num in available_filters:
                            filter_nums.append(num)
                    except ValueError:
                        continue
                
                # Limit to 3 filters max to avoid overwhelming effects
                return filter_nums[:3]
            else:
                print(f"ERROR: Ollama returned status {response.status_code}")
                return []
            
        except requests.exceptions.Timeout:
            print("ERROR: Filter selection timed out")
            return self.current_filters  # Keep previous filters
        except Exception as e:
            print(f"ERROR choosing filters: {e}")
            return []
    
    def should_analyze(self) -> bool:
        """
        Check if enough time has passed to run another analysis.
        
        Returns:
            True if analysis should run, False otherwise
        """
        # Always analyze on first run
        if self.first_run:
            self.first_run = False
            self.last_analysis_time = time.time()
            return True
        
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis_time = current_time
            return True
        return False
    
    def get_filters_for_frame(self, frame: np.ndarray, available_filters: Dict[int, str]) -> List[int]:
        """
        Main method to get filter recommendations for a frame.
        
        Args:
            frame: Input video frame
            available_filters: Dictionary of available filters
            
        Returns:
            List of filter numbers to apply
        """
        if self.should_analyze():
            # Analyze frame and choose new filters
            print(f"\n⏱️  Running AI analysis (next update in {FILTER_CHOOSE_TIME}s)...")
            description = self.analyze_frame(frame)
            self.current_filters = self.choose_filters(description, available_filters)
            
            if self.current_filters:
                filter_names_list = [available_filters[f] for f in self.current_filters]
                print(f"✅ Applied: {', '.join(filter_names_list)}\n")
        
        return self.current_filters
    
    def reset_timer(self):
        """Reset the analysis timer to force immediate analysis on next call"""
        self.first_run = True
    
    def cleanup(self):
        """Cleanup resources"""
        pass