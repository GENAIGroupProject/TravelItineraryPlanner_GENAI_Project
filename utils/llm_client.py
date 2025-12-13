import requests
import json
from typing import Dict, Any, Optional
import time
from config import Config

class LLMClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.LLAMA_MODEL_NAME
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            temperature: Creativity parameter (0.0-1.0)
            
        Returns:
            LLM response text
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=Config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data["response"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1. Ollama is installed (https://ollama.ai/)\n"
                "2. Server is running: 'ollama serve'\n"
                "3. Model is pulled: 'ollama pull llama3'"
            )
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3, 
                           delay: float = 2.0) -> str:
        """Generate with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    def check_health(self) -> bool:
        """Check if LLM server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
