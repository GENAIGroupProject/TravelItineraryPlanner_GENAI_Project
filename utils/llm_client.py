import hashlib
import pickle
import os
import json
import re
import time
from datetime import datetime, timedelta
from typing import Optional, List, Any

import requests

from config import Config

class LLMClient:
    """Client for interacting with Ollama LLM with optimizations."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.LLAMA_MODEL_NAME
        self.cache_dir = ".llm_cache"
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache directory."""
        if Config.ENABLE_CACHING:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate cache key from prompt and parameters."""
        content = f"{prompt}_{temperature}_{self.model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get response from cache if valid."""
        if not Config.ENABLE_CACHING:
            return None
            
        cache_file = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid
                cache_time = cached_data.get('timestamp')
                if cache_time and datetime.now() - cache_time < timedelta(seconds=Config.CACHE_TTL_SECONDS):
                    return cached_data.get('response')
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        if not Config.ENABLE_CACHING:
            return
            
        cache_file = os.path.join(self.cache_dir, cache_key)
        try:
            cached_data = {
                'response': response,
                'timestamp': datetime.now(),
                'model': self.model
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception:
            pass
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response from LLM with caching.
        Uses OLD STYLE - simple and reliable request.
        """
        # Try cache first
        cache_key = self._get_cache_key(prompt, temperature)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        url = f"{self.base_url}/api/generate"
        
        # Use SIMPLE payload like the old version
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        try:
            # Use old timeout value
            resp = requests.post(url, json=payload, timeout=Config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            response = data["response"]
            
            # Cache the response
            self._save_to_cache(cache_key, response)
            
            return response
            
        except requests.exceptions.Timeout:
            raise Exception(f"LLM request timed out after {Config.REQUEST_TIMEOUT} seconds")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1. Ollama is installed (https://ollama.ai/)\n"
                "2. Server is running: 'ollama serve'\n"
                "3. Model is pulled: 'ollama pull llama3'"
            )
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")
    
    def extract_json_from_response(self, response: str) -> List[dict]:
        """
        Extract JSON from LLM response using OLD STYLE parsing.
        This mimics what was working in the old version.
        """
        response = response.strip()
        
        # Try to parse directly first
        try:
            data = json.loads(response)
            return self._normalize_attraction_data(data)
        except json.JSONDecodeError:
            pass
        
        # If direct parse fails, try to extract JSON
        # Look for JSON array pattern
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._normalize_attraction_data(data)
            except json.JSONDecodeError:
                pass
        
        # Try to fix common JSON issues
        fixed_response = self._fix_common_json_issues(response)
        try:
            data = json.loads(fixed_response)
            return self._normalize_attraction_data(data)
        except json.JSONDecodeError:
            pass
        
        # Last resort: extract objects manually
        return self._extract_objects_manually(response)
    
    def _fix_common_json_issues(self, text: str) -> str:
        """Fix common JSON issues found in LLM responses."""
        lines = text.strip().split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            line = line.rstrip()
            
            # Fix missing comma between objects
            if i > 0 and line.startswith('{') and fixed_lines and fixed_lines[-1].endswith('}'):
                fixed_lines[-1] = fixed_lines[-1] + ','
            
            fixed_lines.append(line)
        
        result = '\n'.join(fixed_lines)
        
        # Fix trailing commas
        result = re.sub(r',\s*\]', ']', result)
        result = re.sub(r',\s*\}', '}', result)
        
        return result
    
    def _normalize_attraction_data(self, data: Any) -> List[dict]:
        """Normalize attraction data from various formats."""
        attractions = []
        
        # Handle different data structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Check if there's an array in the dict
            for key, value in data.items():
                if isinstance(value, list):
                    items = value
                    break
            else:
                items = [data]  # Single object
        else:
            return []
        
        # Process each item
        for item in items:
            if isinstance(item, dict):
                name = None
                description = None
                
                # Try different field names
                if 'name' in item:
                    name = item['name']
                elif 'Name' in item:
                    name = item['Name']
                elif 'title' in item:
                    name = item['title']
                elif 'attraction' in item:
                    name = item['attraction']
                
                if 'short_description' in item:
                    description = item['short_description']
                elif 'shortDescription' in item:
                    description = item['shortDescription']
                elif 'description' in item:
                    description = item['description']
                elif 'Description' in item:
                    description = item['Description']
                
                if name:
                    attractions.append({
                        'name': str(name).strip(),
                        'short_description': str(description).strip()[:100] if description else ""
                    })
        
        return attractions
    
    def _extract_objects_manually(self, text: str) -> List[dict]:
        """Extract attraction objects manually using regex patterns."""
        attractions = []
        
        # Pattern for objects with name and description
        pattern = r'"name"\s*:\s*"([^"]+)"[^}]+"short_description"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, text)
        
        for name, desc in matches:
            attractions.append({
                'name': name.strip(),
                'short_description': desc.strip()[:100]
            })
        
        # Alternative pattern
        if not attractions:
            pattern2 = r'"Name"\s*:\s*"([^"]+)"[^}]+"Description"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern2, text)
            for name, desc in matches:
                attractions.append({
                    'name': name.strip(),
                    'short_description': desc.strip()[:100]
                })
        
        return attractions
    
    def generate_attractions(self, city: str, count: int = 5) -> List[dict]:
        """
        Generate attractions for a city.
        Uses OLD STYLE prompt that was working.
        """
        # Use the OLD prompt style that was working
        prompt = f"""Generate a list of {count} tourist attractions in {city}. 
For each attraction, provide the name and a short description.
Format the response as a JSON array of objects, each with "name" and "short_description" fields.

Example:
[
  {{"name": "Attraction 1", "short_description": "Description 1"}},
  {{"name": "Attraction 2", "short_description": "Description 2"}}
]

Now generate {count} attractions for {city}:"""
        
        try:
            print(f"\n📤 Requesting {count} attractions for {city}...")
            response = self.generate(prompt, temperature=0.7)
            
            print(f"\n📥 Received response ({len(response)} chars)")
            print(f"First 200 chars: {response[:200]}...")
            
            # Use OLD STYLE JSON extraction
            attractions = self.extract_json_from_response(response)
            
            print(f"✅ Parsed {len(attractions)} attractions")
            
            # If we got some but not all, that's OK
            if attractions:
                return attractions[:count]
            else:
                print("❌ No attractions parsed from response")
                return []
                
        except Exception as e:
            print(f"❌ Error generating attractions: {e}")
            return []
    
    # Methods from old version
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
                delay *= 2
    
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