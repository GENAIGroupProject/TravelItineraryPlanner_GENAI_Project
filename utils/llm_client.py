import hashlib
import json
import os
import pickle
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

import requests

from config import Config


class LLMClient:
    """Client for interacting with Ollama LLM (+ streaming support)."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or getattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or getattr(Config, "LLAMA_MODEL_NAME", "llama3")
        self.cache_dir = ".llm_cache"
        self._init_cache()

    # -----------------------------
    # Cache
    # -----------------------------
    def _init_cache(self):
        if getattr(Config, "ENABLE_CACHING", False):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        content = f"{prompt}_{temperature}_{self.model}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        if not getattr(Config, "ENABLE_CACHING", False):
            return None

        cache_file = os.path.join(self.cache_dir, cache_key)
        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)

            cache_time = cached_data.get("timestamp")
            ttl = getattr(Config, "CACHE_TTL_SECONDS", 0)
            if cache_time and datetime.now() - cache_time < timedelta(seconds=ttl):
                return cached_data.get("response")
        except Exception:
            return None

        return None

    def _save_to_cache(self, cache_key: str, response: str):
        if not getattr(Config, "ENABLE_CACHING", False):
            return

        cache_file = os.path.join(self.cache_dir, cache_key)
        try:
            cached_data = {
                "response": response,
                "timestamp": datetime.now(),
                "model": self.model,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
        except Exception:
            pass

    # -----------------------------
    # Non-stream generate (existing behavior)
    # -----------------------------
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a full response (non-stream)."""

        cache_key = self._get_cache_key(prompt, temperature)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        timeout = getattr(Config, "REQUEST_TIMEOUT", 60)

        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            response = data.get("response", "")

            self._save_to_cache(cache_key, response)
            return response

        except requests.exceptions.Timeout:
            raise Exception(f"LLM request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1) Ollama is running: 'ollama serve'\n"
                "2) Model is pulled: 'ollama pull llama3'\n"
                f"3) Base URL is correct: {self.base_url}"
            )
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")

    # -----------------------------
    # Streaming (NEW)
    # -----------------------------
    def _stream_ollama_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        extra_options: Optional[Dict[str, Any]] = None,
        request_timeout: Optional[Any] = None,
    ) -> Generator[str, None, None]:
        """
        Low-level streaming generator from Ollama /api/generate.
        Ollama streams JSON lines with fields like:
          {"response":"...", "done": false}
          ...
          {"done": true}
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if extra_options:
            payload["options"].update(extra_options)

        timeout = request_timeout if request_timeout is not None else getattr(Config, "REQUEST_TIMEOUT", 60)

        # For streaming, use (connect, read) timeout tuple if a single number was provided
        if isinstance(timeout, (int, float)):
            timeout = (min(10, timeout), max(30, timeout))

        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    chunk = data.get("response")
                    if chunk:
                        yield chunk

                    if data.get("done") is True:
                        break

        except requests.exceptions.Timeout:
            raise Exception(f"LLM streaming request timed out (timeout={timeout})")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama server. Please ensure:\n"
                "1) Ollama is running: 'ollama serve'\n"
                "2) Model is pulled: 'ollama pull llama3'\n"
                f"3) Base URL is correct: {self.base_url}"
            )
        except Exception as e:
            raise Exception(f"LLM streaming request failed: {str(e)}")

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Public streaming generator: yields raw text chunks."""
        yield from self._stream_ollama_generate(prompt, temperature=temperature, extra_options=extra_options)

    def generate_stream_words(
        self,
        prompt: str,
        temperature: float = 0.7,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generator that yields whitespace-delimited pieces for a typing effect.
        Suitable for Streamlit st.write_stream().
        """
        buffer = ""

        for chunk in self.generate_stream(prompt, temperature=temperature, extra_options=extra_options):
            buffer += chunk

            # Emit pieces whenever we see whitespace boundary
            while True:
                m = re.search(r"\s+", buffer)
                if not m:
                    break
                cut = m.end()
                piece = buffer[:cut]
                buffer = buffer[cut:]
                yield piece

        if buffer:
            yield buffer

    # -----------------------------
    # Your JSON extraction helpers (kept)
    # -----------------------------
    def extract_json_from_response(self, response: str) -> List[dict]:
        response = (response or "").strip()

        try:
            data = json.loads(response)
            return self._normalize_attraction_data(data)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._normalize_attraction_data(data)
            except json.JSONDecodeError:
                pass

        fixed = self._fix_common_json_issues(response)
        try:
            data = json.loads(fixed)
            return self._normalize_attraction_data(data)
        except json.JSONDecodeError:
            pass

        return self._extract_objects_manually(response)

    def _fix_common_json_issues(self, text: str) -> str:
        lines = text.strip().split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            line = line.rstrip()
            if i > 0 and line.startswith("{") and fixed_lines and fixed_lines[-1].endswith("}"):
                fixed_lines[-1] = fixed_lines[-1] + ","
            fixed_lines.append(line)

        result = "\n".join(fixed_lines)
        result = re.sub(r",\s*\]", "]", result)
        result = re.sub(r",\s*\}", "}", result)
        return result

    def _normalize_attraction_data(self, data: Any) -> List[dict]:
        attractions = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for _, value in data.items():
                if isinstance(value, list):
                    items = value
                    break
            else:
                items = [data]
        else:
            return []

        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name") or item.get("Name") or item.get("title") or item.get("attraction")
            desc = item.get("short_description") or item.get("shortDescription") or item.get("description") or item.get("Description")

            if name:
                attractions.append(
                    {
                        "name": str(name).strip(),
                        "short_description": str(desc).strip()[:100] if desc else "",
                    }
                )

        return attractions

    def _extract_objects_manually(self, text: str) -> List[dict]:
        attractions = []

        pattern = r"\"name\"\s*:\s*\"([^\"]+)\"[^}]+\"short_description\"\s*:\s*\"([^\"]+)\""
        matches = re.findall(pattern, text)
        for name, desc in matches:
            attractions.append({"name": name.strip(), "short_description": desc.strip()[:100]})

        if not attractions:
            pattern2 = r"\"Name\"\s*:\s*\"([^\"]+)\"[^}]+\"Description\"\s*:\s*\"([^\"]+)\""
            matches = re.findall(pattern2, text)
            for name, desc in matches:
                attractions.append({"name": name.strip(), "short_description": desc.strip()[:100]})

        return attractions

    # Convenience methods you already used
    def generate_with_retry(self, prompt: str, max_retries: int = 3, delay: float = 2.0) -> str:
        for attempt in range(max_retries):
            try:
                return self.generate(prompt)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass
        return []
