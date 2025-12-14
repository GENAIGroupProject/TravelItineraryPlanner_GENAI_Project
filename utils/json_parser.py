import json
import re
from typing import Any, Dict, List

class JSONParser:
    """Parser for extracting JSON from LLM responses."""
    
    def __init__(self):
        pass
    
    def parse_response(self, text: str) -> Any:
        """
        Extract and parse JSON from LLM response.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Parsed JSON object
        """
        text = text.strip()
        
        # Try multiple extraction strategies
        json_str = self._extract_with_patterns(text)
        if not json_str:
            json_str = self._extract_with_braces(text)
        if not json_str:
            json_str = self._extract_entire_text(text)
        
        if not json_str:
            raise ValueError(f"No valid JSON found in response:\n{text}")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nExtracted text: {json_str}")
    
    def _extract_with_patterns(self, text: str) -> str:
        """Extract JSON using common patterns."""
        patterns = [
            r'```json\s*(.*?)\s*```',  # ```json {...} ```
            r'```\s*(.*?)\s*```',      # ``` {...} ```
            r'JSON:\s*(.*?)(?:\n\n|$)',  # JSON: {...}
            r'Here.*?JSON[:\s]*(.*?)(?:\n\n|$)',  # Here is the JSON: {...}
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if self._looks_like_json(candidate):
                    return candidate
        
        return ""
    
    def _extract_with_braces(self, text: str) -> str:
        """Extract JSON by finding matching braces."""
        # Find first {
        start = text.find('{')
        if start == -1:
            return ""
        
        # Count braces to find matching }
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        
        return ""
    
    def _extract_entire_text(self, text: str) -> str:
        """Try to parse entire text as JSON."""
        if self._looks_like_json(text):
            return text
        return ""
    
    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON."""
        text = text.strip()
        if not text:
            return False
        
        # Quick checks
        if text.startswith('{') and text.endswith('}'):
            return True
        if text.startswith('[') and text.endswith(']'):
            return True
        
        return False
    
    def validate_json_structure(self, data: Any, expected_structure: Dict) -> bool:
        """Validate JSON against expected structure."""
        if not isinstance(data, dict):
            return False
        
        for key, expected_type in expected_structure.items():
            if key not in data:
                return False
            
            if not isinstance(data[key], expected_type):
                return False
        
        return True