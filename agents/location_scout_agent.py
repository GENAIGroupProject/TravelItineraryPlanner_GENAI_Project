import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional
from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import Attraction
from config import Config
from utils.logging_utils import log_agent_communication, log_step

class LocationScoutAgent:
    """Agent for generating attractions in a chosen city."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
        self.log_dir = "logs"
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _log_to_file(self, filename: str, content: str, mode: str = 'w'):
        """Log content to a file for detailed debugging."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Detailed log saved to {filepath}")
    
    def _log_llm_response(self, city: str, prompt: str, response: str, parsed_data: List):
        """Log complete LLM interaction for debugging to separate file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attractions_{city}_{timestamp}.log"
        
        log_content = f"""===========================================
LOCATION SCOUT DEBUG LOG - {timestamp}
City: {city}
===========================================

PROMPT SENT TO LLM:
{prompt}

{'='*50}

FULL LLM RESPONSE:
{response}

{'='*50}

RESPONSE STATS:
- Total length: {len(response)} characters
- Parsed attractions: {len(parsed_data) if isinstance(parsed_data, list) else 'N/A'}

{'='*50}

PARSED DATA:
"""
        if isinstance(parsed_data, list) and parsed_data:
            for i, attraction in enumerate(parsed_data[:5]):  # Log first 5
                log_content += f"\n{i+1}. {attraction.get('name', 'Unknown')}\n"
        
        self._log_to_file(filename, log_content)
    
    def _extract_json_array(self, text: str) -> str:
        """Extract JSON array from text response."""
        # Look for the main JSON array (most complete one)
        matches = re.findall(r'(\[.*\])', text, re.DOTALL)
        
        if matches:
            # Return the longest match (likely the main array)
            return max(matches, key=len)
        
        # If no complete array found, try to find start of array
        start_idx = text.find('[')
        if start_idx != -1:
            # Try to find a matching closing bracket
            bracket_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == '[':
                    bracket_count += 1
                elif text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[start_idx:i+1]
        
        return ""
    
    def _parse_json_safely(self, json_str: str):
        """Try multiple methods to parse JSON safely."""
        # Method 1: Direct parse
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (method 1): {e}")
        
        # Method 2: Try to fix common issues
        fixed_json = self._fix_json_issues(json_str)
        try:
            data = json.loads(fixed_json)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (method 2): {e}")
        
        # Method 3: Extract individual objects
        try:
            objects = self._extract_json_objects(json_str)
            return objects
        except Exception as e:
            print(f"  Object extraction error: {e}")
        
        return []
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix missing quotes on keys (simple cases)
        json_str = re.sub(r'(\{|\,\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Ensure array is properly closed
        if json_str.count('[') > json_str.count(']'):
            json_str += ']' * (json_str.count('[') - json_str.count(']'))
        
        # Ensure objects are properly closed
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    def _extract_json_objects(self, text: str) -> List[Dict]:
        """Extract individual JSON objects from text."""
        objects = []
        # Find all potential JSON objects
        pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and 'name' in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        return objects

    def generate_attractions(self, city: str, refined_profile: str, 
                           constraints: Dict) -> List[Attraction]:
        """Generate attractions for the chosen city."""
        log_step("LOCATION_SCOUT", f"Generating attractions for {city}")
        
        with_children = constraints.get("with_children", False)
        with_disabled = constraints.get("with_disabled", False)
        budget = constraints.get("budget", Config.DEFAULT_BUDGET)
        people = constraints.get("people", Config.DEFAULT_PEOPLE)
        
        # Log agent communication to unified log
        log_agent_communication(
            from_agent="LocationScoutAgent",
            to_agent="LLM",
            message_type="attraction_generation_request",
            data={
                "city": city,
                "refined_profile": refined_profile[:200],  # Truncate for logging
                "constraints": {
                    "with_children": with_children,
                    "with_disabled": with_disabled,
                    "budget": budget,
                    "people": people
                }
            },
            city=city
        )
        
        # Analyze user preferences from refined_profile
        user_wants = self._analyze_user_preferences(refined_profile)
        
        prompt = f"""You are a travel planning assistant.
The user wants a trip to {city}.

USER'S ACTUAL PREFERENCES:
{refined_profile}

SPECIFIC USER REQUIREMENTS:
{user_wants}

CONSTRAINTS:
- With children: {with_children}
- With disabled traveler: {with_disabled}
- Budget (entire trip, group): {budget} EUR
- People: {people}

CRITICAL: Match attractions to user's ACTUAL preferences.
If user wants hiking/forests/parks - suggest outdoor nature attractions.
If user does NOT want cultural/historical - avoid museums and historical sites.

Propose EXACTLY 10 candidate attractions in {city} that match the user's interests and constraints.

For each attraction, output an object with:
- name
- short_description
- approx_price_per_person (number in EUR)
- tags: an array of strings (choose appropriate tags based on user preferences)
- reason_for_user: one sentence explaining why this matches the ACTUAL profile.

IMPORTANT: Return ONLY a JSON array of EXACTLY 10 objects. Do NOT include any text before or after the JSON array.
Example format:
[
  {{
    "name": "Forest Hiking Trail",
    "short_description": "Beautiful trail through ancient forest",
    "approx_price_per_person": 0,
    "tags": ["hiking", "forest", "nature", "outdoor", "free"],
    "reason_for_user": "Perfect for nature lovers who enjoy hiking in forests"
  }},
  {{...}}  // 9 more attractions
]
"""

        try:
            raw_response = self.llm_client.generate(prompt, temperature=0.8)
            
            # Print full response length
            print(f"\n🔍 LLM Response Stats:")
            print(f"  Total length: {len(raw_response)} characters")
            
            # Log to unified log
            log_step("LLM", f"Received response of {len(raw_response)} chars for {city}")
            
            # Try to extract JSON
            json_str = self._extract_json_array(raw_response)
            
            if json_str:
                print(f"✅ Extracted JSON string ({len(json_str)} chars)")
                
                # Try to parse with multiple methods
                response_data = self._parse_json_safely(json_str)
            else:
                print("❌ No JSON array found in response")
                response_data = []
            
            parsed_count = len(response_data) if isinstance(response_data, list) else 0
            print(f"✅ Parsed {parsed_count} attractions from LLM")
            
            # Log LLM response for debugging (separate file)
            self._log_llm_response(city, prompt, raw_response, response_data)
            
            # Log agent communication - response received (unified log)
            log_agent_communication(
                from_agent="LLM",
                to_agent="LocationScoutAgent",
                message_type="attraction_generation_response",
                data={
                    "raw_response_length": len(raw_response),
                    "parsed_count": parsed_count,
                    "sample_attractions": response_data[:2] if response_data else []
                },
                city=city
            )
            
            # Validate and create Attraction objects
            attractions = self._validate_attractions(response_data, city)
            
            log_step("LOCATION_SCOUT", f"Generated {len(attractions)} attractions for {city}")
            
            # Log final attractions to unified log
            log_agent_communication(
                from_agent="LocationScoutAgent",
                to_agent="BudgetAgent",
                message_type="attractions_output",
                data={
                    "city": city,
                    "attraction_count": len(attractions),
                    "attraction_names": [attr.name[:30] for attr in attractions[:5]]
                },
                city=city
            )
            
            return attractions[:10]  # Ensure max 10 attractions
            
        except Exception as e:
            error_msg = f"Error generating attractions: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Log error to unified log
            log_step("LOCATION_SCOUT", error_msg, level="error")
            
            # Return empty list instead of fallback
            return []

    def _analyze_user_preferences(self, refined_profile: str) -> str:
        """Analyze user preferences to guide attraction selection."""
        profile_lower = refined_profile.lower()
        
        requirements = []
        
        if "hiking" in profile_lower or "forest" in profile_lower or "park" in profile_lower:
            requirements.append("- MUST INCLUDE: Hiking trails, forests, parks, outdoor nature areas")
        
        if "not cultural" in profile_lower or "no cultural" in profile_lower or "not historical" in profile_lower:
            requirements.append("- MUST AVOID: Museums, historical sites, cultural attractions")
        elif "cultural" in profile_lower or "museum" in profile_lower or "historical" in profile_lower:
            requirements.append("- SHOULD INCLUDE: Cultural and historical attractions")
        
        if "relaxed" in profile_lower or "slow" in profile_lower:
            requirements.append("- PREFER: Relaxed pace activities, not crowded/touristy")
        elif "fast" in profile_lower or "busy" in profile_lower:
            requirements.append("- PREFER: Active, fast-paced experiences")
        
        if "food" in profile_lower or "cuisine" in profile_lower:
            requirements.append("- INCLUDE: Local food experiences, restaurants, markets")
        
        if not requirements:
            return "No specific requirements detected"
        
        return "\n".join(requirements)
    
    def _validate_attractions(self, raw_attractions: List[Dict], city: str) -> List[Attraction]:
        """Validate and convert raw attraction data to Attraction objects."""
        validated = []
        
        for i, attr in enumerate(raw_attractions):
            if not isinstance(attr, dict):
                print(f"Skipping non-dict item {i}: {type(attr)}")
                continue
            
            # Ensure required fields
            attr.setdefault("name", f"Attraction {i+1} in {city}")
            attr.setdefault("short_description", f"Popular attraction in {city}")
            
            # Handle price - ensure it's a number
            price = attr.get("approx_price_per_person", 15.0)
            try:
                if isinstance(price, (int, float)):
                    attr["approx_price_per_person"] = float(price)
                elif isinstance(price, str):
                    # Try to extract number from string
                    import re
                    numbers = re.findall(r'\d+\.?\d*', price)
                    if numbers:
                        attr["approx_price_per_person"] = float(numbers[0])
                    else:
                        attr["approx_price_per_person"] = 15.0
                else:
                    attr["approx_price_per_person"] = 15.0
            except (ValueError, TypeError):
                attr["approx_price_per_person"] = 15.0
            
            attr.setdefault("tags", ["sightseeing"])
            attr.setdefault("reason_for_user", f"Recommended attraction in {city}")
            
            try:
                attraction = Attraction(**attr)
                validated.append(attraction)
            except Exception as e:
                print(f"⚠️ Skipping invalid attraction {i+1} '{attr.get('name', 'unknown')}': {e}")
                continue
        
        return validated