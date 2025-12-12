import json
from typing import List, Dict, Optional
from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import Attraction
from config import Config

class LocationScoutAgent:
    """Agent for generating attractions in a chosen city."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
    
    def generate_attractions(self, city: str, refined_profile: str, 
                           constraints: Dict) -> List[Attraction]:
        """Generate attractions for the chosen city."""
        with_children = constraints.get("with_children", False)
        with_disabled = constraints.get("with_disabled", False)
        budget = constraints.get("budget", Config.DEFAULT_BUDGET)
        people = constraints.get("people", Config.DEFAULT_PEOPLE)
        
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

Return ONLY a JSON array of EXACTLY 10 objects (no extra text).
Example format:
[
  {{
    "name": "Forest Hiking Trail",
    "short_description": "Beautiful trail through ancient forest",
    "approx_price_per_person": 0,
    "tags": ["hiking", "forest", "nature", "outdoor", "free"],
    "reason_for_user": "Perfect for nature lovers who enjoy hiking in forests"
  }},
  {{...}},  // 9 more attractions
]
"""

        try:
            raw_response = self.llm_client.generate(prompt, temperature=0.8)
            print(f"\n🔍 Raw LLM response for attractions:\n{raw_response[:500]}...\n")
            
            response_data = self.json_parser.parse_response(raw_response)
            
            # Convert to list if needed
            if not isinstance(response_data, list):
                print(f"⚠️ Response is not a list, type: {type(response_data)}")
                if isinstance(response_data, dict):
                    response_data = [response_data]
                else:
                    response_data = []
            
            print(f"✅ Parsed {len(response_data)} attractions from LLM")
            
            # Validate and create Attraction objects
            attractions = self._validate_attractions(response_data, city)
            
            print(f"🎯 Final: {len(attractions)} attractions ready")
            return attractions[:10]  # Ensure max 10 attractions
            
        except Exception as e:
            print(f"❌ Error generating attractions: {e}")
            import traceback
            traceback.print_exc()
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