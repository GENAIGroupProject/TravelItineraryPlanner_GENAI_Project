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
        
        prompt = f"""You are a travel planning assistant.
The user wants a trip to {city}.

REFINED USER PROFILE:
{refined_profile}

CONSTRAINTS:
- With children: {with_children}
- With disabled traveler: {with_disabled}
- Budget (entire trip, group): {budget} EUR
- People: {people}

Propose EXACTLY 10 candidate attractions in {city} that match the user's interests and constraints.

For each attraction, output an object with:
- name
- short_description
- approx_price_per_person (number in EUR)
- tags: an array of strings, including some of: "museum", "outdoor", "nightlife",
        "kid_friendly", "wheelchair_friendly", "food", "viewpoint", "historical", etc.
- reason_for_user: one sentence explaining why this matches the profile.

Return ONLY a JSON array of EXACTLY 10 objects (no extra text).
Example format:
[
  {{
    "name": "Louvre Museum",
    "short_description": "World's largest art museum",
    "approx_price_per_person": 17,
    "tags": ["museum", "art", "wheelchair_friendly"],
    "reason_for_user": "Perfect for art lovers with extensive historical collections"
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
            
            # If too few attractions, use fallback
            if len(attractions) < 5:
                print(f"⚠️ Only {len(attractions)} valid attractions generated. Using fallback.")
                fallback = self.get_fallback_attractions(city)
                # Combine and remove duplicates
                all_attractions = attractions + fallback
                seen_names = set()
                unique_attractions = []
                for attr in all_attractions:
                    if attr.name not in seen_names:
                        seen_names.add(attr.name)
                        unique_attractions.append(attr)
                attractions = unique_attractions[:10]  # Keep only 10
            
            print(f"🎯 Final: {len(attractions)} attractions ready")
            return attractions[:10]  # Ensure max 10 attractions
            
        except Exception as e:
            print(f"❌ Error generating attractions: {e}")
            import traceback
            traceback.print_exc()
            return self.get_fallback_attractions(city)
    
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
    
    def get_fallback_attractions(self, city: str) -> List[Attraction]:
        """Provide fallback attractions when LLM fails."""
        print(f"🔄 Using fallback attractions for {city}")
        
        fallback_data = {
            "Rome": [
                {
                    "name": "Colosseum",
                    "short_description": "Ancient Roman amphitheater, iconic symbol of Rome",
                    "approx_price_per_person": 16.0,
                    "tags": ["historical", "ancient", "architecture", "wheelchair_friendly"],
                    "reason_for_user": "Perfect for ancient building enthusiasts with rich historical significance"
                },
                {
                    "name": "Roman Forum",
                    "short_description": "Ancient Roman government center with ruins and temples",
                    "approx_price_per_person": 12.0,
                    "tags": ["historical", "ancient", "archaeological", "outdoor"],
                    "reason_for_user": "Extensive ancient ruins perfect for history lovers"
                },
                {
                    "name": "Pantheon",
                    "short_description": "Ancient Roman temple with magnificent dome",
                    "approx_price_per_person": 0.0,
                    "tags": ["historical", "architecture", "religious", "free"],
                    "reason_for_user": "Well-preserved ancient building with incredible architecture"
                },
                {
                    "name": "Vatican Museums",
                    "short_description": "Extensive art collections including Sistine Chapel",
                    "approx_price_per_person": 17.0,
                    "tags": ["museum", "art", "religious", "historical"],
                    "reason_for_user": "World-class museum with historical and artistic treasures"
                },
                {
                    "name": "St. Peter's Basilica",
                    "short_description": "Renaissance church with dome and religious art",
                    "approx_price_per_person": 0.0,
                    "tags": ["religious", "architecture", "historical", "free"],
                    "reason_for_user": "Magnificent architecture and historical significance"
                }
            ],
            "Paris": [
                {
                    "name": "Louvre Museum",
                    "short_description": "World's largest art museum in historic palace",
                    "approx_price_per_person": 17.0,
                    "tags": ["museum", "art", "historical", "palace", "wheelchair_friendly"],
                    "reason_for_user": "Historical palace building with world's best art collection"
                },
                {
                    "name": "Eiffel Tower",
                    "short_description": "Iconic iron tower offering city views",
                    "approx_price_per_person": 25.0,
                    "tags": ["landmark", "architecture", "viewpoint", "historical"],
                    "reason_for_user": "Iconic architectural landmark with historical significance"
                },
                {
                    "name": "Notre-Dame Cathedral",
                    "short_description": "Medieval Catholic cathedral on the Île de la Cité",
                    "approx_price_per_person": 0.0,
                    "tags": ["religious", "gothic", "historical", "free"],
                    "reason_for_user": "Masterpiece of French Gothic architecture"
                }
            ],
            "London": [
                {
                    "name": "British Museum",
                    "short_description": "Museum of human history and culture",
                    "approx_price_per_person": 0.0,
                    "tags": ["museum", "history", "free", "wheelchair_friendly"],
                    "reason_for_user": "Free museum with extensive historical collections"
                },
                {
                    "name": "Tower of London",
                    "short_description": "Historic castle on the River Thames",
                    "approx_price_per_person": 29.0,
                    "tags": ["history", "castle", "landmark"],
                    "reason_for_user": "Rich historical site with crown jewels"
                },
                {
                    "name": "Westminster Abbey",
                    "short_description": "Gothic church and site of coronations",
                    "approx_price_per_person": 24.0,
                    "tags": ["religious", "gothic", "historical"],
                    "reason_for_user": "Historical church with royal connections"
                }
            ]
        }
        
        if city in fallback_data:
            attractions = []
            for attr_dict in fallback_data[city]:
                try:
                    attraction = Attraction(**attr_dict)
                    attractions.append(attraction)
                except Exception as e:
                    print(f"Error creating fallback attraction: {e}")
            return attractions
        else:
            # Generic attractions for any city
            return [
                Attraction(
                    name=f"{city} Historical Museum",
                    short_description="Local museum showcasing city history and ancient artifacts",
                    approx_price_per_person=12.0,
                    tags=["museum", "historical", "cultural"],
                    reason_for_user="Perfect for understanding local history and ancient cultures"
                ),
                Attraction(
                    name=f"{city} Old Town",
                    short_description="Historic district with ancient buildings and architecture",
                    approx_price_per_person=0.0,
                    tags=["historical", "architecture", "walking", "free", "outdoor"],
                    reason_for_user="Free exploration of ancient buildings and historical architecture"
                )
            ]
