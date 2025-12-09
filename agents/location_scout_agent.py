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
            response_data = self.json_parser.parse_response(raw_response)
            
            # Convert to list if needed
            if not isinstance(response_data, list):
                response_data = [response_data] if isinstance(response_data, dict) else []
            
            # Validate and create Attraction objects
            attractions = self._validate_attractions(response_data, city)
            
            # If too few attractions, use fallback
            if len(attractions) < 5:
                print(f"Warning: Only {len(attractions)} attractions generated. Using fallback.")
                fallback = self.get_fallback_attractions(city)
                attractions.extend(fallback)
                attractions = attractions[:10]  # Keep only 10
            
            return attractions[:10]  # Ensure max 10 attractions
            
        except Exception as e:
            print(f"Error generating attractions: {e}")
            return self.get_fallback_attractions(city)
    
    def _validate_attractions(self, raw_attractions: List[Dict], city: str) -> List[Attraction]:
        """Validate and convert raw attraction data to Attraction objects."""
        validated = []
        
        for i, attr in enumerate(raw_attractions):
            if not isinstance(attr, dict):
                continue
            
            # Ensure required fields
            attr.setdefault("name", f"Attraction {i+1} in {city}")
            attr.setdefault("short_description", f"Popular attraction in {city}")
            attr.setdefault("approx_price_per_person", 15.0)
            attr.setdefault("tags", ["sightseeing"])
            attr.setdefault("reason_for_user", f"Recommended attraction in {city}")
            
            # Ensure price is numeric
            try:
                attr["approx_price_per_person"] = float(attr["approx_price_per_person"])
            except (ValueError, TypeError):
                attr["approx_price_per_person"] = 15.0
            
            try:
                attraction = Attraction(**attr)
                validated.append(attraction)
            except Exception as e:
                print(f"Skipping invalid attraction {i+1}: {e}")
                continue
        
        return validated
    
    def get_fallback_attractions(self, city: str) -> List[Attraction]:
        """Provide fallback attractions when LLM fails."""
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
                }
            ]
        }
        
        if city in fallback_data:
            attractions = [Attraction(**attr) for attr in fallback_data[city]]
        else:
            attractions = [
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
        
        return attractions