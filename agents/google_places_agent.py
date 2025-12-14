import time
from typing import List, Dict, Optional
import requests
from utils.data_structures import Attraction
from config import Config

class GooglePlacesAgent:
    """Agent for enriching attractions with Google Places data."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.enabled = True
    
    def enrich_attractions(self, attractions: List[Attraction], 
                          city: str) -> List[Attraction]:
        """Enrich attractions with Google Places data."""
        if not self.enabled:
            print("⚠️ Google Places API not enabled. Using default data.")
            return attractions
        
        enriched_attractions = []
        
        for i, attraction in enumerate(attractions):
            print(f"  Enriching attraction {i+1}/{len(attractions)}: {attraction.name}")
            
            try:
                # Find place ID
                place_id = self._find_place_id(attraction.name, city)
                if not place_id:
                    print(f"    Could not find place ID for {attraction.name}")
                    enriched_attractions.append(attraction)
                    continue
                
                # Get place details
                details = self._get_place_details(place_id)
                if not details:
                    enriched_attractions.append(attraction)
                    continue
                
                # Update attraction with enriched data
                enriched_attraction = self._enrich_attraction(attraction, details, place_id)
                enriched_attractions.append(enriched_attraction)
                
                # Polite delay between requests
                time.sleep(Config.GOOGLE_REQUEST_DELAY)
                
            except Exception as e:
                print(f"    Error enriching {attraction.name}: {e}")
                enriched_attractions.append(attraction)
        
        return enriched_attractions
    
    def _find_place_id(self, name: str, city: str) -> Optional[str]:
        """Find Google Place ID for an attraction."""
        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        params = {
            "input": f"{name}, {city}",
            "inputtype": "textquery",
            "fields": "place_id",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("candidates"):
                return data["candidates"][0]["place_id"]
            
        except Exception as e:
            print(f"    Error finding place ID: {e}")
        
        return None
    
    def _get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information for a place."""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,geometry,opening_hours,price_level,types,rating,user_ratings_total,wheelchair_accessible_entrance",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data.get("result")
            
        except Exception as e:
            print(f"    Error getting place details: {e}")
        
        return None
    
    def _enrich_attraction(self, attraction: Attraction, 
                          details: Dict, place_id: str) -> Attraction:
        """Enrich attraction object with Google Places data."""
        # Create a copy of the attraction to avoid modifying original
        enriched_data = attraction.dict()
        
        # Add Google Places data
        enriched_data["google_place_id"] = place_id
        enriched_data["opening_hours"] = details.get("opening_hours")
        enriched_data["google_price_level"] = details.get("price_level")
        enriched_data["location"] = details.get("geometry", {}).get("location")
        enriched_data["google_rating"] = details.get("rating")
        enriched_data["google_user_ratings_total"] = details.get("user_ratings_total")
        
        # Update tags with Google types
        google_types = details.get("types", [])
        existing_tags = set(enriched_data["tags"])
        
        # Map Google types to our tag system
        type_mapping = {
            "museum": "museum",
            "art_gallery": "museum",
            "park": "outdoor",
            "tourist_attraction": "landmark",
            "point_of_interest": "landmark",
            "historical_landmark": "historical",
            "church": "religious",
            "restaurant": "food",
            "cafe": "food",
            "bar": "food",
            "zoo": "nature",
            "aquarium": "nature",
            "amusement_park": "entertainment",
            "movie_theater": "entertainment",
            "shopping_mall": "shopping",
            "store": "shopping",
            "night_club": "entertainment",
            "stadium": "sports",
            "gym": "sports"
        }
        
        for google_type in google_types:
            if google_type in type_mapping:
                existing_tags.add(type_mapping[google_type])
        
        enriched_data["tags"] = list(existing_tags)
        
        return Attraction(**enriched_data)
    
    def is_enabled(self) -> bool:
        """Check if Google Places API is enabled."""
        return self.enabled