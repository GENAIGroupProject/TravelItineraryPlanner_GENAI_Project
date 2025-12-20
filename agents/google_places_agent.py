import time
from typing import List, Dict, Optional
import requests
from utils.data_structures import Attraction
from config import Config
from utils.logging_utils import log_step, log_agent_communication

class GooglePlacesAgent:
    """Agent for enriching attractions with Google Places data."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.enabled = True
        log_step("GOOGLE_PLACES", f"Google Places agent initialized. Enabled: {self.enabled}")
    
    def enrich_attractions(self, attractions: List[Attraction], city: str) -> List[Attraction]:
        """Enrich attractions with Google Places data."""
        if not self.enabled:
            log_step("GOOGLE_PLACES", "API not enabled. Using default data.", level="warning")
            return attractions
        
        log_step("GOOGLE_PLACES", f"Starting enrichment of {len(attractions)} attractions in {city}")
        log_agent_communication(
            from_agent="GooglePlacesAgent",
            to_agent="Processing",
            message_type="enrichment_start",
            data={
                "city": city,
                "attraction_count": len(attractions),
                "api_key_configured": bool(self.api_key and self.api_key != Config.GOOGLE_API_KEY)
            }
        )
        
        enriched_attractions = []
        successful_enrichments = 0
        failed_enrichments = 0
        
        for i, attraction in enumerate(attractions):
            attraction_name = attraction.name[:30]
            log_step("GOOGLE_PLACES", f"Enriching attraction {i+1}/{len(attractions)}: {attraction_name}...", level="debug")
            
            try:
                place_id = self._find_place_id(attraction.name, city)
                if not place_id:
                    log_step("GOOGLE_PLACES", f"Could not find place ID for {attraction_name}", level="warning")
                    enriched_attractions.append(attraction)
                    failed_enrichments += 1
                    continue
                
                details = self._get_place_details(place_id)
                if not details:
                    log_step("GOOGLE_PLACES", f"No details found for {attraction_name}", level="warning")
                    enriched_attractions.append(attraction)
                    failed_enrichments += 1
                    continue
                
                enriched_attraction = self._enrich_attraction(attraction, details, place_id)
                enriched_attractions.append(enriched_attraction)
                successful_enrichments += 1
                
                time.sleep(Config.GOOGLE_REQUEST_DELAY)
                
            except Exception as e:
                log_step("GOOGLE_PLACES", f"Error enriching {attraction_name}: {e}", level="error")
                enriched_attractions.append(attraction)
                failed_enrichments += 1
        
        log_step("GOOGLE_PLACES", f"Enrichment complete: {successful_enrichments} successful, {failed_enrichments} failed")
        log_agent_communication(
            from_agent="GooglePlacesAgent",
            to_agent="BudgetAgent",
            message_type="enrichment_complete",
            data={
                "city": city,
                "total_attractions": len(attractions),
                "successful_enrichments": successful_enrichments,
                "failed_enrichments": failed_enrichments,
                "has_google_data": successful_enrichments > 0
            }
        )
        
        return enriched_attractions
    
    def _find_place_id(self, name: str, city: str) -> Optional[str]:
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
            log_step("GOOGLE_PLACES", f"Error finding place ID for '{name[:30]}...': {e}", level="error")
        
        return None
    
    def _get_place_details(self, place_id: str) -> Optional[Dict]:
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
            log_step("GOOGLE_PLACES", f"Error getting place details: {e}", level="error")
        
        return None
    
    def _enrich_attraction(self, attraction: Attraction, details: Dict, place_id: str) -> Attraction:
        enriched_data = attraction.dict()

        # Ensure tags exists
        if not isinstance(enriched_data.get("tags"), list):
            enriched_data["tags"] = []

        enriched_data["google_place_id"] = place_id
        
        if "opening_hours" in details:
            enriched_data["opening_hours"] = details["opening_hours"]
        
        if "price_level" in details:
            enriched_data["google_price_level"] = details["price_level"]
        
        if "geometry" in details and "location" in details["geometry"]:
            enriched_data["location"] = details["geometry"]["location"]
        
        if "rating" in details:
            enriched_data["google_rating"] = details["rating"]
        
        if "user_ratings_total" in details:
            enriched_data["google_user_ratings_total"] = details["user_ratings_total"]
        
        google_types = details.get("types", [])
        existing_tags = set(enriched_data["tags"])
        
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
        return self.enabled and bool(self.api_key)
