from typing import List, Dict
from utils.data_structures import Attraction, DayItinerary, CompleteItinerary
from config import Config

class SchedulerAgent:
    """Agent for creating daily itineraries."""
    
    def __init__(self):
        self.time_slots = ["morning", "afternoon", "evening"]
    
    def create_itinerary(self, attractions: List[Attraction], 
                        days: int = None) -> CompleteItinerary:
        """Create a complete itinerary."""
        if days is None:
            days = Config.DEFAULT_DAYS
        
        # Initialize empty itinerary
        itinerary = {}
        
        # Group attractions by type for better distribution
        museums = [a for a in attractions if "museum" in a.tags]
        outdoor = [a for a in attractions if "outdoor" in a.tags]
        landmarks = [a for a in attractions if "landmark" in a.tags]
        other = [a for a in attractions if a not in museums + outdoor + landmarks]
        
        # Create mixed list for better variety
        mixed_attractions = []
        max_len = max(len(museums), len(outdoor), len(landmarks), len(other))
        
        for i in range(max_len):
            if i < len(museums):
                mixed_attractions.append(museums[i])
            if i < len(outdoor):
                mixed_attractions.append(outdoor[i])
            if i < len(landmarks):
                mixed_attractions.append(landmarks[i])
            if i < len(other):
                mixed_attractions.append(other[i])
        
        # If mixed list is empty, use original order
        if not mixed_attractions:
            mixed_attractions = attractions
        
        # Distribute attractions across days and time slots
        for day in range(1, days + 1):
            day_key = f"day{day}"
            day_slots = {slot: [] for slot in self.time_slots}
            
            # Assign attractions to time slots
            for slot_idx, slot in enumerate(self.time_slots):
                # Calculate which attraction goes in this slot
                attraction_idx = ((day - 1) * len(self.time_slots)) + slot_idx
                if attraction_idx < len(mixed_attractions):
                    day_slots[slot].append(mixed_attractions[attraction_idx])
            
            itinerary[day_key] = DayItinerary(**day_slots)
        
        # Fill CompleteItinerary object
        if days == 1:
            return CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=DayItinerary(),
                day3=DayItinerary()
            )
        elif days == 2:
            return CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=itinerary.get("day2", DayItinerary()),
                day3=DayItinerary()
            )
        else:
            return CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=itinerary.get("day2", DayItinerary()),
                day3=itinerary.get("day3", DayItinerary())
            )
    
    def optimize_itinerary(self, itinerary: CompleteItinerary) -> CompleteItinerary:
        """Optimize itinerary for better flow."""
        # This is a simple optimization - real implementation would consider:
        # - Opening hours
        # - Location proximity
        # - Travel time
        # - Energy levels throughout the day
        
        return itinerary
    
    def calculate_itinerary_metrics(self, itinerary: CompleteItinerary) -> Dict:
        """Calculate metrics for the itinerary."""
        total_attractions = 0
        morning_attractions = 0
        afternoon_attractions = 0
        evening_attractions = 0
        
        for day in [itinerary.day1, itinerary.day2, itinerary.day3]:
            total_attractions += len(day.morning) + len(day.afternoon) + len(day.evening)
            morning_attractions += len(day.morning)
            afternoon_attractions += len(day.afternoon)
            evening_attractions += len(day.evening)
        
        return {
            "total_attractions": total_attractions,
            "morning_attractions": morning_attractions,
            "afternoon_attractions": afternoon_attractions,
            "evening_attractions": evening_attractions,
            "attractions_per_day": total_attractions / 3 if total_attractions > 0 else 0,
            "balance_score": self._calculate_balance_score(morning_attractions, 
                                                          afternoon_attractions, 
                                                          evening_attractions)
        }
    
    def _calculate_balance_score(self, morning: int, afternoon: int, evening: int) -> float:
        """Calculate how balanced the itinerary is across time slots."""
        total = morning + afternoon + evening
        if total == 0:
            return 0.0
        
        # Ideal distribution would be roughly equal
        ideal = total / 3
        deviation = (abs(morning - ideal) + abs(afternoon - ideal) + abs(evening - ideal)) / 3
        
        # Convert to score (0-100, higher is better)
        max_deviation = total  # Worst case: all in one time slot
        if max_deviation == 0:
            return 100.0
        
        return 100 * (1 - deviation / max_deviation)