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
        
        # If no attractions, return empty itinerary
        if not attractions:
            return self._create_empty_itinerary(days)
        
        print(f"📅 Scheduler: Creating {days}-day itinerary with {len(attractions)} attractions")
        
        # Initialize empty itinerary for all days
        itinerary = {}
        for day in range(1, days + 1):
            day_key = f"day{day}"
            itinerary[day_key] = DayItinerary()
        
        # DEBUG: Show what we're working with
        print(f"  Time slots: {self.time_slots}")
        print(f"  Days to fill: {days}")
        
        # Simple distribution: fill each day's slots in order
        attraction_idx = 0
        day_idx = 1
        
        while attraction_idx < len(attractions) and day_idx <= days:
            day_key = f"day{day_idx}"
            day_itinerary = itinerary[day_key]
            
            # Try to fill each time slot for this day
            for slot in self.time_slots:
                if attraction_idx >= len(attractions):
                    break
                    
                # Get current slot list
                slot_list = getattr(day_itinerary, slot)
                
                # Add attraction to this slot
                slot_list.append(attractions[attraction_idx])
                print(f"  ➡️ Added '{attractions[attraction_idx].name[:30]}...' to day{day_idx} {slot}")
                attraction_idx += 1
            
            # Move to next day
            day_idx += 1
        
        # If we still have attractions and all days are "full" (1 per slot),
        # add more to existing days
        if attraction_idx < len(attractions):
            print(f"  ⚠️ Still have {len(attractions) - attraction_idx} attractions left, adding to existing days")
            
            # Distribute remaining attractions evenly across days
            day_idx = 1
            while attraction_idx < len(attractions):
                day_key = f"day{day_idx}"
                day_itinerary = itinerary[day_key]
                
                # Find which slot has the fewest attractions
                slot_counts = {
                    "morning": len(day_itinerary.morning),
                    "afternoon": len(day_itinerary.afternoon),
                    "evening": len(day_itinerary.evening)
                }
                min_slot = min(slot_counts, key=slot_counts.get)
                
                # Add to the slot with fewest attractions
                slot_list = getattr(day_itinerary, min_slot)
                slot_list.append(attractions[attraction_idx])
                print(f"  ➕ Added '{attractions[attraction_idx].name[:30]}...' to day{day_idx} {min_slot} (balance)")
                attraction_idx += 1
                
                # Move to next day for next attraction
                day_idx = (day_idx % days) + 1
        
        # Create CompleteItinerary object
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
    
    def _create_empty_itinerary(self, days: int) -> CompleteItinerary:
        """Create an empty itinerary structure."""
        if days == 1:
            return CompleteItinerary(
                day1=DayItinerary(),
                day2=DayItinerary(),
                day3=DayItinerary()
            )
        elif days == 2:
            return CompleteItinerary(
                day1=DayItinerary(),
                day2=DayItinerary(),
                day3=DayItinerary()
            )
        else:
            return CompleteItinerary(
                day1=DayItinerary(),
                day2=DayItinerary(),
                day3=DayItinerary()
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