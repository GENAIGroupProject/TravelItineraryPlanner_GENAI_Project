from typing import List, Dict
from utils.data_structures import Attraction, DayItinerary, CompleteItinerary
from config import Config
from utils.logging_utils import log_step, log_agent_communication

class SchedulerAgent:
    """Agent for creating daily itineraries."""
    
    def __init__(self):
        self.time_slots = ["morning", "afternoon", "evening"]
        log_step("SCHEDULER", "Scheduler agent initialized")
    
    def create_itinerary(self, attractions: List[Attraction], 
                        days: int = None) -> CompleteItinerary:
        """Create a complete itinerary."""
        if days is None:
            days = Config.DEFAULT_DAYS
        
        log_step("SCHEDULER", f"Starting itinerary creation with {len(attractions)} attractions for {days} days")
        log_agent_communication(
            from_agent="SchedulerAgent",
            to_agent="Processing",
            message_type="itinerary_creation_start",
            data={
                "attraction_count": len(attractions),
                "days": days,
                "time_slots": self.time_slots
            }
        )
        
        # If no attractions, return empty itinerary
        if not attractions:
            log_step("SCHEDULER", "No attractions provided, returning empty itinerary", level="warning")
            return self._create_empty_itinerary(days)
        
        print(f"📅 Scheduler: Creating {days}-day itinerary with {len(attractions)} attractions")
        
        # Initialize empty itinerary for all days
        itinerary = {}
        for day in range(1, days + 1):
            day_key = f"day{day}"
            itinerary[day_key] = DayItinerary()
        
        # DEBUG: Show what we're working with
        log_step("SCHEDULER", f"Time slots: {self.time_slots}, Days to fill: {days}", level="debug")
        
        # Simple distribution: fill each day's slots in order
        attraction_idx = 0
        day_idx = 1
        
        log_step("SCHEDULER", "Distributing attractions across days and time slots")
        
        while attraction_idx < len(attractions) and day_idx <= days:
            day_key = f"day{day_idx}"
            day_itinerary = itinerary[day_key]
            
            # Try to fill each time slot for this day
            for slot in self.time_slots:
                if attraction_idx >= len(attractions):
                    log_step("SCHEDULER", f"No more attractions to distribute", level="debug")
                    break
                    
                # Get current slot list
                slot_list = getattr(day_itinerary, slot)
                
                # Add attraction to this slot
                slot_list.append(attractions[attraction_idx])
                log_step("SCHEDULER", f"Added '{attractions[attraction_idx].name[:30]}...' to day{day_idx} {slot}", level="debug")
                attraction_idx += 1
            
            # Move to next day
            day_idx += 1
        
        # If we still have attractions and all days are "full" (1 per slot),
        # add more to existing days
        if attraction_idx < len(attractions):
            remaining = len(attractions) - attraction_idx
            log_step("SCHEDULER", f"Still have {remaining} attractions left, adding to existing days")
            
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
                log_step("SCHEDULER", f"Added '{attractions[attraction_idx].name[:30]}...' to day{day_idx} {min_slot} (balance)", level="debug")
                attraction_idx += 1
                
                # Move to next day for next attraction
                day_idx = (day_idx % days) + 1
        
        # Create CompleteItinerary object
        if days == 1:
            final_itinerary = CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=DayItinerary(),
                day3=DayItinerary()
            )
        elif days == 2:
            final_itinerary = CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=itinerary.get("day2", DayItinerary()),
                day3=DayItinerary()
            )
        else:
            final_itinerary = CompleteItinerary(
                day1=itinerary.get("day1", DayItinerary()),
                day2=itinerary.get("day2", DayItinerary()),
                day3=itinerary.get("day3", DayItinerary())
            )
        
        # Calculate metrics
        metrics = self.calculate_itinerary_metrics(final_itinerary)
        log_step("SCHEDULER", f"Itinerary created successfully: {metrics}")
        
        log_agent_communication(
            from_agent="SchedulerAgent",
            to_agent="EvaluationAgent",
            message_type="itinerary_complete",
            data={
                "days": days,
                "total_attractions": metrics["total_attractions"],
                "balance_score": metrics["balance_score"],
                "attractions_per_day": metrics["attractions_per_day"]
            }
        )
        
        return final_itinerary
    
    def _create_empty_itinerary(self, days: int) -> CompleteItinerary:
        """Create an empty itinerary structure."""
        log_step("SCHEDULER", f"Creating empty itinerary for {days} days", level="debug")
        
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
        log_step("SCHEDULER", "Optimizing itinerary")
        
        # This is a simple optimization - real implementation would consider:
        # - Opening hours
        # - Location proximity
        # - Travel time
        # - Energy levels throughout the day
        
        log_step("SCHEDULER", "Itinerary optimization complete", level="debug")
        return itinerary
    
    def calculate_itinerary_metrics(self, itinerary: CompleteItinerary) -> Dict:
        """Calculate metrics for the itinerary."""
        log_step("SCHEDULER", "Calculating itinerary metrics")
        
        total_attractions = 0
        morning_attractions = 0
        afternoon_attractions = 0
        evening_attractions = 0
        
        for day in [itinerary.day1, itinerary.day2, itinerary.day3]:
            total_attractions += len(day.morning) + len(day.afternoon) + len(day.evening)
            morning_attractions += len(day.morning)
            afternoon_attractions += len(day.afternoon)
            evening_attractions += len(day.evening)
        
        balance_score = self._calculate_balance_score(morning_attractions, 
                                                      afternoon_attractions, 
                                                      evening_attractions)
        
        metrics = {
            "total_attractions": total_attractions,
            "morning_attractions": morning_attractions,
            "afternoon_attractions": afternoon_attractions,
            "evening_attractions": evening_attractions,
            "attractions_per_day": total_attractions / 3 if total_attractions > 0 else 0,
            "balance_score": balance_score
        }
        
        log_step("SCHEDULER", f"Metrics calculated: {metrics}", level="debug")
        return metrics
    
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
        
        score = 100 * (1 - deviation / max_deviation)
        log_step("SCHEDULER", f"Balance score: morning={morning}, afternoon={afternoon}, evening={evening}, score={score:.1f}", level="debug")
        return score