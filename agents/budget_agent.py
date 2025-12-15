from typing import List, Dict
from utils.data_structures import Attraction
from config import Config
from utils.logging_utils import log_step, log_agent_communication

class BudgetAgent:
    """Agent for filtering attractions based on budget constraints."""
    
    def __init__(self):
        pass
    
    def filter_by_budget(self, attractions: List[Attraction], 
                        total_budget: float, days: int, 
                        people: int) -> List[Attraction]:
        """Filter attractions to fit within budget."""
        log_step("BUDGET_AGENT", f"Starting budget filtering for {len(attractions)} attractions")
        log_agent_communication(
            from_agent="BudgetAgent",
            to_agent="Processing",
            message_type="filter_start",
            data={
                "total_budget": total_budget,
                "days": days,
                "people": people,
                "input_attractions": len(attractions)
            }
        )
        
        # Calculate max attractions (3 per day)
        max_attractions = days * 3
        
        # Calculate total price for each attraction
        log_step("BUDGET_AGENT", "Calculating prices for attractions")
        for attraction in attractions:
            price_per_person = self._estimate_price(attraction)
            attraction.final_price_estimate = price_per_person * people
        
        # Sort by price (cheapest first)
        sorted_attractions = sorted(attractions, 
                                  key=lambda x: x.final_price_estimate or float('inf'))
        
        # Select attractions within budget
        selected = []
        total_cost = 0.0
        
        log_step("BUDGET_AGENT", f"Selecting up to {max_attractions} attractions within {total_budget}€ budget")
        
        for attraction in sorted_attractions:
            if len(selected) >= max_attractions:
                log_step("BUDGET_AGENT", f"Reached maximum attractions per day ({max_attractions})")
                break
            
            attraction_cost = attraction.final_price_estimate or 0
            if total_cost + attraction_cost <= total_budget:
                selected.append(attraction)
                total_cost += attraction_cost
                log_step("BUDGET_AGENT", f"Added '{attraction.name[:30]}...' ({attraction_cost}€), total: {total_cost}€", level="debug")
            else:
                # Try to see if we can still add free attractions
                if attraction_cost == 0 and len(selected) < max_attractions:
                    selected.append(attraction)
                    # total_cost doesn't increase for free attractions
                    log_step("BUDGET_AGENT", f"Added free attraction '{attraction.name[:30]}...'", level="debug")
        
        log_step("BUDGET_AGENT", f"Selected {len(selected)} attractions with total cost {total_cost:.2f}€")
        log_agent_communication(
            from_agent="BudgetAgent",
            to_agent="SchedulerAgent",
            message_type="filter_complete",
            data={
                "selected_count": len(selected),
                "total_cost": total_cost,
                "remaining_budget": total_budget - total_cost,
                "budget_utilization": (total_cost / total_budget * 100) if total_budget > 0 else 0
            }
        )
        
        return selected
    
    def _estimate_price(self, attraction: Attraction) -> float:
        """Estimate price for an attraction."""
        # Use approx_price_per_person if available
        if attraction.approx_price_per_person is not None:
            return attraction.approx_price_per_person
        
        # Use Google price level if available
        if attraction.google_price_level is not None:
            # Convert price level (0-4) to EUR
            price_map = {0: 0, 1: 10, 2: 20, 3: 40, 4: 60}
            estimated_price = price_map.get(attraction.google_price_level, 20.0)
            log_step("BUDGET_AGENT", f"Using Google price level {attraction.google_price_level} -> {estimated_price}€ for '{attraction.name[:20]}...'", level="debug")
            return estimated_price
        
        # Default price based on tags
        if any(tag in attraction.tags for tag in ["museum", "gallery"]):
            log_step("BUDGET_AGENT", f"Using museum price (15€) for '{attraction.name[:20]}...'", level="debug")
            return 15.0
        elif any(tag in attraction.tags for tag in ["landmark", "viewpoint"]):
            log_step("BUDGET_AGENT", f"Using landmark price (20€) for '{attraction.name[:20]}...'", level="debug")
            return 20.0
        elif any(tag in attraction.tags for tag in ["outdoor", "park", "free"]):
            log_step("BUDGET_AGENT", f"Using free price (0€) for '{attraction.name[:20]}...'", level="debug")
            return 0.0
        else:
            log_step("BUDGET_AGENT", f"Using default price (10€) for '{attraction.name[:20]}...'", level="debug")
            return 10.0
    
    def calculate_budget_summary(self, attractions: List[Attraction], 
                               total_budget: float, people: int) -> Dict:
        """Calculate budget summary."""
        log_step("BUDGET_AGENT", "Calculating budget summary")
        
        total_cost = sum(attr.final_price_estimate or 0 for attr in attractions)
        remaining_budget = total_budget - total_cost
        cost_per_person = total_cost / people if people > 0 else 0
        
        summary = {
            "total_cost": total_cost,
            "remaining_budget": remaining_budget,
            "cost_per_person": cost_per_person,
            "budget_utilization": (total_cost / total_budget * 100) if total_budget > 0 else 0,
            "number_of_attractions": len(attractions)
        }
        
        log_step("BUDGET_AGENT", f"Budget summary: {summary}")
        
        return summary