#!/usr/bin/env python3
"""
Main entry point for Travel Itinerary Planner
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import json
    import time
    from typing import Dict
    
    from config import Config
    
    # Try to import logging utils
    try:
        from utils.logging_utils import setup_logging, log_step, Timer
        LOGGING_AVAILABLE = True
    except ImportError as e:
        print(f"Note: Logging utils not available: {e}")
        LOGGING_AVAILABLE = False
    
    # Try to import agents
    try:
        from agents.semantic_agent import SemanticAgent
        from agents.interest_refinement_agent import InterestRefinementAgent
        from agents.location_scout_agent import LocationScoutAgent
        from agents.budget_agent import BudgetAgent
        from agents.scheduler_agent import SchedulerAgent
        from agents.evaluation_agent import EvaluationAgent
        from agents.google_places_agent import GooglePlacesAgent
        from utils.data_structures import PreferenceState
        AGENTS_AVAILABLE = True
    except ImportError as e:
        print(f"Error importing agents: {e}")
        AGENTS_AVAILABLE = False
        sys.exit(1)
    
except ImportError as e:
    print(f"Fatal import error: {e}")
    print("\nPlease ensure:")
    print("1. All required packages are installed: pip install -r requirements.txt")
    print("2. You're in the correct directory")
    sys.exit(1)


class TravelPlanner:
    """Main orchestrator for travel planning pipeline."""
    
    def __init__(self):
        if LOGGING_AVAILABLE:
            self.logger = setup_logging()
        else:
            self.logger = None
        
        self.semantic_agent = SemanticAgent()
        self.interest_agent = InterestRefinementAgent()
        self.location_agent = LocationScoutAgent()
        self.budget_agent = BudgetAgent()
        self.scheduler_agent = SchedulerAgent()
        self.evaluation_agent = EvaluationAgent()
        self.places_agent = GooglePlacesAgent()
        
        self.state = PreferenceState()
    
    def log(self, step_name: str, message: str):
        """Log a message."""
        if self.logger:
            log_step(step_name, message)
        else:
            print(f"[{step_name.upper()}] {message}")
    
    def collect_basic_info(self) -> Dict:
        """Collect basic trip information."""
        print("\n" + "="*50)
        print("TRAVEL PLANNER - Initial Setup")
        print("="*50)
        
        try:
            budget = float(input(f"Enter total budget in EUR (default: {Config.DEFAULT_BUDGET}): ") 
                          or Config.DEFAULT_BUDGET)
            people = int(input(f"Enter number of people (default: {Config.DEFAULT_PEOPLE}): ") 
                        or Config.DEFAULT_PEOPLE)
            days = int(input(f"Enter number of days (default: {Config.DEFAULT_DAYS}): ") 
                      or Config.DEFAULT_DAYS)
            
            print(f"\nPlanning {days}-day trip for {people} people with {budget} EUR budget")
            return {"budget": budget, "people": people, "days": days}
            
        except ValueError as e:
            print(f"Invalid input: {e}. Using defaults.")
            return {
                "budget": Config.DEFAULT_BUDGET,
                "people": Config.DEFAULT_PEOPLE,
                "days": Config.DEFAULT_DAYS
            }
    
    def run_interest_dialogue(self, budget: float, people: int, days: int):
        """Run interest refinement dialogue."""
        self.log("INTEREST REFINEMENT", "Starting dialogue")
        
        print(f"\nPlanning a {days}-day trip for {people} people with {budget} EUR budget")
        initial_msg = input("Tell me about your trip preferences (interests, activities, etc.): ")
        
        # Initial update with user preferences
        initial_info = f"Budget is {budget} EUR for {people} people for {days} days. {initial_msg}"
        self.state = self.semantic_agent.update_state(self.state, initial_info)
        
        turn_count = 0
        max_turns = Config.MAX_DIALOGUE_TURNS
        
        while turn_count < max_turns:
            turn_count += 1
            self.log("DIALOGUE", f"Turn {turn_count}/{max_turns}")
            
            # Get agent response
            response = self.interest_agent.process_turn(
                self.state, initial_msg, budget, people, days
            )
            
            if response["action"] == "ask_question":
                question = response["question"]
                # Skip budget questions
                if any(word in question.lower() for word in ['budget', 'price', 'cost']):
                    print(f"\n🔄 Skipping budget question (already known: {budget} EUR)")
                    if turn_count >= 3:
                        print("Multiple budget questions detected, finalizing...")
                        response["action"] = "finalize"
                    else:
                        question = "What specific attractions or activities interest you most?"
                
                if response["action"] == "ask_question":
                    print(f"\nAgent: {question}")
                    user_response = input("You: ")
                    self.state = self.semantic_agent.update_state(self.state, user_response)
                    initial_msg = user_response
                    continue
            
            if response["action"] == "finalize":
                profile = self.interest_agent.create_final_profile(self.state, response)
                self.log("INTEREST REFINEMENT", "Profile finalized successfully")
                print(f"\n✅ Selected City: {profile.chosen_city}")
                return profile
        
        # Max turns reached - USE INTEREST AGENT TO GET CITY FROM LLM
        self.log("INTEREST REFINEMENT", f"Max turns ({max_turns}) reached")
        
        # Get city recommendation from the interest agent based on user preferences
        user_preferences = self.semantic_agent.build_profile_summary(self.state)
        
        # Let the interest agent handle the city recommendation
        profile = self.interest_agent.create_final_profile(self.state, {
            "refined_profile": user_preferences,
            "chosen_city": None,  # Let the agent decide
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": budget,
                "people": people
            },
            "travel_style": "medium"
        })
        
        print(f"\n✅ Selected City: {profile.chosen_city}")
        return profile
    
    def run_pipeline(self):
        """Execute complete planning pipeline."""
        self.log("MAIN PIPELINE", "Starting travel planning")
        
        # Validate configuration
        try:
            if not Config.validate_config():
                print("⚠️ Configuration warnings found. Please review.")
        except:
            pass  # Skip config validation if it fails
        
        # Step 1: Collect basic info
        basic_info = self.collect_basic_info()
        budget = basic_info["budget"]
        people = basic_info["people"]
        days = basic_info["days"]
        
        # Step 2: Interest refinement
        self.log("INTEREST REFINEMENT", "Starting dialogue")
        profile = self.run_interest_dialogue(budget, people, days)
        
        # Step 3: Generate attractions
        self.log("LOCATION SCOUT", f"Generating attractions for {profile.chosen_city}")
        attractions = self.location_agent.generate_attractions(
            profile.chosen_city, profile.refined_profile, profile.constraints.model_dump()
        )
        
        # REMOVED: Fallback attractions call (method doesn't exist anymore)
        if len(attractions) < 3:
            print(f"⚠️ Only {len(attractions)} attractions generated. Continuing with available attractions.")
        
        # Step 4: Enrich with Google Places
        if hasattr(self.places_agent, 'is_enabled') and self.places_agent.is_enabled():
            self.log("GOOGLE PLACES", f"Enriching {len(attractions)} attractions")
            attractions = self.places_agent.enrich_attractions(attractions, profile.chosen_city)
        else:
            self.log("GOOGLE PLACES", "API not enabled, skipping enrichment")
        
        # Step 5: Budget filtering
        self.log("BUDGET AGENT", f"Filtering attractions for {people} people, {budget} EUR")
        affordable = self.budget_agent.filter_by_budget(attractions, budget, days, people)
        
        if len(affordable) < 3:
            print(f"⚠️ Only {len(affordable)} affordable attractions. Using available attractions.")
            affordable = attractions[:min(len(attractions), days * 3)]
        
        # Step 6: Create itinerary
        self.log("SCHEDULER", f"Creating {days}-day itinerary")
        itinerary = self.scheduler_agent.create_itinerary(affordable, days)
        
        # Step 7: Evaluate itinerary
        self.log("EVALUATION", "Evaluating itinerary quality")
        evaluation = self.evaluation_agent.evaluate_itinerary(profile.model_dump(), itinerary.model_dump())
        
        # Display results
        self.display_results(profile, itinerary, evaluation, affordable)
        
        self.log("MAIN PIPELINE", "Planning completed successfully")
    
    def display_results(self, profile, itinerary, evaluation, attractions):
        """Display final results."""
        print("\n" + "="*60)
        print("FINAL TRAVEL PLAN")
        print("="*60)
        
        print(f"\n📍 Destination: {profile.chosen_city}")
        print(f"👥 People: {profile.constraints.people}")
        print(f"💰 Budget: {profile.constraints.budget} EUR")
        print(f"📅 Days: {len([d for d in itinerary.model_dump().keys() if d.startswith('day')])}")
        
        # Calculate budget summary
        total_cost = sum(attr.final_price_estimate or 0 for attr in attractions)
        print(f"💵 Estimated Cost: {total_cost:.2f} EUR")
        print(f"💶 Remaining Budget: {profile.constraints.budget - total_cost:.2f} EUR")
        
        print("\n📋 ITINERARY:")
        print(json.dumps(itinerary.model_dump(), indent=2, default=str))
        
        print("\n⭐ EVALUATION:")
        print(f"  Interest Match: {evaluation.interest_match}/5")
        print(f"  Budget Realism: {evaluation.budget_realism}/5")
        print(f"  Logistics: {evaluation.logistics}/5")
        print(f"  Suitability: {evaluation.suitability_for_constraints}/5")
        print(f"  Overall: {self.evaluation_agent.calculate_overall_score(evaluation):.1f}/5")
        print(f"  Comment: {evaluation.comment}")
        
        print("\n" + "="*60)
        print("🎉 Your travel plan is ready! Have a great trip!")
        print("="*60)

def main():
    """Main function."""
    try:
        planner = TravelPlanner()
        planner.run_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️ Planning cancelled by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()