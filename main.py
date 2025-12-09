#!/usr/bin/env python3
"""
Main entry point for Travel Itinerary Planner
"""

import json
from typing import Dict

from config import Config
from utils.logging_utils import setup_logging, log_step, Timer
from agents.semantic_agent import SemanticAgent
from agents.interest_refinement_agent import InterestRefinementAgent
from agents.location_scout_agent import LocationScoutAgent
from agents.budget_agent import BudgetAgent
from agents.scheduler_agent import SchedulerAgent
from agents.evaluation_agent import EvaluationAgent
from agents.google_places_agent import GooglePlacesAgent
from utils.data_structures import PreferenceState

class TravelPlanner:
    """Main orchestrator for travel planning pipeline."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.semantic_agent = SemanticAgent()
        self.interest_agent = InterestRefinementAgent()
        self.location_agent = LocationScoutAgent()
        self.budget_agent = BudgetAgent()
        self.scheduler_agent = SchedulerAgent()
        self.evaluation_agent = EvaluationAgent()
        self.places_agent = GooglePlacesAgent()
        
        self.state = PreferenceState()
    
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
        log_step("INTEREST REFINEMENT", "Starting dialogue")
        
        print(f"\nPlanning a {days}-day trip for {people} people with {budget} EUR budget")
        initial_msg = input("Tell me about your trip preferences (interests, activities, etc.): ")
        
        # Initial update with user preferences
        initial_info = f"Budget is {budget} EUR for {people} people for {days} days. {initial_msg}"
        self.state = self.semantic_agent.update_state(self.state, initial_info)
        
        turn_count = 0
        
        while turn_count < Config.MAX_DIALOGUE_TURNS:
            turn_count += 1
            log_step("DIALOGUE", f"Turn {turn_count}/{Config.MAX_DIALOGUE_TURNS}")
            
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
                log_step("INTEREST REFINEMENT", "Profile finalized successfully")
                print(f"\n✅ Selected City: {profile.chosen_city}")
                return profile
        
        # Max turns reached
        log_step("INTEREST REFINEMENT", f"Max turns ({Config.MAX_DIALOGUE_TURNS}) reached")
        return self.interest_agent.create_final_profile(self.state, {
            "refined_profile": self.semantic_agent.build_profile_summary(self.state),
            "chosen_city": "Rome",
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": budget,
                "people": people
            },
            "travel_style": "medium"
        })
    
    def run_pipeline(self):
        """Execute complete planning pipeline."""
        log_step("MAIN PIPELINE", "Starting travel planning")
        
        # Validate configuration
        if not Config.validate_config():
            print("⚠️ Configuration warnings found. Please review.")
        
        # Step 1: Collect basic info
        basic_info = self.collect_basic_info()
        budget = basic_info["budget"]
        people = basic_info["people"]
        days = basic_info["days"]
        
        # Step 2: Interest refinement
        with Timer("Interest Refinement"):
            profile = self.run_interest_dialogue(budget, people, days)
        
        # Step 3: Generate attractions
        log_step("LOCATION SCOUT", f"Generating attractions for {profile.chosen_city}")
        with Timer("Location Scout"):
            attractions = self.location_agent.generate_attractions(
                profile.chosen_city, profile.refined_profile, profile.constraints.dict()
            )
        
        if len(attractions) < 3:
            print("⚠️ Few attractions generated. Using fallback.")
            attractions = self.location_agent.get_fallback_attractions(profile.chosen_city)
        
        # Step 4: Enrich with Google Places
        if self.places_agent.is_enabled():
            log_step("GOOGLE PLACES", f"Enriching {len(attractions)} attractions")
            with Timer("Google Places Enrichment"):
                attractions = self.places_agent.enrich_attractions(attractions, profile.chosen_city)
        else:
            log_step("GOOGLE PLACES", "API not enabled, skipping enrichment")
        
        # Step 5: Budget filtering
        log_step("BUDGET AGENT", f"Filtering attractions for {people} people, {budget} EUR")
        with Timer("Budget Filtering"):
            affordable = self.budget_agent.filter_by_budget(attractions, budget, days, people)
        
        if len(affordable) < 3:
            print("⚠️ Few affordable attractions. Adjusting selection...")
            affordable = attractions[:days * 2]
        
        # Step 6: Create itinerary
        log_step("SCHEDULER", f"Creating {days}-day itinerary")
        with Timer("Scheduling"):
            itinerary = self.scheduler_agent.create_itinerary(affordable, days)
        
        # Step 7: Evaluate itinerary
        log_step("EVALUATION", "Evaluating itinerary quality")
        with Timer("Evaluation"):
            evaluation = self.evaluation_agent.evaluate_itinerary(profile.dict(), itinerary.dict())
        
        # Display results
        self.display_results(profile, itinerary, evaluation, affordable)
        
        log_step("MAIN PIPELINE", "Planning completed successfully")
    
    def display_results(self, profile, itinerary, evaluation, attractions):
        """Display final results."""
        print("\n" + "="*60)
        print("FINAL TRAVEL PLAN")
        print("="*60)
        
        print(f"\n📍 Destination: {profile.chosen_city}")
        print(f"👥 People: {profile.constraints.people}")
        print(f"💰 Budget: {profile.constraints.budget} EUR")
        print(f"📅 Days: {len([d for d in itinerary.dict().keys() if d.startswith('day')])}")
        
        # Calculate budget summary
        total_cost = sum(attr.final_price_estimate or 0 for attr in attractions)
        print(f"💵 Estimated Cost: {total_cost:.2f} EUR")
        print(f"💶 Remaining Budget: {profile.constraints.budget - total_cost:.2f} EUR")
        
        print("\n📋 ITINERARY:")
        print(json.dumps(itinerary.dict(), indent=2, default=str))
        
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