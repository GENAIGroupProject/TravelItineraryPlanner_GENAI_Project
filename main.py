#!/usr/bin/env python3
"""
Main entry point for Travel Itinerary Planner - User-Friendly Version
"""

import sys
import os
import time
import json
from typing import Dict

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    
    # Import logging utilities
    try:
        from utils.logging_utils import (
            setup_logging, log_step, Timer, 
            ConsoleFormatter, LoadingSpinner, 
            log_to_file_only, clear_screen,
            log_performance_with_threshold
        )
        LOGGING_AVAILABLE = True
        print("✅ Logging utilities loaded successfully")
    except ImportError as e:
        print(f"Logging utils import error: {e}")
        print("Creating simple fallback formatters...")
        LOGGING_AVAILABLE = False
        
        # Simple fallback formatters
        class SimpleConsoleFormatter:
            @staticmethod
            def loading(msg): return f"Processing {msg}..."
            @staticmethod
            def success(msg): return f"✓ {msg}"
            @staticmethod
            def info(msg): return f"ℹ️  {msg}"
            @staticmethod
            def warning(msg): return f"⚠️  {msg}"
            @staticmethod
            def error(msg): return f"✗ {msg}"
            @staticmethod
            def question(msg): return f"? {msg}"
            @staticmethod
            def step(msg): return f"→ {msg}"
            @staticmethod
            def travel(msg): return f"✈️ {msg}"
        
        ConsoleFormatter = SimpleConsoleFormatter
        
        class SimpleLoadingSpinner:
            def __init__(self, message="Loading"):
                self.message = message
            def __enter__(self):
                print(f"⌛ {self.message}...", end="", flush=True)
                return self
            def __exit__(self, *args):
                print(" Done!")
        
        LoadingSpinner = SimpleLoadingSpinner
        
        def log_to_file_only(*args, **kwargs):
            pass  # Do nothing in fallback mode
        
        def setup_logging():
            print("Simple logging setup")
            return None
    
    # Import agents
    try:
        from agents.semantic_agent import SemanticAgent
        from agents.interest_refinement_agent import InterestRefinementAgent
        from agents.location_scout_agent import LocationScoutAgent
        from agents.budget_agent import BudgetAgent
        from agents.scheduler_agent import SchedulerAgent
        from agents.evaluation_agent import EvaluationAgent
        from agents.google_places_agent import GooglePlacesAgent
        from utils.data_structures import PreferenceState, TripConstraints, TravelProfile
        AGENTS_AVAILABLE = True
        print("✅ All agents loaded successfully")
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
    """Main orchestrator for travel planning pipeline - User Friendly Version."""
    
    def __init__(self):
        if LOGGING_AVAILABLE:
            self.logger = setup_logging()
            if self.logger:
                print("✅ Logger initialized")
            else:
                print("⚠️ Logger setup returned None")
        else:
            self.logger = None
        
        # Initialize agents
        print(ConsoleFormatter.step("Initializing travel planning system..."))
        
        with LoadingSpinner("Starting agents"):
            self.semantic_agent = SemanticAgent()
            self.interest_agent = InterestRefinementAgent()
            self.location_agent = LocationScoutAgent()
            self.budget_agent = BudgetAgent()
            self.scheduler_agent = SchedulerAgent()
            self.evaluation_agent = EvaluationAgent()
            self.places_agent = GooglePlacesAgent()
            
            self.state = PreferenceState()
        
        print(ConsoleFormatter.success("System ready!"))
        time.sleep(0.5)
    
    def show_header(self):
        """Display application header."""
        try:
            if hasattr(ConsoleFormatter, 'clear_screen'):
                ConsoleFormatter.clear_screen()
            elif LOGGING_AVAILABLE:
                clear_screen()
        except:
            print("\n" * 50)  # Fallback to newlines
        
        print("=" * 60)
        print("           🌍 TRAVEL ITINERARY PLANNER 🌍")
        print("=" * 60)
        print()
    
    def log_to_file(self, category: str, message: str):
        """Log message to file only."""
        try:
            log_to_file_only(f"[{category}] {message}", "info")
        except:
            # Fallback logging
            with open("travel_planner.log", "a", encoding="utf-8") as f:
                f.write(f"[{category}] {message}\n")
    
    def collect_basic_info(self) -> Dict:
        """Collect basic trip information including constraints."""
        self.show_header()
        print(ConsoleFormatter.travel("Let's plan your dream trip! 🎉"))
        print()
        
        try:
            print("📅 Trip Details")
            print("─" * 40)
            
            budget_input = input(f"   Total budget in EUR (press Enter for {Config.DEFAULT_BUDGET}€): ")
            budget = float(budget_input) if budget_input.strip() else Config.DEFAULT_BUDGET
            
            people_input = input(f"   Number of travelers (press Enter for {Config.DEFAULT_PEOPLE}): ")
            people = int(people_input) if people_input.strip() else Config.DEFAULT_PEOPLE
            
            days_input = input(f"   Trip duration in days (press Enter for {Config.DEFAULT_DAYS}): ")
            days = int(days_input) if days_input.strip() else Config.DEFAULT_DAYS
            
            print()
            print("👥 Traveler Constraints")
            print("─" * 40)
            
            # Ask about children
            children_input = input("   Will there be children traveling? (yes/no, press Enter for no): ")
            with_children = children_input.lower().strip() in ['yes', 'y', '1', 'true']
            
            # Ask about disabilities
            disability_input = input("   Any travelers with mobility/disability needs? (yes/no, press Enter for no): ")
            with_disabled = disability_input.lower().strip() in ['yes', 'y', '1', 'true']
            
            print()
            print(ConsoleFormatter.success(f"Planning {days}-day trip for {people} traveler(s)"))
            print(ConsoleFormatter.success(f"Budget: {budget}€"))
            if with_children:
                print(ConsoleFormatter.info("✓ Traveling with children"))
            if with_disabled:
                print(ConsoleFormatter.info("✓ Accessibility needs considered"))
            
            # Log to file only
            self.log_to_file("USER_INPUT", f"Budget: {budget}, People: {people}, Days: {days}")
            self.log_to_file("USER_INPUT", f"With children: {with_children}")
            self.log_to_file("USER_INPUT", f"With disabled: {with_disabled}")
            
            return {
                "budget": budget, 
                "people": people, 
                "days": days,
                "with_children": with_children,
                "with_disabled": with_disabled
            }
            
        except ValueError as e:
            print(ConsoleFormatter.error(f"Invalid input: {e}"))
            print(ConsoleFormatter.info("Using default values"))
            return {
                "budget": Config.DEFAULT_BUDGET,
                "people": Config.DEFAULT_PEOPLE,
                "days": Config.DEFAULT_DAYS,
                "with_children": False,
                "with_disabled": False
            }
    
    def run_interest_dialogue(self, basic_info: Dict):
        """Run interest refinement dialogue with better UX."""
        budget = basic_info["budget"]
        people = basic_info["people"]
        days = basic_info["days"]
        with_children = basic_info["with_children"]
        with_disabled = basic_info["with_disabled"]
        
        self.show_header()
        print(ConsoleFormatter.travel(f"Planning your {days}-day adventure! 🌟"))
        print()
        print(f"💰 Budget: {budget}€")
        print(f"👥 Travelers: {people}")
        if with_children:
            print(f"👶 Children: Yes")
        if with_disabled:
            print(f"♿ Accessibility needs: Yes")
        print()
        print("=" * 60)
        print()
        
        # Get initial preferences
        print(ConsoleFormatter.question("What kind of trip are you looking for?"))
        print("Examples: 'I love hiking and nature', 'Museums and historical sites',")
        print("'Relaxing beach vacation', 'City exploration and food'")
        print()
        initial_msg = input("Your preferences: ")
        
        with LoadingSpinner("Analyzing your preferences"):
            # Process initial input
            initial_info = f"Budget is {budget} EUR for {people} people for {days} days. {initial_msg}"
            self.state = self.semantic_agent.update_state(self.state, initial_info)
            time.sleep(1)  # Simulate processing
        
        turn_count = 0
        max_turns = min(getattr(Config, 'MAX_DIALOGUE_TURNS', 5), 3)  # Shorter dialogues
        
        while turn_count < max_turns:
            turn_count += 1
            
            print()
            print(ConsoleFormatter.step(f"Refining your preferences ({turn_count}/{max_turns})"))
            
            with LoadingSpinner("Thinking of the perfect destination"):
                # Get agent response
                response = self.interest_agent.process_turn(
                    self.state, initial_msg, budget, people, days
                )
            
            if response and response.get("action") == "ask_question":
                question = response.get("question", "")
                if not question:
                    continue
                    
                # Skip budget questions
                if any(word in question.lower() for word in ['budget', 'price', 'cost']):
                    print(ConsoleFormatter.info(f"Skipping budget question (already set to {budget}€)"))
                    continue
                
                print()
                print(ConsoleFormatter.question(question))
                print("(You can type 'done' if you're ready to finalize)")
                print()
                user_response = input("Your answer: ")
                
                if user_response.lower() in ['done', 'ready', 'finalize', 'that\'s all', '']:
                    break
                
                with LoadingSpinner("Updating your preferences"):
                    self.state = self.semantic_agent.update_state(self.state, user_response)
                    initial_msg = user_response
            
            elif response and response.get("action") == "finalize":
                with LoadingSpinner("Creating your travel profile"):
                    # Add constraints to response
                    response["constraints"]["with_children"] = with_children
                    response["constraints"]["with_disabled"] = with_disabled
                    profile = self.interest_agent.create_final_profile(self.state, response)
                
                print()
                print(ConsoleFormatter.success("✓ Destination found!"))
                print("=" * 40)
                print(f"   🌍 {profile.chosen_city}")
                print("=" * 40)
                
                # Log profile details to file
                self.log_to_file("PROFILE", f"City: {profile.chosen_city}")
                self.log_to_file("PROFILE", f"Profile: {profile.refined_profile}")
                self.log_to_file("PROFILE", f"Constraints: Children={with_children}, Disabled={with_disabled}")
                
                return profile
        
        # If we exit the loop without finalizing
        print()
        print(ConsoleFormatter.info("Finalizing with your current preferences..."))
        
        with LoadingSpinner("Selecting the perfect destination"):
            # Get city recommendation
            try:
                user_preferences = self.semantic_agent.build_profile_summary(self.state)
            except:
                user_preferences = "General travel preferences"
            
            # Let the interest agent handle the city recommendation
            profile = self.interest_agent.create_final_profile(self.state, {
                "refined_profile": user_preferences,
                "chosen_city": None,
                "constraints": {
                    "with_children": with_children,
                    "with_disabled": with_disabled,
                    "budget": budget,
                    "people": people
                },
                "travel_style": "medium"
            })
        
        print()
        print(ConsoleFormatter.success("✓ Destination selected!"))
        print("=" * 40)
        print(f"   🌍 {profile.chosen_city}")
        print("=" * 40)
        
        return profile
    
    def run_pipeline(self):
        """Execute complete planning pipeline with improved UX."""
        self.show_header()
        
        # Welcome message
        print(ConsoleFormatter.travel("Welcome to the Travel Itinerary Planner! ✈️"))
        print()
        print("I'll help you plan your perfect trip step by step.")
        print("Let's start with some basic information.")
        print()
        input("Press Enter to continue...")
        
        # Step 1: Collect basic info
        basic_info = self.collect_basic_info()
        budget = basic_info["budget"]
        people = basic_info["people"]
        days = basic_info["days"]
        with_children = basic_info["with_children"]
        with_disabled = basic_info["with_disabled"]
        
        # Step 2: Interest refinement
        print()
        input("Press Enter to tell me about your travel preferences...")
        profile = self.run_interest_dialogue(basic_info)
        
        if not profile:
            print(ConsoleFormatter.error("Failed to create profile. Please try again."))
            return
        
        # Step 3: Generate attractions
        print()
        print(ConsoleFormatter.step(f"Finding amazing places in {profile.chosen_city}..."))
        
        with LoadingSpinner("Discovering attractions"):
            attractions = self.location_agent.generate_attractions(
                profile.chosen_city, profile.refined_profile, profile.constraints.model_dump()
            )
        
        if attractions:
            print(ConsoleFormatter.success(f"Found {len(attractions)} potential attractions"))
        else:
            print(ConsoleFormatter.warning("No attractions found. Using fallback..."))
            # Create fallback attractions
            attractions = []
        
        if len(attractions) < 3:
            print(ConsoleFormatter.warning(f"Only {len(attractions)} attractions found. Continuing..."))
        
        # Step 4: Enrich with Google Places
        if hasattr(self.places_agent, 'is_enabled') and self.places_agent.is_enabled():
            print()
            print(ConsoleFormatter.step("Getting current information about attractions..."))
            
            with LoadingSpinner("Contacting Google Places"):
                attractions = self.places_agent.enrich_attractions(attractions, profile.chosen_city)
            
            print(ConsoleFormatter.success("Attraction information updated"))
        else:
            self.log_to_file("INFO", "Google Places API not enabled, skipping enrichment")
        
        # Step 5: Budget filtering
        print()
        print(ConsoleFormatter.step("Checking what fits your budget..."))
        
        with LoadingSpinner("Calculating costs"):
            affordable = self.budget_agent.filter_by_budget(attractions, budget, days, people)
        
        print(ConsoleFormatter.success(f"{len(affordable)} attractions fit your budget"))
        
        if len(affordable) < 3:
            print(ConsoleFormatter.warning("Limited attractions within budget. Showing all available."))
            affordable = attractions[:min(len(attractions), days * 3)]
        
        # Step 6: Create itinerary
        print()
        print(ConsoleFormatter.step("Creating your daily schedule..."))
        
        with LoadingSpinner("Optimizing your itinerary"):
            itinerary = self.scheduler_agent.create_itinerary(affordable, days)
        
        print(ConsoleFormatter.success("Daily itinerary created"))
        
        # Step 7: Evaluate itinerary
        print()
        print(ConsoleFormatter.step("Evaluating your travel plan..."))
        
        with LoadingSpinner("Checking quality and feasibility"):
            evaluation = self.evaluation_agent.evaluate_itinerary(
                profile.model_dump(), itinerary.model_dump()
            )
        
        print(ConsoleFormatter.success("Evaluation complete"))
        
        # Display results
        self.display_results(profile, itinerary, evaluation, affordable)
        
        print()
        print(ConsoleFormatter.success("Your travel plan is ready!"))
        self.log_to_file("COMPLETION", "Planning completed successfully")
    
    def _format_opening_hours(self, opening_hours: Dict) -> str:
        """Format opening hours from Google Places API."""
        if not opening_hours:
            return "Opening hours not available"
        
        try:
            if opening_hours.get("open_now") is not None:
                open_now = "Open now" if opening_hours["open_now"] else "Closed now"
            else:
                open_now = ""
            
            periods = opening_hours.get("periods", [])
            if periods:
                today_schedule = []
                for period in periods[:2]:  # Show first two periods
                    open_time = period.get("open", {}).get("time", "")
                    close_time = period.get("close", {}).get("time", "")
                    if open_time and close_time:
                        # Format time from 0900 to 09:00
                        open_formatted = f"{open_time[:2]}:{open_time[2:]}"
                        close_formatted = f"{close_time[:2]}:{close_time[2:]}"
                        today_schedule.append(f"{open_formatted}-{close_formatted}")
                
                if today_schedule:
                    return f"{open_now} ({', '.join(today_schedule)})"
            
            return open_now if open_now else "Opening hours not available"
        except:
            return "Opening hours available"
    
    def display_results(self, profile, itinerary, evaluation, attractions):
        """Display final results in a user-friendly format with Google Places info."""
        self.show_header()
        
        print("=" * 60)
        print("           ✨ YOUR TRAVEL PLAN IS READY! ✨")
        print("=" * 60)
        print()
        
        # Trip summary
        print(ConsoleFormatter.travel("TRIP SUMMARY"))
        print("─" * 40)
        print(f"📍 Destination: {profile.chosen_city}")
        print(f"👥 Travelers: {profile.constraints.people}")
        if profile.constraints.with_children:
            print(f"👶 Includes children")
        if profile.constraints.with_disabled:
            print(f"♿ Accessibility considered")
        
        # Count days with content
        itinerary_dict = itinerary.model_dump() if hasattr(itinerary, 'model_dump') else itinerary
        days_with_content = 0
        for day_key in ['day1', 'day2', 'day3']:
            day_data = itinerary_dict.get(day_key, {})
            if any(len(day_data.get(slot, [])) > 0 for slot in ['morning', 'afternoon', 'evening']):
                days_with_content += 1
        
        print(f"📅 Duration: {days_with_content} days")
        print(f"💰 Budget: {profile.constraints.budget}€")
        
        # Calculate costs
        total_cost = sum((attr.final_price_estimate or 0) for attr in attractions)
        remaining = profile.constraints.budget - total_cost
        
        print(f"💵 Estimated cost: {total_cost:.2f}€")
        print(f"💶 Remaining budget: {remaining:.2f}€")
        
        if remaining > profile.constraints.budget * 0.2:
            print(ConsoleFormatter.success("Great budget management! 🎉"))
        elif remaining < 0:
            print(ConsoleFormatter.warning("Note: Slightly over budget"))
        
        print()
        
        # Display itinerary with Google Places info
        print(ConsoleFormatter.travel("DAILY ITINERARY"))
        print("─" * 40)
        
        for day_key in ['day1', 'day2', 'day3']:
            day_data = itinerary_dict.get(day_key, {})
            if any(len(day_data.get(slot, [])) > 0 for slot in ['morning', 'afternoon', 'evening']):
                print(f"\n📅 {day_key.upper().replace('DAY', 'DAY ')}")
                print("  " + "─" * 30)
                
                for time_slot in ['morning', 'afternoon', 'evening']:
                    slot_attractions = day_data.get(time_slot, [])
                    if slot_attractions:
                        print(f"  ⏰ {time_slot.capitalize()}:")
                        for attr in slot_attractions:
                            # Handle both object and dict
                            if hasattr(attr, 'get'):
                                name = attr.get('name', 'Unknown')
                                cost = attr.get('final_price_estimate', 0) or 0
                                rating = attr.get('google_rating')
                                opening_hours = attr.get('opening_hours')
                                place_id = attr.get('google_place_id')
                                tags = attr.get('tags', [])
                            else:
                                name = getattr(attr, 'name', 'Unknown')
                                cost = getattr(attr, 'final_price_estimate', 0) or 0
                                rating = getattr(attr, 'google_rating', None)
                                opening_hours = getattr(attr, 'opening_hours', None)
                                place_id = getattr(attr, 'google_place_id', None)
                                tags = getattr(attr, 'tags', [])
                            
                            # Display basic info
                            if profile.constraints.people > 1:
                                cost_per_person = cost / profile.constraints.people
                                cost_str = f"({cost_per_person:.1f}€ per person)"
                            else:
                                cost_str = f"({cost:.1f}€)"
                            
                            print(f"    • {name} {cost_str}")
                            
                            # Display Google Places info if available
                            if rating:
                                print(f"      ⭐ Rating: {rating}/5")
                            
                            if opening_hours:
                                hours_str = self._format_opening_hours(opening_hours)
                                print(f"      🕒 {hours_str}")
                            
                            if place_id:
                                print(f"      📍 Google Places ID: {place_id[:20]}...")
                            
                            if tags:
                                print(f"      🏷️  Tags: {', '.join(tags[:3])}")
                            
                            print()  # Blank line between attractions
        
        print()
        
        # Evaluation
        print(ConsoleFormatter.travel("PLAN EVALUATION"))
        print("─" * 40)
        
        overall_score = self.evaluation_agent.calculate_overall_score(evaluation)
        
        # Convert score to stars
        stars = "⭐" * int(overall_score)
        if overall_score - int(overall_score) >= 0.5:
            stars += "½"
        
        print(f"Overall rating: {stars} ({overall_score:.1f}/5)")
        print()
        
        score_details = [
            ("Interest match", evaluation.interest_match),
            ("Budget realism", evaluation.budget_realism),
            ("Schedule flow", evaluation.logistics),
            ("Suitability", evaluation.suitability_for_constraints)
        ]
        
        for name, score in score_details:
            bar = "█" * score + "░" * (5 - score)
            print(f"  {name:20} {bar} {score}/5")
        
        print()
        print("💬 Feedback:", evaluation.comment)
        
        print()
        print("=" * 60)
        print(ConsoleFormatter.success("🎉 HAVE AN AMAZING TRIP! 🎉"))
        print("=" * 60)
        
        # Save full details to log file
        self.log_to_file("FINAL_PLAN", f"Destination: {profile.chosen_city}")
        self.log_to_file("FINAL_PLAN", f"Total cost: {total_cost:.2f}€")
        self.log_to_file("FINAL_PLAN", f"Evaluation score: {overall_score:.1f}/5")
        self.log_to_file("FINAL_PLAN", f"Evaluation comment: {evaluation.comment}")
        
        # Save itinerary to JSON file
        self.save_itinerary_to_file(profile, itinerary, evaluation, total_cost, attractions)
    
    def save_itinerary_to_file(self, profile, itinerary, evaluation, total_cost, attractions):
        """Save the complete itinerary to a JSON file."""
        import json
        from datetime import datetime
        
        try:
            filename = f"itinerary_{profile.chosen_city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to dicts if they're objects
            if hasattr(profile, 'model_dump'):
                profile_dict = profile.model_dump()
            else:
                profile_dict = profile
                
            if hasattr(itinerary, 'model_dump'):
                itinerary_dict = itinerary.model_dump()
            else:
                itinerary_dict = itinerary
                
            if hasattr(evaluation, 'model_dump'):
                evaluation_dict = evaluation.model_dump()
            else:
                evaluation_dict = evaluation
            
            # Prepare attractions data
            attractions_data = []
            for attr in attractions:
                if hasattr(attr, 'model_dump'):
                    attr_dict = attr.model_dump()
                else:
                    attr_dict = {
                        'name': getattr(attr, 'name', 'Unknown'),
                        'description': getattr(attr, 'short_description', ''),
                        'price': getattr(attr, 'final_price_estimate', 0),
                        'rating': getattr(attr, 'google_rating', None),
                        'opening_hours': getattr(attr, 'opening_hours', None),
                        'place_id': getattr(attr, 'google_place_id', None),
                        'tags': getattr(attr, 'tags', [])
                    }
                attractions_data.append(attr_dict)
            
            data = {
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "destination": profile.chosen_city,
                    "travelers": profile.constraints.people if hasattr(profile.constraints, 'people') else 1,
                    "days": 3,
                    "constraints": {
                        "with_children": getattr(profile.constraints, 'with_children', False),
                        "with_disabled": getattr(profile.constraints, 'with_disabled', False)
                    },
                    "budget": {
                        "total": profile.constraints.budget if hasattr(profile.constraints, 'budget') else 0,
                        "estimated_cost": total_cost,
                        "remaining": (profile.constraints.budget if hasattr(profile.constraints, 'budget') else 0) - total_cost
                    }
                },
                "profile": profile_dict,
                "attractions": attractions_data,
                "itinerary": itinerary_dict,
                "evaluation": {
                    "scores": evaluation_dict,
                    "overall": self.evaluation_agent.calculate_overall_score(evaluation)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print()
            print(ConsoleFormatter.info(f"Itinerary saved to: {filename}"))
        except Exception as e:
            print(f"⚠️ Failed to save itinerary: {e}")
            self.log_to_file("ERROR", f"Failed to save itinerary: {e}")

def main():
    """Main function with better error handling."""
    try:
        planner = TravelPlanner()
        planner.run_pipeline()
        
        # Ask if user wants to save or plan another trip
        print()
        print("=" * 60)
        print()
        print("What would you like to do next?")
        print("1. Plan another trip")
        print("2. Exit")
        print()
        
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                print()
                print(ConsoleFormatter.info("Starting new planning session..."))
                time.sleep(1)
                main()  # Restart
        except KeyboardInterrupt:
            pass
        
    except KeyboardInterrupt:
        print("\n\n" + ConsoleFormatter.info("Planning cancelled by user."))
    except Exception as e:
        print("\n" + ConsoleFormatter.error(f"An error occurred: {e}"))
        
        # Try to log the error
        try:
            with open("error.log", "a") as f:
                from datetime import datetime
                f.write(f"\n{datetime.now().isoformat()}\n")
                import traceback
                traceback.print_exc(file=f)
            print(ConsoleFormatter.info("Error details saved to error.log"))
        except:
            pass

if __name__ == "__main__":
    main()