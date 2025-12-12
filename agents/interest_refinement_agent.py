
import json
from typing import Dict, Optional
from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import PreferenceState, TripConstraints, TravelProfile
from config import Config

class InterestRefinementAgent:
    """Agent for refining user interests and selecting destination city."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
        self.max_turns = Config.MAX_DIALOGUE_TURNS
    
    def generate_dialogue_prompt(self, state: PreferenceState, last_user_msg: str,
                                budget: float, people: int, days: int) -> str:
        """Generate prompt for dialogue turn."""
        
        prompt = f"""You are the "Interest-Refinement & City-Matchmaker Agent" in a travel planning system.

We are interviewing a user to plan a {days}-day city trip in Europe for {people} people with a total budget of {budget} EUR.

CRITICAL INFORMATION ALREADY COLLECTED:
- Total budget: {budget} EUR (already confirmed)
- Number of people: {people} (already confirmed) 
- Trip duration: {days} days (already confirmed)

CURRENT SEMANTIC PROFILE (built from all previous messages):
{self._build_profile_summary(state)}

Most recent user message:
\"\"\"{last_user_msg}\"\"\"

Your tasks:
1. DO NOT ask about budget, number of people, or trip duration - these are already confirmed.
2. Focus on understanding interests, preferences, pace, food, and constraints.
3. If we have enough information, recommend ONE European city and finalize.
4. If not, ask ONE clarifying question.

Return ONLY valid JSON with this structure:
{{
  "action": "ask_question" or "finalize",
  "question": "string (if action == 'ask_question', else empty)",
  "refined_profile": "short natural language summary",
  "chosen_city": "string or null",
  "constraints": {{
    "with_children": true/false/null,
    "with_disabled": true/false/null,
    "budget": {budget},
    "people": {people}
  }},
  "travel_style": "slow"/"medium"/"fast"/null
}}

IMPORTANT: 
- DO NOT ask about budget, people count, or trip duration
- Maximum {self.max_turns} questions total - be efficient!
- Return ONLY the JSON object, no additional text"""
        
        return prompt
    
    def _build_profile_summary(self, state: PreferenceState) -> str:
        """Helper to build profile summary."""
        summary = []
        for key, label in [
            ("activities", "Activities"),
            ("pace", "Pace"),
            ("food", "Food"),
            ("constraints", "Constraints"),
            ("budget", "Budget")
        ]:
            value = state.slots.get(key, "") or "not specified yet"
            summary.append(f"{label}: {value}")
        return "\n".join(summary)
    
    def process_turn(self, state: PreferenceState, last_user_msg: str,
                    budget: float, people: int, days: int) -> Dict:
        """Process a dialogue turn."""
        prompt = self.generate_dialogue_prompt(state, last_user_msg, budget, people, days)
        
        try:
            raw_response = self.llm_client.generate(prompt, temperature=0.7)
            response_data = self.json_parser.parse_response(raw_response)
            
            # Validate and add defaults
            response_data = self._validate_response(response_data, budget, people)
            return response_data
            
        except Exception as e:
            print(f"Error in dialogue turn: {e}")
            return self._create_fallback_response(state, budget, people)
    
    def _validate_response(self, data: Dict, budget: float, people: int) -> Dict:
        """Validate and add default values to response."""
        # Ensure action is valid
        if data.get("action") not in ["ask_question", "finalize"]:
            data["action"] = "ask_question"
        
        # Ensure constraints exist
        if not isinstance(data.get("constraints"), dict):
            data["constraints"] = {}
        
        # Set required constraint fields
        constraints = data["constraints"]
        constraints.setdefault("with_children", None)
        constraints.setdefault("with_disabled", None)
        constraints.setdefault("budget", budget)
        constraints.setdefault("people", people)
        
        # Set other defaults
        data.setdefault("question", "")
        data.setdefault("refined_profile", "User preferences not fully specified")
        data.setdefault("chosen_city", None)
        data.setdefault("travel_style", None)
        
        return data
    
    def _create_fallback_response(self, state: PreferenceState, 
                                 budget: float, people: int) -> Dict:
        """Create fallback response when LLM fails."""
        return {
            "action": "ask_question",
            "question": "What type of activities or attractions interest you most?",
            "refined_profile": "Preferences still being refined",
            "chosen_city": None,
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": budget,
                "people": people
            },
            "travel_style": "medium"
        }
    
    def create_final_profile(self, state: PreferenceState, llm_output: Dict) -> TravelProfile:
        """Create final travel profile from dialogue results."""
        return TravelProfile(
            refined_profile=llm_output["refined_profile"],
            chosen_city=llm_output["chosen_city"] or "Rome",
            constraints=TripConstraints(**llm_output["constraints"]),
            travel_style=llm_output["travel_style"],
            semantic_profile_slots=state.slots,
            interest_embedding=(
                state.global_embedding.tolist() 
                if state.global_embedding is not None 
                else None
            )
        )
