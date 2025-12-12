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
        
        # Extract key user preferences
        user_prefs = self._extract_user_preferences(state, last_user_msg)
        
        prompt = f"""You are the "Interest-Refinement & City-Matchmaker Agent" in a travel planning system.

We are interviewing a user to plan a {days}-day city trip for {people} people with a total budget of {budget} EUR.

CRITICAL INFORMATION ALREADY COLLECTED:
- Total budget: {budget} EUR (already confirmed)
- Number of people: {people} (already confirmed) 
- Trip duration: {days} days (already confirmed)

USER PREFERENCES (so far):
{user_prefs}

Most recent user message:
\"\"\"{last_user_msg}\"\"\"

Your tasks:
1. DO NOT ask about budget, number of people, or trip duration - these are already confirmed.
2. Focus on understanding interests, preferences, pace, food, and constraints.
3. If we have enough information, recommend ONE city that matches user preferences.
4. If not, ask ONE clarifying question.

IMPORTANT GUIDELINES:
- If user mentions hiking, forests, parks, nature - recommend a city with good outdoor activities
- If user says NO to cultural/historical - avoid recommending cultural cities
- Consider relaxed vs fast pace preferences
- The city must match user's actual interests

Return ONLY valid JSON with this structure:
{{
  "action": "ask_question" or "finalize",
  "question": "string (if action == 'ask_question', else empty)",
  "refined_profile": "short natural language summary",
  "chosen_city": "string (must be a real city name) or null",
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
    
    def _extract_user_preferences(self, state: PreferenceState, last_user_msg: str) -> str:
        """Extract and format user preferences."""
        preferences = []
        
        # Check for key preferences in slots
        for key, label in [
            ("activities", "Activities"),
            ("pace", "Pace"),
            ("food", "Food"),
            ("constraints", "Constraints"),
            ("interests", "Interests")
        ]:
            value = state.slots.get(key, "") or "not specified yet"
            if value and value != "not specified yet":
                preferences.append(f"{label}: {value}")
        
        # Also check recent message for key terms
        lower_msg = last_user_msg.lower()
        if "hiking" in lower_msg or "forest" in lower_msg or "park" in lower_msg:
            preferences.append("Nature activities: Yes (hiking/forests/parks mentioned)")
        if "culture" in lower_msg or "museum" in lower_msg or "historical" in lower_msg:
            if "not" in lower_msg or "no" in lower_msg:
                preferences.append("Cultural interests: No (user said not interested)")
            else:
                preferences.append("Cultural interests: Yes")
        if "relaxed" in lower_msg or "slow" in lower_msg:
            preferences.append("Pace preference: Relaxed/slow")
        if "fast" in lower_msg or "busy" in lower_msg:
            preferences.append("Pace preference: Fast/busy")
        
        if not preferences:
            return "User hasn't specified preferences yet"
        
        return "\n".join(preferences)
    
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
            print(f"🔍 Raw LLM response for city recommendation:\n{raw_response}\n")
            response_data = self.json_parser.parse_response(raw_response)
            
            # Validate and add defaults
            response_data = self._validate_response(response_data, budget, people)
            
            # If finalizing, ensure city recommendation
            if response_data["action"] == "finalize" and not response_data["chosen_city"]:
                # Get city recommendation based on preferences
                response_data["chosen_city"] = self._get_city_recommendation(
                    state, last_user_msg, budget, days
                )
            
            return response_data
            
        except Exception as e:
            print(f"Error in dialogue turn: {e}")
            return self._create_fallback_response(state, budget, people)
    
    def _get_city_recommendation(self, state: PreferenceState, last_user_msg: str,
                                budget: float, days: int) -> str:
        """Use LLM to recommend a city based on user preferences."""
        user_prefs = self._extract_user_preferences(state, last_user_msg)
        
        prompt = f"""Based on user preferences, recommend ONE specific city:

USER PREFERENCES:
{user_prefs}

TRIP DETAILS:
- Budget: {budget} EUR for {days} days
- Must match user's stated preferences

IMPORTANT: If user mentioned hiking, forests, parks, or nature - recommend a city with good outdoor activities.
If user said NO to cultural/historical - avoid cultural cities.

Return ONLY the city name, nothing else.
Example: "Interlaken" or "Salzburg" or "Vancouver"
"""
        
        try:
            response = self.llm_client.generate(prompt, temperature=0.7)
            city = response.strip()
            
            # Clean response
            city = city.replace('"', '').replace("'", "").split('\n')[0].split(',')[0].strip()
            
            if 2 < len(city) < 50 and city[0].isalpha():
                print(f"🌍 LLM Recommended City: {city}")
                return city
            else:
                return self._get_fallback_city(user_prefs)
                
        except Exception as e:
            print(f"❌ Error getting city from LLM: {e}")
            return self._get_fallback_city(user_prefs)
    
    def _get_fallback_city(self, user_prefs: str) -> str:
        """Get fallback city based on preferences."""
        if "hiking" in user_prefs.lower() or "forest" in user_prefs.lower():
            return "Interlaken"  # Good for hiking/nature
        elif "beach" in user_prefs.lower() or "coast" in user_prefs.lower():
            return "Barcelona"
        else:
            # Let the LLM try one more time with a simpler prompt
            try:
                simple_prompt = "Recommend one city for a relaxed trip. City name only."
                response = self.llm_client.generate(simple_prompt, temperature=0.5)
                return response.strip().split('\n')[0]
            except:
                return "Paris"  # Last resort fallback
    
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
        # If no city chosen, get one from LLM
        chosen_city = llm_output.get("chosen_city")
        if not chosen_city:
            chosen_city = self._get_city_recommendation(
                state, 
                state.slots.get("last_message", ""), 
                llm_output["constraints"]["budget"],
                3  # default days
            )
        
        return TravelProfile(
            refined_profile=llm_output["refined_profile"],
            chosen_city=chosen_city,  # NO MORE "Rome" hardcode!
            constraints=TripConstraints(**llm_output["constraints"]),
            travel_style=llm_output["travel_style"],
            semantic_profile_slots=state.slots,
            interest_embedding=(
                state.global_embedding.tolist() 
                if state.global_embedding is not None 
                else None
            )
        )