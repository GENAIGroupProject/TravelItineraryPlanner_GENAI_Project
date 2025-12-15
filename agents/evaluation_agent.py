import json
from typing import Dict
from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import EvaluationScores
from utils.logging_utils import log_step, log_agent_communication

class EvaluationAgent:
    """Agent for evaluating itinerary quality."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
        log_step("EVALUATION", "Evaluation agent initialized")
    
    def evaluate_itinerary(self, profile: Dict, itinerary: Dict) -> EvaluationScores:
        """Evaluate an itinerary against user profile."""
        log_step("EVALUATION", "Starting itinerary evaluation")
        log_agent_communication(
            from_agent="EvaluationAgent",
            to_agent="LLM",
            message_type="evaluation_request",
            data={
                "profile_city": profile.get("chosen_city", "Unknown"),
                "itinerary_days": 3,  # Assuming 3 days
                "has_itinerary_data": bool(itinerary)
            }
        )
        
        prompt = self._create_evaluation_prompt(profile, itinerary)
        
        try:
            log_step("EVALUATION", "Generating evaluation prompt")
            raw_response = self.llm_client.generate(prompt, temperature=0.3)
            
            log_step("EVALUATION", f"Received evaluation response ({len(raw_response)} chars)")
            log_agent_communication(
                from_agent="LLM",
                to_agent="EvaluationAgent",
                message_type="evaluation_response",
                data={
                    "response_length": len(raw_response),
                    "preview": raw_response[:200]
                }
            )
            
            response_data = self.json_parser.parse_response(raw_response)
            
            # Validate scores are in range 1-5
            response_data = self._validate_scores(response_data)
            
            evaluation = EvaluationScores(**response_data)
            
            overall_score = self.calculate_overall_score(evaluation)
            log_step("EVALUATION", f"Evaluation complete: overall score {overall_score:.1f}/5")
            log_agent_communication(
                from_agent="EvaluationAgent",
                to_agent="Main",
                message_type="evaluation_complete",
                data={
                    "overall_score": overall_score,
                    "scores": {
                        "interest_match": evaluation.interest_match,
                        "budget_realism": evaluation.budget_realism,
                        "logistics": evaluation.logistics,
                        "suitability": evaluation.suitability_for_constraints
                    }
                }
            )
            
            return evaluation
            
        except Exception as e:
            error_msg = f"Error evaluating itinerary: {e}"
            log_step("EVALUATION", error_msg, level="error")
            
            # Create fallback evaluation
            fallback = self._create_fallback_evaluation()
            overall_score = self.calculate_overall_score(fallback)
            
            log_step("EVALUATION", f"Using fallback evaluation with score {overall_score:.1f}/5", level="warning")
            
            return fallback
    
    def _create_evaluation_prompt(self, profile: Dict, itinerary: Dict) -> str:
        """Create evaluation prompt."""
        log_step("EVALUATION", "Creating evaluation prompt")
        
        # Simplify the profile for evaluation
        simplified_profile = {
            "interests": profile.get("refined_profile", "Unknown interests"),
            "city": profile.get("chosen_city", "Unknown city"),
            "budget": profile.get("constraints", {}).get("budget", 0),
            "people": profile.get("constraints", {}).get("people", 1),
            "constraints": {
                "with_children": profile.get("constraints", {}).get("with_children", False),
                "with_disabled": profile.get("constraints", {}).get("with_disabled", False)
            }
        }
        
        # Simplify itinerary for evaluation
        simplified_itinerary = {}
        for day_key in ['day1', 'day2', 'day3']:
            day_data = itinerary.get(day_key, {})
            if day_data:
                simplified_day = {}
                for time_slot in ['morning', 'afternoon', 'evening']:
                    slot_attractions = day_data.get(time_slot, [])
                    if slot_attractions:
                        simplified_day[time_slot] = [
                            attr.get('name', 'Unknown')[:50] for attr in slot_attractions[:3]
                        ]
                if simplified_day:
                    simplified_itinerary[day_key] = simplified_day
        
        log_step("EVALUATION", f"Evaluation context: city={simplified_profile['city']}, budget={simplified_profile['budget']}€", level="debug")
        
        return f"""You are an impartial travel expert evaluating a travel itinerary.

USER PROFILE:
{json.dumps(simplified_profile, indent=2)}

PROPOSED ITINERARY:
{json.dumps(simplified_itinerary, indent=2)}

Rate the itinerary from 1 to 5 (integers only) on these dimensions:
1. interest_match: How well does the itinerary match the user's stated interests?
2. budget_realism: Is the budget allocation realistic and appropriate?
3. logistics: How well does the schedule flow? Are time slots reasonably filled?
4. suitability_for_constraints: Does it accommodate any constraints (children, disabilities)?

Also provide a short comment summarizing your assessment.

Return ONLY valid JSON with this exact structure:
{{
  "interest_match": 1-5,
  "budget_realism": 1-5,
  "logistics": 1-5,
  "suitability_for_constraints": 1-5,
  "comment": "Your concise evaluation comment here"
}}

Important:
- Be objective and fair in your assessment
- Consider the user's budget, interests, and constraints
- Focus on practical feasibility"""
    
    def _validate_scores(self, scores: Dict) -> Dict:
        """Ensure scores are valid integers between 1-5."""
        log_step("EVALUATION", "Validating evaluation scores")
        
        validated_scores = {}
        for key in ["interest_match", "budget_realism", "logistics", "suitability_for_constraints"]:
            if key in scores:
                try:
                    score = int(scores[key])
                    validated_scores[key] = max(1, min(5, score))  # Clamp to 1-5
                    log_step("EVALUATION", f"{key}: {validated_scores[key]}/5", level="debug")
                except (ValueError, TypeError):
                    validated_scores[key] = 3  # Default to average
                    log_step("EVALUATION", f"{key}: Invalid score, using default 3", level="warning")
            else:
                validated_scores[key] = 3  # Default to average
                log_step("EVALUATION", f"{key}: Missing, using default 3", level="warning")
        
        # Ensure comment exists
        validated_scores.setdefault("comment", "Evaluation completed")
        log_step("EVALUATION", f"Comment: {validated_scores['comment'][:100]}...", level="debug")
        
        return validated_scores
    
    def _create_fallback_evaluation(self) -> EvaluationScores:
        """Create fallback evaluation when LLM fails."""
        log_step("EVALUATION", "Creating fallback evaluation", level="warning")
        
        return EvaluationScores(
            interest_match=3,
            budget_realism=3,
            logistics=3,
            suitability_for_constraints=3,
            comment="Evaluation could not be completed. Using default scores."
        )
    
    def calculate_overall_score(self, evaluation: EvaluationScores) -> float:
        """Calculate overall score from individual dimensions."""
        scores = [
            evaluation.interest_match,
            evaluation.budget_realism,
            evaluation.logistics,
            evaluation.suitability_for_constraints
        ]
        overall = sum(scores) / len(scores)
        log_step("EVALUATION", f"Overall score calculated: {overall:.1f}/5", level="debug")
        return overall