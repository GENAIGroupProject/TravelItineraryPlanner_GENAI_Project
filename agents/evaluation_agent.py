import json
from typing import Dict
from utils.llm_client import LLMClient
from utils.json_parser import JSONParser
from utils.data_structures import EvaluationScores

class EvaluationAgent:
    """Agent for evaluating itinerary quality."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.json_parser = JSONParser()
    
    def evaluate_itinerary(self, profile: Dict, itinerary: Dict) -> EvaluationScores:
        """Evaluate an itinerary against user profile."""
        prompt = self._create_evaluation_prompt(profile, itinerary)
        
        try:
            print(f"\n📊 Generating evaluation prompt...")
            raw_response = self.llm_client.generate(prompt, temperature=0.3)
            print(f"📊 Raw evaluation response:\n{raw_response[:500]}...\n")
            
            response_data = self.json_parser.parse_response(raw_response)
            
            # Validate scores are in range 1-5
            response_data = self._validate_scores(response_data)
            
            return EvaluationScores(**response_data)
            
        except Exception as e:
            print(f"❌ Error evaluating itinerary: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_evaluation()
    
    def _create_evaluation_prompt(self, profile: Dict, itinerary: Dict) -> str:
        """Create evaluation prompt."""
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
        
        return f"""You are an impartial travel expert evaluating a travel itinerary.

USER PROFILE:
{json.dumps(simplified_profile, indent=2)}

PROPOSED ITINERARY:
{json.dumps(itinerary, indent=2)}

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
        for key in ["interest_match", "budget_realism", "logistics", "suitability_for_constraints"]:
            if key in scores:
                try:
                    score = int(scores[key])
                    scores[key] = max(1, min(5, score))  # Clamp to 1-5
                except (ValueError, TypeError):
                    scores[key] = 3  # Default to average
        
        # Ensure comment exists
        scores.setdefault("comment", "Evaluation completed")
        
        return scores
    
    def _create_fallback_evaluation(self) -> EvaluationScores:
        """Create fallback evaluation when LLM fails."""
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
        return sum(scores) / len(scores)
