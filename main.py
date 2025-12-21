#!/usr/bin/env python3
"""
main.py

Defines TravelPlanner and orchestrates:
1) User provides initial preferences
2) Interest refinement dialogue: asks at least 1 question, max 3 questions total
3) Finalizes TravelProfile
4) Runs full pipeline: attractions -> google enrich -> budget -> schedule -> evaluation
5) Saves evaluation + full run artifact into ./logs/

IMPORTANT:
- This file must NOT import TravelPlanner from main (no self-import).
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Optional, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

from agents.semantic_agent import SemanticAgent
from agents.interest_refinement_agent import InterestRefinementAgent
from agents.location_scout_agent import LocationScoutAgent
from agents.budget_agent import BudgetAgent
from agents.scheduler_agent import SchedulerAgent
from agents.evaluation_agent import EvaluationAgent
from agents.google_places_agent import GooglePlacesAgent

from utils.data_structures import PreferenceState, TravelProfile
from utils.llm_client import LLMClient


def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _json_safe(obj: Any):
    """Best-effort conversion to JSON-serializable types."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


class TravelPlanner:
    """
    Orchestrator for UI (Streamlit) use.

    UX requirement:
    - After initial user preferences, the system MUST ask at least one refinement question.
    - It may ask fewer than 3 if it can finalize early, but max is 3 total.
    """

    MAX_INTEREST_QUESTIONS = 3

    def __init__(self):
        self.llm_client = LLMClient()

        self.semantic_agent = SemanticAgent()
        self.interest_agent = InterestRefinementAgent()
        self.location_agent = LocationScoutAgent()
        self.budget_agent = BudgetAgent()
        self.scheduler_agent = SchedulerAgent()
        self.evaluation_agent = EvaluationAgent()
        self.places_agent = GooglePlacesAgent()

        self.state = PreferenceState()
        self.basic_info: Optional[Dict[str, Any]] = None

        # How many refinement questions have been asked (assistant questions)
        self.interest_questions_asked: int = 0

        self.initial_preferences_text: str = ""
        self.last_evaluation_path: Optional[str] = None
        self.last_run_path: Optional[str] = None

    def reset(self, basic_info: Dict[str, Any]):
        """
        Reset planner state for a new trip.
        """
        self.basic_info = basic_info
        self.state = PreferenceState()
        self.interest_questions_asked = 0
        self.initial_preferences_text = ""
        self.last_evaluation_path = None
        self.last_run_path = None

    # ----------------------------
    # Dialogue: start + continue
    # ----------------------------
    def start_refinement(self, initial_preferences: str) -> Dict[str, Any]:
        """
        Called once after user submits initial preference text.

        GUARANTEE:
        - This returns an ask_question response (at least one question).
        """
        if not self.basic_info:
            raise RuntimeError("basic_info not set. Call reset() first.")

        self.initial_preferences_text = (initial_preferences or "").strip()

        if self.initial_preferences_text:
            self.state = self.semantic_agent.update_state(self.state, self.initial_preferences_text)

        budget = float(self.basic_info["budget"])
        people = int(self.basic_info["people"])
        days = int(self.basic_info["days"])

        # Ask the agent for the first question.
        # We pass questions_asked_so_far=0 so the agent MUST ask a question.
        llm_output = self.interest_agent.process_turn(
            self.state,
            last_user_msg=self.initial_preferences_text,
            budget=budget,
            people=people,
            days=days,
            questions_asked_so_far=0,
        )

        # Defensive: always ask at least one question
        if llm_output.get("action") != "ask_question":
            llm_output["action"] = "ask_question"
            llm_output["question"] = llm_output.get("question") or "What is one must-have experience or vibe you want on this trip?"

        self.interest_questions_asked = 1

        return {
            "action": "ask_question",
            "question": llm_output.get("question") or "What is one must-have experience or vibe you want on this trip?",
            "profile": None,
        }

    def process_refinement_turn(self, user_text: str) -> Dict[str, Any]:
        """
        Called after each user answer to the refinement question.

        Returns:
          - {"action":"ask_question","question":...}
          - {"action":"finalize","profile": TravelProfile}
        """
        if not self.basic_info:
            raise RuntimeError("basic_info not set. Call reset() first.")

        user_text = (user_text or "").strip()
        if user_text:
            self.state = self.semantic_agent.update_state(self.state, user_text)

        budget = float(self.basic_info["budget"])
        people = int(self.basic_info["people"])
        days = int(self.basic_info["days"])

        llm_output = self.interest_agent.process_turn(
            self.state,
            last_user_msg=user_text,
            budget=budget,
            people=people,
            days=days,
            questions_asked_so_far=self.interest_questions_asked,
        )

        action = (llm_output.get("action") or "").strip().lower()
        if action not in ("ask_question", "finalize"):
            action = "ask_question"

        if action == "ask_question":
            # If we're at cap already, force finalize
            if self.interest_questions_asked >= self.MAX_INTEREST_QUESTIONS:
                profile = self._force_finalize(last_user_msg=user_text)
                return {"action": "finalize", "profile": profile, "question": ""}

            self.interest_questions_asked += 1
            return {
                "action": "ask_question",
                "question": llm_output.get("question") or "Anything else you want included?",
                "profile": None,
            }

        # finalize
        profile = self.interest_agent.create_final_profile(self.state, llm_output)
        return {"action": "finalize", "profile": profile, "question": ""}

    def _force_finalize(self, last_user_msg: str) -> TravelProfile:
        """
        If the agent keeps asking beyond the cap, finalize using fallback.
        """
        budget = float(self.basic_info["budget"])
        people = int(self.basic_info["people"])
        days = int(self.basic_info["days"])

        # Prefer agent's own fallback city method if present
        if hasattr(self.interest_agent, "_get_fallback_city"):
            prefs_compact = getattr(self.interest_agent, "_extract_user_preferences_optimized", lambda s, m: m)(self.state, last_user_msg)
            chosen_city = self.interest_agent._get_fallback_city(prefs_compact)
        else:
            chosen_city = "Paris"

        refined_profile = (
            (self.state.slots.get("activities") or "").strip()
            or (self.state.slots.get("pace") or "").strip()
            or (self.state.slots.get("food") or "").strip()
            or "Preferences collected"
        )

        llm_output = {
            "action": "finalize",
            "question": "",
            "chosen_city": chosen_city,
            "constraints": {
                "with_children": bool(self.basic_info.get("with_children", False)),
                "with_disabled": bool(self.basic_info.get("with_disabled", False)),
                "budget": budget,
                "people": people,
            },
            "refined_profile": refined_profile,
        }

        return self.interest_agent.create_final_profile(self.state, llm_output)

    # ----------------------------
    # Pipeline + Save artifacts
    # ----------------------------
    def run_pipeline_from_profile(self, profile: TravelProfile) -> Dict[str, Any]:
        """
        Slow step. Call only after refinement finalizes.
        Saves:
          logs/evaluation_*.json
          logs/run_*.json
        """
        basic_info = self.basic_info or {}

        budget = float(basic_info.get("budget", getattr(profile.constraints, "budget", getattr(Config, "DEFAULT_BUDGET", 800))))
        people = int(basic_info.get("people", getattr(profile.constraints, "people", getattr(Config, "DEFAULT_PEOPLE", 2))))
        days = int(basic_info.get("days", getattr(Config, "DEFAULT_DAYS", 3)))

        constraints_dict = profile.constraints.model_dump() if hasattr(profile.constraints, "model_dump") else dict(profile.constraints)

        # 1) Generate attractions
        attractions = self.location_agent.generate_attractions(
            profile.chosen_city,
            profile.refined_profile,
            constraints_dict,
        )

        # 2) Enrich with Google Places
        if hasattr(self.places_agent, "is_enabled") and self.places_agent.is_enabled():
            attractions_enriched = self.places_agent.enrich_attractions(attractions, profile.chosen_city)
        else:
            attractions_enriched = attractions

        # 3) Budget filter
        affordable = self.budget_agent.filter_by_budget(attractions_enriched, budget, days, people)

        # 4) Schedule
        itinerary = self.scheduler_agent.create_itinerary(affordable, days)

        # 5) Evaluate
        evaluation = self.evaluation_agent.evaluate_itinerary(
            profile.model_dump() if hasattr(profile, "model_dump") else profile,
            itinerary.model_dump() if hasattr(itinerary, "model_dump") else itinerary,
        )

        # Save artifacts
        logs_dir = _ensure_logs_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        evaluation_path = os.path.join(logs_dir, f"evaluation_{stamp}.json")
        with open(evaluation_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(evaluation), f, ensure_ascii=False, indent=2)

        run_path = os.path.join(logs_dir, f"run_{stamp}.json")
        run_artifact = {
            "timestamp": stamp,
            "basic_info": _json_safe(self.basic_info),
            "profile": _json_safe(profile),
            "attractions_generated": _json_safe(attractions),
            "attractions_enriched": _json_safe(attractions_enriched),
            "attractions_budget_filtered": _json_safe(affordable),
            "itinerary": _json_safe(itinerary),
            "evaluation": _json_safe(evaluation),
        }
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(run_artifact, f, ensure_ascii=False, indent=2)

        self.last_evaluation_path = evaluation_path
        self.last_run_path = run_path

        return {
            "profile": _json_safe(profile),
            "attractions_generated": _json_safe(attractions),
            "attractions_enriched": _json_safe(attractions_enriched),
            "attractions_budget_filtered": _json_safe(affordable),
            "itinerary": _json_safe(itinerary),
            "evaluation": _json_safe(evaluation),
            "evaluation_file": evaluation_path,
            "run_file": run_path,
        }

    # Optional: Stream a natural-language summary (requires llm_client streaming support)
    def stream_final_summary_words(self, profile_dict: Dict[str, Any], itinerary_dict: Dict[str, Any]):
        prompt = f"""You are a friendly travel assistant.
Write a short helpful summary of the itinerary. No JSON.

PROFILE:
{profile_dict}

ITINERARY:
{itinerary_dict}

Assistant:"""
        # Must exist in your LLMClient; if not, implement it in utils/llm_client.py
        yield from self.llm_client.generate_stream_words(prompt, temperature=0.7)
