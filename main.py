#!/usr/bin/env python3
"""
main.py

Defines TravelPlanner ONLY.
Do NOT import TravelPlanner from main inside this file (circular import).
Streamlit pages can import TravelPlanner using: from main import TravelPlanner
"""

import os
import sys
from typing import Dict, Optional, Any

# ensure project root in path
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


class TravelPlanner:
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

        self.interest_questions_asked = 0
        self.initial_preferences_text: str = ""

    def reset(self, basic_info: Dict[str, Any]):
        self.basic_info = basic_info
        self.state = PreferenceState()
        self.interest_questions_asked = 0
        self.initial_preferences_text = ""

    # ----------------------------
    # Step 1: Start refinement after initial preferences
    # ----------------------------
    def start_refinement(self, initial_preferences: str) -> Dict[str, Any]:
        if not self.basic_info:
            raise RuntimeError("basic_info not set. Call reset() first.")

        self.initial_preferences_text = (initial_preferences or "").strip()

        if self.initial_preferences_text:
            self.state = self.semantic_agent.update_state(self.state, self.initial_preferences_text)

        budget = float(self.basic_info["budget"])
        people = int(self.basic_info["people"])
        days = int(self.basic_info["days"])

        llm_output = self.interest_agent.process_turn(
            self.state,
            last_user_msg=self.initial_preferences_text,
            budget=budget,
            people=people,
            days=days
        )

        action = llm_output.get("action")

        if action == "ask_question":
            self.interest_questions_asked = 1
            return {
                "action": "ask_question",
                "question": llm_output.get("question") or "What kinds of activities do you enjoy most?"
            }

        profile = self.interest_agent.create_final_profile(self.state, llm_output)
        return {"action": "finalize", "profile": profile, "question": ""}

    # ----------------------------
    # Step 2: Continue refinement turns (cap at 3)
    # ----------------------------
    def process_refinement_turn(self, user_text: str) -> Dict[str, Any]:
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
            days=days
        )

        action = llm_output.get("action")

        if action == "ask_question":
            if self.interest_questions_asked >= self.MAX_INTEREST_QUESTIONS:
                profile = self._force_finalize(last_user_msg=user_text)
                return {"action": "finalize", "profile": profile, "question": ""}

            self.interest_questions_asked += 1
            return {
                "action": "ask_question",
                "question": llm_output.get("question") or "Anything else you want included?"
            }

        profile = self.interest_agent.create_final_profile(self.state, llm_output)
        return {"action": "finalize", "profile": profile, "question": ""}

    def _force_finalize(self, last_user_msg: str) -> TravelProfile:
        budget = float(self.basic_info["budget"])
        people = int(self.basic_info["people"])
        days = int(self.basic_info["days"])

        chosen_city = self.interest_agent._get_city_recommendation(
            self.state, last_user_msg=last_user_msg, budget=budget, days=days
        )

        refined_profile = (
            self.state.slots.get("activities")
            or self.state.slots.get("pace")
            or self.state.slots.get("food")
            or "Preferences collected"
        )

        constraints = {
            "with_children": bool(self.basic_info.get("with_children", False)),
            "with_disabled": bool(self.basic_info.get("with_disabled", False)),
            "budget": budget,
            "people": people,
        }

        llm_output = {
            "action": "finalize",
            "question": "",
            "chosen_city": chosen_city,
            "refined_profile": refined_profile,
            "constraints": constraints,
            "travel_style": "medium",
        }

        return self.interest_agent.create_final_profile(self.state, llm_output)

    # ----------------------------
    # Step 3: Full pipeline after finalize (slow)
    # ----------------------------
    def run_pipeline_from_profile(self, profile: TravelProfile) -> Dict[str, Any]:
        basic_info = self.basic_info or {}
        budget = float(basic_info.get("budget", getattr(profile.constraints, "budget", Config.DEFAULT_BUDGET)))
        people = int(basic_info.get("people", getattr(profile.constraints, "people", Config.DEFAULT_PEOPLE)))
        days = int(basic_info.get("days", Config.DEFAULT_DAYS))

        constraints_dict = profile.constraints.model_dump() if hasattr(profile.constraints, "model_dump") else dict(profile.constraints)

        attractions = self.location_agent.generate_attractions(profile.chosen_city, profile.refined_profile, constraints_dict)

        if hasattr(self.places_agent, "is_enabled") and self.places_agent.is_enabled():
            attractions_enriched = self.places_agent.enrich_attractions(attractions, profile.chosen_city)
        else:
            attractions_enriched = attractions

        affordable = self.budget_agent.filter_by_budget(attractions_enriched, budget, days, people)

        itinerary = self.scheduler_agent.create_itinerary(affordable, days)

        evaluation = self.evaluation_agent.evaluate_itinerary(
            profile.model_dump() if hasattr(profile, "model_dump") else profile,
            itinerary.model_dump() if hasattr(itinerary, "model_dump") else itinerary
        )

        def dump(x):
            if hasattr(x, "model_dump"):
                return x.model_dump()
            if hasattr(x, "dict"):
                return x.dict()
            return x

        return {
            "profile": dump(profile),
            "attractions_generated": [dump(a) for a in attractions],
            "attractions_enriched": [dump(a) for a in attractions_enriched],
            "attractions_budget_filtered": [dump(a) for a in affordable],
            "itinerary": dump(itinerary),
            "evaluation": dump(evaluation),
        }

    def stream_final_summary_words(self, profile_dict: Dict[str, Any], itinerary_dict: Dict[str, Any]):
        prompt = f"""You are a friendly travel assistant.
Write a short helpful summary of the itinerary. No JSON.

PROFILE:
{profile_dict}

ITINERARY:
{itinerary_dict}

Assistant:"""
        yield from self.llm_client.generate_stream_words(prompt, temperature=0.7)
