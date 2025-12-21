import json
import re
from typing import Dict, Optional

from utils.llm_client import LLMClient
from utils.data_structures import PreferenceState, TripConstraints, TravelProfile
from config import Config
from utils.logging_utils import log_step, log_agent_output, log_agent_communication


class InterestRefinementAgent:
    """
    LLM-only dialogue agent.

    Requirements:
    - Questions must come ONLY from the LLM (no default questions).
    - Ask at least one question: first turn must be ask_question.
    - May ask fewer than 3 if it can finalize early (after at least one question).
    - Output should be JSON (for structured handling), but if parsing fails we
      re-ask the LLM to fix output or return question-only.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        log_step("INTEREST_REFINEMENT", "Interest refinement agent initialized (LLM-only questions)")

    # ----------------------------
    # JSON extraction (no JSONParser dependency)
    # ----------------------------
    def _extract_json_object(self, text: str) -> Optional[Dict]:
        if not text:
            return None

        # direct parse
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # attempt extract first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None

        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _compact_prefs(self, state: PreferenceState, last_user_msg: str) -> str:
        parts = []
        for key in ["activities", "pace", "food", "constraints"]:
            v = (state.slots.get(key) or "").strip()
            if v:
                parts.append(v[:200])
        if not parts and last_user_msg:
            parts.append(last_user_msg[:260])
        return " | ".join(parts) if parts else (last_user_msg[:260] if last_user_msg else "not specified")

    # ----------------------------
    # LLM calls
    # ----------------------------
    def _prompt_turn_json(
        self,
        prefs: str,
        last_user_msg: str,
        budget: float,
        people: int,
        days: int,
        questions_asked_so_far: int,
    ) -> str:
        """
        JSON-only prompt for either asking a question or finalizing.
        We keep it short for speed.
        """
        must_ask = "YES" if questions_asked_so_far <= 0 else "NO"

        return f"""Return ONLY valid JSON (no markdown, no extra text).

Trip:
days={days}, people={people}, budget_eur={budget}

Preferences: {prefs}
Last user message: {last_user_msg}

RULES:
- Ask at most one question this turn.
- If MUST_ASK_FIRST_QUESTION=YES then action MUST be "ask_question".
- Otherwise action can be "ask_question" or "finalize".
- Keep the question short and specific.

MUST_ASK_FIRST_QUESTION={must_ask}

Schema:
{{
  "action": "ask_question" or "finalize",
  "question": "string (required if ask_question, empty if finalize)",
  "chosen_city": "string or null",
  "refined_profile": "short summary string",
  "constraints": {{
    "with_children": true/false,
    "with_disabled": true/false,
    "budget": number,
    "people": number
  }}
}}
"""

    def _prompt_fix_json(self, bad_output: str) -> str:
        """
        Ask LLM to convert whatever it wrote into valid JSON of the required schema.
        """
        return f"""Convert the following into ONLY valid JSON (no markdown, no extra text).
If fields are missing, fill them with null/empty values but keep the schema.

Required schema:
{{
  "action": "ask_question" or "finalize",
  "question": "string",
  "chosen_city": "string or null",
  "refined_profile": "string",
  "constraints": {{
    "with_children": true/false,
    "with_disabled": true/false,
    "budget": number,
    "people": number
  }}
}}

BAD_OUTPUT:
{bad_output}
"""

    def _prompt_question_only(self, prefs: str, last_user_msg: str) -> str:
        """
        Absolute fallback that still respects your requirement:
        question comes from LLM, and we accept plain text (not JSON).
        """
        return f"""Write ONE short question to clarify the user's travel preferences.
No JSON, no bullets—just the question as a single sentence.

Preferences: {prefs}
Last user message: {last_user_msg}
"""

    def _normalize_output(self, parsed: Dict, budget: float, people: int) -> Dict:
        """
        Clean up the parsed dict and ensure required fields exist.
        """
        action = (parsed.get("action") or "").strip().lower()
        if action not in ("ask_question", "finalize"):
            action = "ask_question"

        question = (parsed.get("question") or "").strip()
        chosen_city = parsed.get("chosen_city")
        refined_profile = (parsed.get("refined_profile") or "").strip()

        constraints = parsed.get("constraints") or {}
        constraints = {
            "with_children": bool(constraints.get("with_children", False)),
            "with_disabled": bool(constraints.get("with_disabled", False)),
            "budget": float(constraints.get("budget", budget)),
            "people": int(constraints.get("people", people)),
        }

        return {
            "action": action,
            "question": question,
            "chosen_city": chosen_city,
            "refined_profile": refined_profile,
            "constraints": constraints,
        }

    # ----------------------------
    # Public API
    # ----------------------------
    def process_turn(
        self,
        state: PreferenceState,
        last_user_msg: str,
        budget: float,
        people: int,
        days: int,
        questions_asked_so_far: int = 0,
    ) -> Dict:
        """
        Returns dict with:
          action: ask_question/finalize
          question / chosen_city / refined_profile / constraints
        """

        log_step("INTEREST_REFINEMENT", "Processing dialogue turn (LLM-only)")
        last_user_msg = (last_user_msg or "").strip()
        prefs = self._compact_prefs(state, last_user_msg)

        prompt = self._prompt_turn_json(
            prefs=prefs,
            last_user_msg=last_user_msg,
            budget=budget,
            people=people,
            days=days,
            questions_asked_so_far=questions_asked_so_far,
        )

        log_agent_communication(
            from_agent="InterestRefinementAgent",
            to_agent="LLM",
            message_type="dialogue_turn_request",
            data={"questions_asked_so_far": questions_asked_so_far, "prompt_preview": prompt[:260]},
        )

        raw = self.llm_client.generate(prompt, temperature=0.3)
        parsed = self._extract_json_object(raw)

        # Retry 1: ask LLM to fix JSON
        if parsed is None:
            fix_prompt = self._prompt_fix_json(raw[:2000])
            raw2 = self.llm_client.generate(fix_prompt, temperature=0.0)
            parsed = self._extract_json_object(raw2)

        # Retry 2: still no JSON => get question-only from LLM (no defaults!)
        if parsed is None:
            q_prompt = self._prompt_question_only(prefs, last_user_msg)
            q_text = (self.llm_client.generate(q_prompt, temperature=0.2) or "").strip()
            # Force ask_question if first turn; otherwise still ask question (safe)
            out = {
                "action": "ask_question",
                "question": q_text if q_text else (self.llm_client.generate(q_prompt, temperature=0.2) or "").strip(),
                "chosen_city": None,
                "refined_profile": "",
                "constraints": {
                    "with_children": False,
                    "with_disabled": False,
                    "budget": float(budget),
                    "people": int(people),
                },
            }
            # Enforce: first turn must be a question
            if questions_asked_so_far <= 0:
                out["action"] = "ask_question"
            log_agent_output("InterestRefinementAgent", out, context="turn_output_question_only_fallback")
            return out

        out = self._normalize_output(parsed, budget=budget, people=people)

        # Enforce: first turn must be ask_question (even if model tried finalize)
        if questions_asked_so_far <= 0:
            out["action"] = "ask_question"
            # If question missing, ask LLM again for one (still no defaults)
            if not out["question"]:
                q_prompt = self._prompt_question_only(prefs, last_user_msg)
                out["question"] = (self.llm_client.generate(q_prompt, temperature=0.2) or "").strip()

        # If action ask_question but question empty => ask LLM for question (no defaults)
        if out["action"] == "ask_question" and not out["question"]:
            q_prompt = self._prompt_question_only(prefs, last_user_msg)
            out["question"] = (self.llm_client.generate(q_prompt, temperature=0.2) or "").strip()

        log_agent_output("InterestRefinementAgent", out, context="turn_output")
        return out

    def create_final_profile(self, state: PreferenceState, llm_output: Dict) -> TravelProfile:
        chosen_city = (llm_output.get("chosen_city") or "").strip()
        refined_profile = (llm_output.get("refined_profile") or "").strip()
        constraints_dict = llm_output.get("constraints") or {}

        # If chosen_city missing, we ask LLM quickly for a city (still LLM-based)
        if not chosen_city:
            prefs = self._compact_prefs(state, "")
            city_prompt = f"""Return ONLY the city name as plain text (no JSON).
Pick one destination city that fits: {prefs}
"""
            chosen_city = (self.llm_client.generate(city_prompt, temperature=0.0) or "").strip() or "Paris"

        if not refined_profile:
            refined_profile = self._compact_prefs(state, "")

        constraints = TripConstraints(
            with_children=bool(constraints_dict.get("with_children", False)),
            with_disabled=bool(constraints_dict.get("with_disabled", False)),
            budget=float(constraints_dict.get("budget", getattr(Config, "DEFAULT_BUDGET", 800))),
            people=int(constraints_dict.get("people", getattr(Config, "DEFAULT_PEOPLE", 2))),
        )

        profile = TravelProfile(
            chosen_city=chosen_city,
            refined_profile=refined_profile,
            constraints=constraints,
        )

        log_step("INTEREST_REFINEMENT", f"Profile created: {chosen_city}")
        return profile
