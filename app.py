# app.py
import re
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

###############################################################################
# CONFIG
###############################################################################

LLAMA_MODEL_NAME = "llama3"  # change if needed
GOOGLE_API_KEY = "YOUR_GOOGLE_PLACES_API_KEY_HERE"  # TODO: set real key

# For trial / demo:
DEFAULT_BUDGET = 600.0  # simple per-trip budget example
DEFAULT_DAYS = 3


###############################################################################
# LLM CALL – YOU MUST IMPLEMENT THIS FOR YOUR ENV
###############################################################################

def call_llama(prompt: str, model: str = LLAMA_MODEL_NAME) -> str:
    """
    Placeholder for calling your Llama model.

    Examples:
    - If using Ollama: (uncomment and adapt)

        import subprocess
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8")

    - If using an HTTP endpoint: use requests.post(...)
    """
    # TODO: replace this with your real LLM call
    raise NotImplementedError("call_llama() must be implemented for your environment.")


###############################################################################
# SEMANTIC LAYER – MINILM MODEL & SLOT CLASSIFICATION
###############################################################################

emb_model = SentenceTransformer("all-MiniLM-L6-v2")

SLOT_LABELS = {
    "activities": "The user talks about preferred activities or attractions like museums, parks, nightlife, beaches.",
    "pace": "The user describes how fast they like to travel, for example slow, relaxed, or packed schedule.",
    "budget": "The user mentions budget, price range, cheap, expensive or how much money to spend.",
    "constraints": "The user mentions children, kids, disabled travelers, mobility limitations or accessibility.",
    "food": "The user talks about restaurants, food or cuisine preferences.",
    "other": "Other preferences not covered above."
}

SLOT_EMBS = {
    k: emb_model.encode([v], normalize_embeddings=True)[0]
    for k, v in SLOT_LABELS.items()
}


@dataclass
class PreferenceSnippet:
    text: str
    embedding: np.ndarray
    slot: str


@dataclass
class PreferenceState:
    snippets: List[PreferenceSnippet] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=lambda: {
        "activities": "",
        "pace": "",
        "food": "",
        "constraints": "",
        "budget": "",
        "other": ""
    })
    global_embedding: Optional[np.ndarray] = None
    turns: int = 0


def split_into_sentences(text: str) -> List[str]:
    # simple heuristic; good enough for trial
    parts = re.split(r"[.!?]\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def classify_slot(sentence_emb: np.ndarray) -> str:
    best_slot, best_score = None, -1.0
    for slot, label_emb in SLOT_EMBS.items():
        score = float(sentence_emb @ label_emb)
        if score > best_score:
            best_score, best_slot = score, slot
    return best_slot


SIM_UPDATE_THRESHOLD = 0.75


def update_state_with_message(state: PreferenceState, user_msg: str) -> PreferenceState:
    sentences = split_into_sentences(user_msg)
    if not sentences:
        return state

    new_snippets = []
    for s in sentences:
        emb = emb_model.encode([s], normalize_embeddings=True)[0]
        slot = classify_slot(emb)

        same_slot = [sn for sn in state.snippets if sn.slot == slot]
        best_sn, best_sim = None, -1.0
        for sn in same_slot:
            sim = float(emb @ sn.embedding)
            if sim > best_sim:
                best_sim, best_sn = sim, sn

        if best_sn is not None and best_sim > SIM_UPDATE_THRESHOLD:
            # refinement of existing info
            best_sn.text = s
            best_sn.embedding = emb
        else:
            new_snippets.append(PreferenceSnippet(text=s, embedding=emb, slot=slot))

    state.snippets.extend(new_snippets)

    # rebuild slot texts
    state.slots = {k: "" for k in state.slots}
    for sn in state.snippets:
        sep = " " if state.slots[sn.slot] else ""
        state.slots[sn.slot] += sep + sn.text

    # global embedding = mean of all snippet embeddings
    all_embs = [sn.embedding for sn in state.snippets]
    state.global_embedding = np.mean(all_embs, axis=0) if all_embs else None

    state.turns += 1
    return state


def build_profile_summary(state: PreferenceState) -> str:
    return (
        f"Activities: {state.slots['activities'] or 'not specified yet'}\n"
        f"Pace: {state.slots['pace'] or 'not specified yet'}\n"
        f"Food: {state.slots['food'] or 'not specified yet'}\n"
        f"Constraints (children, disabled, etc.): {state.slots['constraints'] or 'not specified yet'}\n"
        f"Budget: {state.slots['budget'] or 'not specified yet'}"
    )


###############################################################################
# INTEREST-REFINEMENT & CITY-MATCHMAKER (LLAMA LAYER)
###############################################################################

def llama_next_turn(state: PreferenceState, last_user_msg: str, budget: float):
    profile_summary = build_profile_summary(state)

    prompt = f"""
You are the "Interest-Refinement & City-Matchmaker Agent" in a travel planning system.

We are interviewing a user to plan a 3-day city trip in Europe.

CURRENT SEMANTIC PROFILE (built from all previous messages):
{profile_summary}

Most recent user message:
\"\"\"{last_user_msg}\"\"\"

Your tasks:
1. Decide if we have enough information to recommend a city now.
2. If NOT, ask ONE next clarifying question that is maximally useful (about activities, budget, constraints, or pace).
3. If YES, propose ONE European city and refine the user profile.

Output ONLY valid JSON with this structure:
{{
  "action": "ask_question" or "finalize",
  "question": "string (if action == 'ask_question', else empty)",
  "refined_profile": "short natural language summary of what the user wants",
  "chosen_city": "string or null if not ready",
  "constraints": {{
    "with_children": true or false or null,
    "with_disabled": true or false or null,
    "budget": {budget}
  }},
  "travel_style": "slow" or "medium" or "fast" or null
}}
"""

    raw = call_llama(prompt)
    # Expect JSON in the model output; if model adds text, extract JSON area.
    json_str = extract_json(raw)
    data = json.loads(json_str)
    return data


def extract_json(text: str) -> str:
    """
    Simple helper to extract first {...} block. Not bulletproof, but OK for trial.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")
    return text[start:end + 1]


def run_interest_and_city_dialogue(budget: float) -> Dict:
    """
    Simple CLI-style dialogue:
    - Ask initial open question
    - Loop until Llama returns action == "finalize"
    """
    state = PreferenceState()

    print("Welcome to the travel planner!")
    user_msg = input("Tell me about your trip preferences: ")
    state = update_state_with_message(state, user_msg)

    while True:
        llama_output = llama_next_turn(state, user_msg, budget)
        action = llama_output["action"]
        if action == "ask_question":
            q = llama_output["question"]
            print(f"\nAgent: {q}")
            user_msg = input("You: ")
            state = update_state_with_message(state, user_msg)
        elif action == "finalize":
            # finalize and break
            profile = {
                "refined_profile": llama_output["refined_profile"],
                "chosen_city": llama_output["chosen_city"],
                "constraints": llama_output["constraints"],
                "travel_style": llama_output["travel_style"],
                "semantic_profile_slots": state.slots,
                "interest_embedding": state.global_embedding.tolist()
                if state.global_embedding is not None else None,
            }
            print("\n--- Finalized profile ---")
            print(json.dumps(profile, indent=2))
            return profile
        else:
            raise ValueError(f"Unknown action from LLM: {action}")


###############################################################################
# GOOGLE PLACES API (OPENING HOURS)
###############################################################################

def google_find_place(name: str, city: str) -> Optional[dict]:
    """
    Uses 'Find Place From Text' to get a place_id.
    """
    if GOOGLE_API_KEY == "YOUR_GOOGLE_PLACES_API_KEY_HERE":
        # Keep code safe to run without key; just return None in trial
        return None

    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": f"{name}, {city}",
        "inputtype": "textquery",
        "fields": "place_id",
        "key": GOOGLE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    return candidates[0]


def google_place_details(place_id: str) -> Optional[dict]:
    if GOOGLE_API_KEY == "YOUR_GOOGLE_PLACES_API_KEY_HERE":
        return None

    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,geometry,opening_hours,price_level,types,rating,user_ratings_total",
        "key": GOOGLE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("result")


def enrich_with_google_places(attractions: List[dict], city: str) -> List[dict]:
    """
    For each attraction: look up place_id and add opening_hours, price_level etc.
    """
    enriched = []
    for a in attractions:
        place = google_find_place(a["name"], city)
        if not place:
            a["google_place_id"] = None
            enriched.append(a)
            continue

        place_id = place["place_id"]
        details = google_place_details(place_id)
        if not details:
            a["google_place_id"] = place_id
            enriched.append(a)
            continue

        a["google_place_id"] = place_id
        a["opening_hours"] = details.get("opening_hours", {})
        a["google_price_level"] = details.get("price_level")
        a["location"] = details.get("geometry", {}).get("location")
        a["google_rating"] = details.get("rating")
        a["google_user_ratings_total"] = details.get("user_ratings_total")
        enriched.append(a)
        time.sleep(0.2)  # polite delay
    return enriched


###############################################################################
# LOCATION-SCOUT AGENT (Llama prompt)
###############################################################################

def location_scout_agent(city: str, refined_profile: str, constraints: Dict) -> List[dict]:
    """
    Ask Llama to propose ~10 attractions in JSON.
    """
    with_children = constraints.get("with_children")
    with_disabled = constraints.get("with_disabled")
    budget = constraints.get("budget")

    prompt = f"""
You are a travel planning assistant.
User wants a 3-day trip to {city}.

REFINED USER PROFILE:
{refined_profile}

CONSTRAINTS:
- With children: {with_children}
- With disabled traveler: {with_disabled}
- Budget (entire trip, group): {budget}

Propose 10 candidate attractions in {city} that match the user's interests and constraints.

For each attraction, output an object with:
- name
- short_description
- approx_price_per_person (number)
- tags: an array of strings, including some of: "museum", "outdoor", "nightlife", "kid_friendly", "wheelchair_friendly", "food", "viewpoint", etc.
- reason_for_user: one sentence explaining why this matches the profile.

Return ONLY a JSON array of 10 such objects.
"""

    raw = call_llama(prompt)
    json_str = extract_json(raw)
    attractions = json.loads(json_str)
    return attractions


###############################################################################
# BUDGET AGENT (deterministic)
###############################################################################

def estimate_price(a: dict) -> float:
    if a.get("approx_price_per_person") is not None:
        return float(a["approx_price_per_person"])
    # fallback from google_price_level if exists
    level = a.get("google_price_level")
    if level is None:
        return 30.0  # default guess
    # simple mapping: 0-4 -> 0-100
    return (level + 1) * 20.0


def budget_agent(attractions: List[dict], budget: float, days: int) -> List[dict]:
    """
    Very simple: try to keep total attractions cost within budget.
    Assume user visits ~3 attractions per day (so up to days*3).
    """
    max_attractions = days * 3
    # sort by price ascending to favor cheaper
    for a in attractions:
        a["final_price_estimate"] = estimate_price(a)
    attractions_sorted = sorted(attractions, key=lambda x: x["final_price_estimate"])

    selected = []
    total_cost = 0.0
    for a in attractions_sorted:
        if len(selected) >= max_attractions:
            break
        new_total = total_cost + a["final_price_estimate"]
        if new_total <= budget:
            selected.append(a)
            total_cost = new_total

    return selected


###############################################################################
# SCHEDULER AGENT (deterministic, simple)
###############################################################################

def scheduler_agent(attractions: List[dict], days: int) -> Dict:
    """
    Simple scheduling: distribute attractions across days and timeslots.
    We ignore real distances and just do round-robin for trial.
    """
    slots = ["morning", "afternoon", "evening"]
    itinerary = {f"day{i+1}": {s: [] for s in slots} for i in range(days)}

    i = 0
    for a in attractions:
        day_idx = i // len(slots)
        slot_idx = i % len(slots)
        if day_idx >= days:
            break
        day_key = f"day{day_idx+1}"
        itinerary[day_key][slots[slot_idx]].append(a)
        i += 1

    return itinerary


###############################################################################
# EVALUATION AGENT (optional Llama-based scorer)
###############################################################################

def evaluation_agent(profile: Dict, itinerary: Dict) -> Dict:
    """
    Ask Llama to rate the itinerary 1–5 on a few dimensions.
    """
    prompt = f"""
You are an impartial travel expert.

USER PROFILE:
{json.dumps(profile, indent=2)}

ITINERARY:
{json.dumps(itinerary, indent=2)}

Rate from 1 to 5 (integers) on:
- interest_match
- budget_realism
- logistics
- suitability_for_constraints (children/disabled if present)

Return ONLY JSON:
{{
  "interest_match": 1-5,
  "budget_realism": 1-5,
  "logistics": 1-5,
  "suitability_for_constraints": 1-5,
  "comment": "short explanation"
}}
"""

    raw = call_llama(prompt)
    json_str = extract_json(raw)
    scores = json.loads(json_str)
    return scores


###############################################################################
# FULL PIPELINE
###############################################################################

def run_pipeline():
    # 1) Interest & city dialogue
    profile = run_interest_and_city_dialogue(budget=DEFAULT_BUDGET)

    # 2) Location scout with Llama
    city = profile["chosen_city"]
    refined_profile = profile["refined_profile"]
    constraints = profile["constraints"]
    print("\n--- Generating attractions ---")
    attractions = location_scout_agent(city, refined_profile, constraints)

    # 3) Enrich with Google Places (opening hours, etc.)
    print("\n--- Enriching with Google Places (if API key set) ---")
    attractions_enriched = enrich_with_google_places(attractions, city)

    # 4) Budget filtering
    print("\n--- Applying budget filter ---")
    affordable_attractions = budget_agent(attractions_enriched, DEFAULT_BUDGET, DEFAULT_DAYS)

    # 5) Scheduling
    print("\n--- Building itinerary ---")
    itinerary = scheduler_agent(affordable_attractions, DEFAULT_DAYS)
    print(json.dumps(itinerary, indent=2))

    # 6) Evaluation (optional)
    print("\n--- Evaluating itinerary (LLM-based) ---")
    evaluation = evaluation_agent(profile, itinerary)
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    run_pipeline()
