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

# Llama model served by Ollama
LLAMA_MODEL_NAME = "llama3"  # change if you use a different tag, e.g. "llama3:8b"

# Google Places API key (optional – leave as is if you don't want real enrichment)
GOOGLE_API_KEY = "AIzaSyANxjWJzD0BetncqmnBp069mfnawH9xO6g"

# Simple trial defaults
DEFAULT_BUDGET = 600.0  # whole-trip budget for the group
DEFAULT_DAYS = 3
DEFAULT_PEOPLE = 2

###############################################################################
# LLM CALL – OLLAMA
###############################################################################

def call_llama(prompt: str, model: str = LLAMA_MODEL_NAME) -> str:
    """
    Calls a local Ollama model via HTTP.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["response"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama server. Please make sure:\n"
            "1. Ollama is installed (https://ollama.ai/)\n"
            "2. Server is running: 'ollama serve'\n"
            "3. Model is pulled: 'ollama pull llama3'"
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500:
            raise ConnectionError(
                "Ollama server error. The model might not be available.\n"
                "Please run: 'ollama pull llama3' and try again."
            )
        else:
            raise

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
    """Very simple sentence splitter."""
    parts = re.split(r"[.!?]\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def classify_slot(sentence_emb: np.ndarray) -> str:
    """Assign a sentence embedding to the best slot based on cosine similarity."""
    best_slot, best_score = None, -1.0
    for slot, label_emb in SLOT_EMBS.items():
        score = float(sentence_emb @ label_emb)  # cosine if normalized
        if score > best_score:
            best_score, best_slot = score, slot
    return best_slot


SIM_UPDATE_THRESHOLD = 0.75


def update_state_with_message(state: PreferenceState, user_msg: str) -> PreferenceState:
    """Use MiniLM embeddings to update the semantic preference state each turn."""
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
            # refinement / override of existing info in this slot
            best_sn.text = s
            best_sn.embedding = emb
        else:
            new_snippets.append(PreferenceSnippet(text=s, embedding=emb, slot=slot))

    state.snippets.extend(new_snippets)

    # rebuild slot texts from snippets
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
    """Create a compact semantic summary for the LLM."""
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

def extract_json(text: str) -> str:
    """
    Robust JSON extraction that handles various LLM output formats.
    """
    import re
    
    # Clean the text
    text = text.strip()
    
    # Common patterns where LLMs add explanations
    patterns = [
        r'```json\s*(.*?)\s*```',  # ```json {...} ```
        r'```\s*(.*?)\s*```',      # ``` {...} ```
        r'JSON:\s*(.*?)(?:\n\n|\Z)',  # JSON: {...}
        r'Here.*?JSON[:\s]*(.*?)(?:\n\n|\Z)',  # Here is the JSON: {...}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    
    # Try to find any JSON object
    brace_count = 0
    start_index = -1
    chars = []
    
    for i, char in enumerate(text):
        if char == '{':
            brace_count += 1
            if start_index == -1:
                start_index = i
        if char == '}':
            brace_count -= 1
        
        if start_index != -1:
            chars.append(char)
        
        if brace_count == 0 and start_index != -1:
            candidate = ''.join(chars)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Reset and continue searching
                start_index = -1
                chars = []
    
    # If we get here, try the simple approach
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError as e:
            raise ValueError(f"Found JSON-like structure but it's invalid: {e}\nRaw text: {text}")
    
    raise ValueError(f"No valid JSON found in LLM output. Raw output:\n{text}")


def llama_next_turn(state: PreferenceState, last_user_msg: str, budget: float, people: int, days: int) -> Dict:
    """Ask Llama whether to ask another question or finalize city & profile."""
    profile_summary = build_profile_summary(state)

    # Check if we have critical information - budget is already provided!
    has_budget = state.slots['budget'] and any(word in state.slots['budget'].lower() for word in ['budget', 'price', 'cost', 'money', 'euro', '$'])
    has_people_info = state.slots['constraints'] and any(word in state.slots['constraints'].lower() for word in ['people', 'person', 'group', 'family', 'children', 'kids'])
    
    prompt = f"""
You are the "Interest-Refinement & City-Matchmaker Agent" in a travel planning system.

We are interviewing a user to plan a {days}-day city trip in Europe for {people} people with a total budget of {budget} EUR.

CRITICAL INFORMATION ALREADY COLLECTED:
- Total budget: {budget} EUR (already confirmed)
- Number of people: {people} (already confirmed) 
- Trip duration: {days} days (already confirmed)

CURRENT SEMANTIC PROFILE (built from all previous messages):
{profile_summary}

Most recent user message:
\"\"\"{last_user_msg}\"\"\"

ADDITIONAL INFORMATION STATUS:
- Budget preferences discussed in conversation: {'YES' if has_budget else 'NO'}
- Group details discussed in conversation: {'YES' if has_people_info else 'NO'}

Your tasks:
1. DO NOT ask about budget, number of people, or trip duration - these are already confirmed above.
2. Focus on understanding their interests, preferences, and any constraints.
3. If we have enough interest information, recommend a European city and finalize.
4. If not, ask ONE clarifying question about activities, pace, food preferences, or constraints.

Return ONLY valid JSON with this structure:
{{
  "action": "ask_question" or "finalize",
  "question": "string (if action == 'ask_question', else empty)",
  "refined_profile": "short natural language summary of what the user wants",
  "chosen_city": "string or null if not ready",
  "constraints": {{
    "with_children": true or false or null,
    "with_disabled": true or false or null,
    "budget": {budget},
    "people": {people}
  }},
  "travel_style": "slow" or "medium" or "fast" or null
}}

IMPORTANT: 
- DO NOT ask about budget, people count, or trip duration - these are already known
- Focus on interests, activities, pace, food, and constraints
- Maximum 5 questions total - be efficient!
- Return ONLY the JSON object, no additional text
"""

    raw = call_llama(prompt)
    
    json_str = extract_json(raw)
    
    try:
        data = json.loads(json_str)
        
        # Validate required fields
        required_fields = ["action", "question", "refined_profile", "chosen_city", "constraints", "travel_style"]
        
        for field in required_fields:
            if field not in data:
                print(f"Warning: Missing field '{field}' in LLM response. Adding default.")
        
        # Ensure action is valid
        if data.get("action") not in ["ask_question", "finalize"]:
            print(f"Warning: Invalid action '{data.get('action')}'. Defaulting to 'ask_question'.")
            data["action"] = "ask_question"
        
        # Ensure constraints is a dict with required fields
        if not isinstance(data.get("constraints"), dict):
            data["constraints"] = {}
        
        constraint_fields = ["with_children", "with_disabled", "budget", "people"]
        for field in constraint_fields:
            if field not in data["constraints"]:
                if field == "budget":
                    data["constraints"][field] = budget
                elif field == "people":
                    data["constraints"][field] = people
                else:
                    data["constraints"][field] = None
        
        # Ensure other fields exist
        if "question" not in data:
            data["question"] = ""
        if "refined_profile" not in data:
            data["refined_profile"] = build_profile_summary(state)
        if "chosen_city" not in data:
            data["chosen_city"] = None
        if "travel_style" not in data:
            data["travel_style"] = None
            
        return data
        
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing LLM response: {e}")
        print("Using fallback response...")
        
        # Smart fallback - don't ask about budget since it's already provided
        if not has_people_info:
            question = f"Are you traveling alone or with others? Any children or people with mobility needs?"
        elif not state.slots['activities']:
            question = "What specific types of ancient buildings and museums interest you most?"
        else:
            question = "What pace do you prefer for your trip - relaxed, moderate, or packed with activities?"
        
        return {
            "action": "ask_question",
            "question": question,
            "refined_profile": f"User interested in {state.slots['activities']}. Budget: {budget} EUR for {people} people.",
            "chosen_city": None,
            "constraints": {
                "with_children": False,
                "with_disabled": False,
                "budget": budget,
                "people": people
            },
            "travel_style": "medium"
        }
def run_interest_and_city_dialogue(budget: float, people: int, days: int) -> Dict:
    """
    Efficient CLI dialogue with maximum 5 questions.
    """
    print("🚀 [INTEREST-REFINEMENT AGENT] Starting preference collection...")
    
    state = PreferenceState()
    max_turns = 5  # Maximum 5 questions

    print("Welcome to the travel planner!")
    print(f"Planning a {days}-day trip for {people} people with {budget} EUR budget")
    
    # Pre-populate the state with the known budget information
    initial_info = f"Budget is {budget} EUR for {people} people for {days} days. "
    user_msg = input("Tell me about your trip preferences (interests, activities, etc.): ")
    
    # Combine the known info with user preferences
    full_initial_msg = initial_info + user_msg
    state = update_state_with_message(state, full_initial_msg)

    turn_count = 0
    
    while turn_count < max_turns:
        turn_count += 1
        print(f"\n📝 [INTEREST-REFINEMENT AGENT] Turn {turn_count}/{max_turns}")
        
        try:
            llama_output = llama_next_turn(state, user_msg, budget, people, days)
            action = llama_output["action"]

            if action == "ask_question":
                q = llama_output["question"]
                # Check if the question is about budget and skip it
                if any(word in q.lower() for word in ['budget', 'price', 'cost', 'money', 'how much']):
                    print(f"🔄 Skipping budget question (already known: {budget} EUR)")
                    # Force finalization or ask a different question
                    if turn_count >= 3:  # If we've had several turns, just finalize
                        print("🔄 Multiple budget questions detected, finalizing with current info...")
                        action = "finalize"
                    else:
                        # Ask a different question instead
                        print(f"\nAgent: What type of ancient buildings and museums interest you most?")
                        user_msg = input("You: ")
                        state = update_state_with_message(state, user_msg)
                        continue
                else:
                    print(f"\nAgent: {q}")
                    user_msg = input("You: ")
                    state = update_state_with_message(state, user_msg)

            if action == "finalize":
                profile = {
                    "refined_profile": llama_output["refined_profile"],
                    "chosen_city": llama_output["chosen_city"],
                    "constraints": llama_output["constraints"],
                    "travel_style": llama_output["travel_style"],
                    "semantic_profile_slots": state.slots,
                    "interest_embedding": state.global_embedding.tolist()
                    if state.global_embedding is not None else None,
                }
                print("\n✅ [INTEREST-REFINEMENT AGENT] Profile finalized!")
                print(json.dumps(profile, indent=2))
                return profile

        except Exception as e:
            print(f"Error in dialogue turn: {e}")
            if turn_count < max_turns:
                print("Asking a follow-up question to continue...")
                print(f"\nAgent: What other activities or preferences would you like to share?")
                user_msg = input("You: ")
                state = update_state_with_message(state, user_msg)
            else:
                break

    # If we reach max turns, finalize with what we have
    print(f"\n⚠️ [INTEREST-REFINEMENT AGENT] Reached maximum questions ({max_turns}). Finalizing...")
    profile = {
        "refined_profile": build_profile_summary(state),
        "chosen_city": "Rome",  # Default for European travel with ancient buildings
        "constraints": {
            "with_children": "children" in state.slots['constraints'].lower() if state.slots['constraints'] else False,
            "with_disabled": any(word in state.slots['constraints'].lower() for word in ['disabled', 'mobility', 'wheelchair']) if state.slots['constraints'] else False,
            "budget": budget,
            "people": people
        },
        "travel_style": "medium",
        "semantic_profile_slots": state.slots,
        "interest_embedding": state.global_embedding.tolist()
        if state.global_embedding is not None else None,
    }
    print("\n✅ [INTEREST-REFINEMENT AGENT] Profile finalized with available information!")
    print(json.dumps(profile, indent=2))
    return profile


###############################################################################
# GOOGLE PLACES API (OPENING HOURS & DETAILS)
###############################################################################

def google_find_place(name: str, city: str) -> Optional[dict]:
    """
    Uses 'Find Place From Text' to get a place_id.
    Returns None if GOOGLE_API_KEY is not set or no result.
    """
    if GOOGLE_API_KEY == "YOUR_GOOGLE_PLACES_API_KEY_HERE":
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
    For each attraction: look up place_id and add opening_hours, price_level, etc.
    If no API key, returns attractions unchanged except for google_place_id=None.
    """
    print(f"🔍 [GOOGLE PLACES AGENT] Enriching {len(attractions)} attractions with real-time data...")
    
    enriched = []
    for i, a in enumerate(attractions):
        print(f"   Enriching attraction {i+1}/{len(attractions)}: {a.get('name', 'Unknown')}")
        
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
        time.sleep(0.1)  # Reduced delay for faster processing
    
    print("✅ [GOOGLE PLACES AGENT] Enrichment completed!")
    return enriched


###############################################################################
# LOCATION-SCOUT AGENT (LLAMA)
###############################################################################

def location_scout_agent(city: str, refined_profile: str, constraints: Dict) -> List[dict]:
    """
    Ask Llama to propose ~10 attractions in JSON in the chosen city.
    """
    print(f"🗺️ [LOCATION-SCOUT AGENT] Generating attractions for {city}...")
    
    with_children = constraints.get("with_children")
    with_disabled = constraints.get("with_disabled")
    budget = constraints.get("budget")
    people = constraints.get("people")

    prompt = f"""
You are a travel planning assistant.
The user wants a trip to {city}.

REFINED USER PROFILE:
{refined_profile}

CONSTRAINTS:
- With children: {with_children}
- With disabled traveler: {with_disabled}
- Budget (entire trip, group): {budget} EUR
- People: {people}

Propose EXACTLY 10 candidate attractions in {city} that match the user's interests and constraints.

For each attraction, output an object with:
- name
- short_description
- approx_price_per_person (number in EUR)
- tags: an array of strings, including some of: "museum", "outdoor", "nightlife",
        "kid_friendly", "wheelchair_friendly", "food", "viewpoint", "historical", etc.
- reason_for_user: one sentence explaining why this matches the profile.

Return ONLY a JSON array of EXACTLY 10 objects (no extra text).
Example format:
[
  {{
    "name": "Louvre Museum",
    "short_description": "World's largest art museum",
    "approx_price_per_person": 17,
    "tags": ["museum", "art", "wheelchair_friendly"],
    "reason_for_user": "Perfect for art lovers with extensive historical collections"
  }},
  {{...}},  // 9 more attractions
]
"""

    raw = call_llama(prompt)
    
    json_str = extract_json(raw)
    
    try:
        attractions = json.loads(json_str)
        
        # Validate that we got a list
        if not isinstance(attractions, list):
            print(f"Warning: Expected list but got {type(attractions)}. Converting...")
            if isinstance(attractions, dict):
                attractions = [attractions]  # Convert single dict to list
            else:
                attractions = []
        
        # If we have too few attractions, use fallback
        if len(attractions) < 5:
            print(f"Warning: Only {len(attractions)} attractions generated. Using fallback.")
            fallback_attractions = get_fallback_attractions(city)
            # Combine what we have with fallback
            combined_attractions = attractions + fallback_attractions
            # Remove duplicates by name
            seen_names = set()
            unique_attractions = []
            for attr in combined_attractions:
                name = attr.get('name', '')
                if name not in seen_names:
                    seen_names.add(name)
                    unique_attractions.append(attr)
            attractions = unique_attractions[:10]  # Take up to 10
        
        # Validate each attraction has the required fields
        validated_attractions = []
        for i, attr in enumerate(attractions):
            if not isinstance(attr, dict):
                print(f"Warning: Skipping attraction {i} - not a dictionary: {attr}")
                continue
                
            # Ensure required fields exist
            if "name" not in attr:
                attr["name"] = f"Attraction {i+1}"
            if "short_description" not in attr:
                attr["short_description"] = "No description provided"
            if "approx_price_per_person" not in attr:
                # Set reasonable default based on attraction type
                if any(tag in str(attr.get('tags', [])).lower() for tag in ['museum', 'historical']):
                    attr["approx_price_per_person"] = 15.0
                elif any(tag in str(attr.get('tags', [])).lower() for tag in ['landmark', 'viewpoint']):
                    attr["approx_price_per_person"] = 20.0
                else:
                    attr["approx_price_per_person"] = 10.0
            if "tags" not in attr:
                attr["tags"] = ["sightseeing"]
            if "reason_for_user" not in attr:
                attr["reason_for_user"] = "Matches user interests"
                
            validated_attractions.append(attr)
        
        print(f"✅ [LOCATION-SCOUT AGENT] Generated {len(validated_attractions)} attractions")
        return validated_attractions
        
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ [LOCATION-SCOUT AGENT] Error parsing attractions: {e}")
        print("Using fallback attractions...")
        return get_fallback_attractions(city)


def get_fallback_attractions(city: str) -> List[dict]:
    """
    Provide comprehensive fallback attractions when LLM fails to return valid JSON.
    """
    print(f"🔄 [LOCATION-SCOUT AGENT] Using comprehensive fallback attractions for {city}")
    
    comprehensive_fallback = {
        "Rome": [
            {
                "name": "Colosseum",
                "short_description": "Ancient Roman amphitheater, iconic symbol of Rome",
                "approx_price_per_person": 16.0,
                "tags": ["historical", "ancient", "architecture", "wheelchair_friendly"],
                "reason_for_user": "Perfect for ancient building enthusiasts with rich historical significance"
            },
            {
                "name": "Roman Forum",
                "short_description": "Ancient Roman government center with ruins and temples",
                "approx_price_per_person": 12.0,
                "tags": ["historical", "ancient", "archaeological", "outdoor"],
                "reason_for_user": "Extensive ancient ruins perfect for history lovers"
            },
            {
                "name": "Pantheon",
                "short_description": "Ancient Roman temple with magnificent dome",
                "approx_price_per_person": 0.0,
                "tags": ["historical", "architecture", "religious", "free"],
                "reason_for_user": "Well-preserved ancient building with incredible architecture"
            },
            {
                "name": "Vatican Museums",
                "short_description": "Extensive art collections including Sistine Chapel",
                "approx_price_per_person": 17.0,
                "tags": ["museum", "art", "religious", "historical"],
                "reason_for_user": "World-class museum with historical and artistic treasures"
            },
            {
                "name": "St. Peter's Basilica",
                "short_description": "Renaissance church with dome and religious art",
                "approx_price_per_person": 0.0,
                "tags": ["religious", "architecture", "historical", "free"],
                "reason_for_user": "Magnificent architecture and historical significance"
            },
            {
                "name": "Trevi Fountain",
                "short_description": "Baroque fountain and famous landmark",
                "approx_price_per_person": 0.0,
                "tags": ["landmark", "baroque", "sculpture", "free"],
                "reason_for_user": "Iconic Baroque architecture and sculpture"
            },
            {
                "name": "Palatine Hill",
                "short_description": "Ancient hill with imperial palace ruins",
                "approx_price_per_person": 12.0,
                "tags": ["historical", "archaeological", "outdoor", "ancient"],
                "reason_for_user": "Ancient imperial ruins with panoramic views"
            },
            {
                "name": "Borghese Gallery",
                "short_description": "Art museum in former villa with Renaissance works",
                "approx_price_per_person": 13.0,
                "tags": ["museum", "art", "renaissance", "sculpture"],
                "reason_for_user": "Art museum in historical villa building"
            },
            {
                "name": "Castel Sant'Angelo",
                "short_description": "Ancient Roman mausoleum turned fortress museum",
                "approx_price_per_person": 15.0,
                "tags": ["historical", "museum", "architecture", "fortress"],
                "reason_for_user": "Historical building with multiple architectural eras"
            },
            {
                "name": "Baths of Caracalla",
                "short_description": "Ancient Roman public bath complex ruins",
                "approx_price_per_person": 8.0,
                "tags": ["historical", "ancient", "archaeological", "outdoor"],
                "reason_for_user": "Impressive ancient Roman bath complex ruins"
            }
        ],
        "Paris": [
            {
                "name": "Louvre Museum",
                "short_description": "World's largest art museum in historic palace",
                "approx_price_per_person": 17.0,
                "tags": ["museum", "art", "historical", "palace", "wheelchair_friendly"],
                "reason_for_user": "Historical palace building with world's best art collection"
            },
            {
                "name": "Eiffel Tower",
                "short_description": "Iconic iron tower offering city views",
                "approx_price_per_person": 25.0,
                "tags": ["landmark", "architecture", "viewpoint", "historical"],
                "reason_for_user": "Iconic architectural landmark with historical significance"
            },
            # Add more Paris attractions...
        ],
        "Athens": [
            {
                "name": "Acropolis",
                "short_description": "Ancient citadel with Parthenon and temples",
                "approx_price_per_person": 20.0,
                "tags": ["historical", "ancient", "architecture", "archaeological"],
                "reason_for_user": "Most famous ancient Greek building complex"
            },
            {
                "name": "Acropolis Museum",
                "short_description": "Modern museum showcasing Acropolis artifacts",
                "approx_price_per_person": 10.0,
                "tags": ["museum", "archaeological", "modern", "wheelchair_friendly"],
                "reason_for_user": "Modern museum dedicated to ancient Greek artifacts"
            },
            # Add more Athens attractions...
        ]
    }
    
    # Return city-specific attractions or generic ancient building attractions
    if city in comprehensive_fallback:
        return comprehensive_fallback[city]
    else:
        # Generic ancient building attractions for any European city
        return [
            {
                "name": f"{city} Historical Museum",
                "short_description": "Local museum showcasing city history and ancient artifacts",
                "approx_price_per_person": 12.0,
                "tags": ["museum", "historical", "cultural", "wheelchair_friendly"],
                "reason_for_user": "Perfect for understanding local history and ancient cultures"
            },
            {
                "name": f"{city} Old Town",
                "short_description": "Historic district with ancient buildings and architecture",
                "approx_price_per_person": 0.0,
                "tags": ["historical", "architecture", "walking", "free", "outdoor"],
                "reason_for_user": "Free exploration of ancient buildings and historical architecture"
            },
            {
                "name": f"{city} Archaeological Site",
                "short_description": "Ancient ruins and archaeological excavations",
                "approx_price_per_person": 8.0,
                "tags": ["archaeological", "historical", "ancient", "outdoor"],
                "reason_for_user": "Direct experience with ancient ruins and historical sites"
            },
            {
                "name": f"{city} Cathedral",
                "short_description": "Historical religious building with ancient architecture",
                "approx_price_per_person": 5.0,
                "tags": ["religious", "architecture", "historical", "cultural"],
                "reason_for_user": "Ancient religious architecture with historical significance"
            },
            {
                "name": f"{city} Castle/Fortress",
                "short_description": "Ancient defensive structure with historical importance",
                "approx_price_per_person": 10.0,
                "tags": ["historical", "architecture", "fortress", "outdoor"],
                "reason_for_user": "Ancient defensive architecture and historical military site"
            }
        ]
###############################################################################
# BUDGET AGENT (DETERMINISTIC)
###############################################################################

def estimate_price(a: dict) -> float:
    """Estimate attraction price using approx_price_per_person or Google price level."""
    if a.get("approx_price_per_person") is not None:
        try:
            return float(a["approx_price_per_person"])
        except (TypeError, ValueError):
            pass

    level = a.get("google_price_level")
    if level is None:
        return 20.0  # default guess
    # simple mapping: 0-4 -> ~15-80
    return (level + 1) * 15.0


def budget_agent(attractions: List[dict], budget: float, days: int, people: int) -> List[dict]:
    """
    Fast budget filter with per-person calculation.
    """
    print(f"💰 [BUDGET AGENT] Filtering {len(attractions)} attractions for {people} people with {budget} EUR budget...")
    
    max_attractions = days * 3  # 3 attractions per day

    # Calculate total prices for all attractions
    for a in attractions:
        a["final_price_estimate"] = estimate_price(a) * people  # Multiply by number of people
    
    # Sort by cheapest first
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
        else:
            break  # Stop early since list is sorted by price

    print(f"✅ [BUDGET AGENT] Selected {len(selected)} attractions within budget. Total cost: {total_cost:.2f} EUR")
    return selected


###############################################################################
# SCHEDULER AGENT (DETERMINISTIC, FAST)
###############################################################################

def scheduler_agent(attractions: List[dict], days: int) -> Dict:
    """
    Fast scheduling: distribute attractions across days and timeslots.
    """
    print(f"📅 [SCHEDULER AGENT] Creating {days}-day itinerary with {len(attractions)} attractions...")
    
    slots = ["morning", "afternoon", "evening"]
    itinerary = {f"day{i+1}": {s: [] for s in slots} for i in range(days)}

    # Simple round-robin distribution
    for i, attraction in enumerate(attractions):
        day_idx = i // len(slots)
        if day_idx >= days:  # Don't exceed planned days
            break
        slot_idx = i % len(slots)
        day_key = f"day{day_idx+1}"
        itinerary[day_key][slots[slot_idx]].append(attraction)

    print("✅ [SCHEDULER AGENT] Itinerary created successfully!")
    return itinerary


###############################################################################
# EVALUATION AGENT (LLAMA-BASED)
###############################################################################

def evaluation_agent(profile: Dict, itinerary: Dict) -> Dict:
    """
    Ask Llama to score the itinerary on several dimensions (1–5).
    """
    print("📊 [EVALUATION AGENT] Evaluating itinerary quality...")
    
    prompt = f"""
You are an impartial travel expert.

USER PROFILE:
{json.dumps(profile, indent=2)}

ITINERARY:
{json.dumps(itinerary, indent=2)}

Rate from 1 to 5 (integers) on:
- interest_match: How well the itinerary matches user interests
- budget_realism: How realistic the budget allocation is
- logistics: How well the schedule flows
- suitability_for_constraints: How well it accommodates constraints (children/disabled if present)

Return ONLY JSON:
{{
  "interest_match": 1-5,
  "budget_realism": 1-5,
  "logistics": 1-5,
  "suitability_for_constraints": 1-5,
  "comment": "short explanation"
}}
Be fast and concise.
"""

    raw = call_llama(prompt)
    json_str = extract_json(raw)
    scores = json.loads(json_str)
    
    print("✅ [EVALUATION AGENT] Evaluation completed!")
    return scores


###############################################################################
# FULL PIPELINE
###############################################################################

def run_pipeline():
    print("🎯 [MAIN PIPELINE] Starting travel planning pipeline...")
    
    try:
        # Get basic information first
        print("\n💬 Collecting basic information...")
        budget = float(input(f"Enter total budget in EUR (default: {DEFAULT_BUDGET}): ") or DEFAULT_BUDGET)
        people = int(input(f"Enter number of people (default: {DEFAULT_PEOPLE}): ") or DEFAULT_PEOPLE)
        days = int(input(f"Enter number of days (default: {DEFAULT_DAYS}): ") or DEFAULT_DAYS)

        # 1) Interest & city dialogue (semantic layer + Llama interview)
        print(f"\n🌟 Starting planning for {people} people, {days} days, {budget} EUR budget")
        profile = run_interest_and_city_dialogue(budget, people, days)

        # 2) Location scout with Llama
        city = profile["chosen_city"]
        refined_profile = profile["refined_profile"]
        constraints = profile["constraints"]
        attractions = location_scout_agent(city, refined_profile, constraints)
        
        if not attractions:
            print("❌ No attractions generated. Using fallback.")
            attractions = get_fallback_attractions(city)

        # 3) Enrich with Google Places (opening hours, etc.) – optional if key missing
        attractions_enriched = enrich_with_google_places(attractions, city)

        # 4) Budget filtering
        affordable_attractions = budget_agent(attractions_enriched, budget, days, people)
        
        if not affordable_attractions:
            print("⚠️ No attractions fit the budget. Using top attractions with budget warning.")
            affordable_attractions = attractions_enriched[:days * 2]  # Just take a few

        # 5) Scheduling
        itinerary = scheduler_agent(affordable_attractions, days)
        print("\n📋 FINAL ITINERARY:")
        print(json.dumps(itinerary, indent=2))

        # 6) Evaluation
        evaluation = evaluation_agent(profile, itinerary)
        print("\n⭐ ITINERARY EVALUATION:")
        print(json.dumps(evaluation, indent=2))
        
        print("\n🎉 [MAIN PIPELINE] Travel planning completed successfully!")
        
    except Exception as e:
        print(f"❌ [MAIN PIPELINE] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline()