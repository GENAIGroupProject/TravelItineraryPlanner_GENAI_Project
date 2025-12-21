import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from ui.style import inject_global_css
from ui.state import ensure_session_state
from ui.sidebar import render_sidebar
from main import TravelPlanner
from config import Config

# Generate attractions immediately so we can display them right after LLM returns
from agents.location_scout_agent import LocationScoutAgent


# -------------------------
# Bootstrapping
# -------------------------
inject_global_css()
ensure_session_state()
render_sidebar()

# Session defaults
if "dialogue_stage" not in st.session_state:
    st.session_state.dialogue_stage = "idle"

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

if "chat" not in st.session_state:
    st.session_state.chat = []

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

if "preview" not in st.session_state:
    st.session_state.preview = {
        "ready": False,
        "shown": False,
        "auto_redirect": True,
        "redirect_in_sec": 2.0,
        "data": None,
        "city": None,
        "city_photo_url": None,
        "attractions_10": [],
        "refined_attractions": [],
    }

# Cache attractions immediately after LLM returns
if "llm_attractions_10" not in st.session_state:
    st.session_state.llm_attractions_10 = []


# -------------------------
# Planner getter
# -------------------------
def get_planner() -> TravelPlanner:
    if st.session_state.get("planner") is None:
        st.session_state.planner = TravelPlanner()
    return st.session_state.planner


# -------------------------
# UI header
# -------------------------
st.markdown('<h1 class="main-header">✈️ Plan Your Trip</h1>', unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def add_msg(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})


def stream_words(text: str, delay: float = 0.02):
    for w in (text or "").split():
        yield w + " "
        time.sleep(delay)


def normalize_question(q: str) -> str:
    return (q or "").strip()


def get_profile_city() -> Optional[str]:
    profile = st.session_state.get("profile")
    if profile is None:
        return None

    if hasattr(profile, "chosen_city"):
        city = getattr(profile, "chosen_city", None)
        return (str(city).strip() if city else None)

    if isinstance(profile, dict):
        city = profile.get("chosen_city") or profile.get("city")
        return (str(city).strip() if city else None)

    return None


def get_profile_refined_profile_text() -> str:
    profile = st.session_state.get("profile")
    if profile is None:
        return ""

    if hasattr(profile, "refined_profile"):
        return (getattr(profile, "refined_profile", "") or "").strip()

    if isinstance(profile, dict):
        return (profile.get("refined_profile") or "").strip()

    return ""


def get_profile_constraints_dict() -> Dict[str, Any]:
    profile = st.session_state.get("profile")
    if profile is None:
        return {}

    if hasattr(profile, "constraints"):
        c = getattr(profile, "constraints", None)
        if c is None:
            return {}
        if hasattr(c, "dict"):
            return c.dict()
        if hasattr(c, "__dict__"):
            return dict(c.__dict__)
        if isinstance(c, dict):
            return c

    if isinstance(profile, dict):
        c = profile.get("constraints") or {}
        return c if isinstance(c, dict) else {}

    return {}


def ensure_first_question_exists():
    if st.session_state.get("dialogue_stage") != "refine":
        return

    prefs = (st.session_state.get("user_inputs") or {}).get("preferences", "").strip()
    if not prefs:
        return

    if st.session_state.get("chat"):
        return

    planner = get_planner()
    step = planner.start_refinement(prefs)
    q = normalize_question(step.get("question", ""))

    if not q:
        step = planner.start_refinement(prefs)
        q = normalize_question(step.get("question", ""))

    if q:
        add_msg("assistant", q)
        st.session_state.questions_asked = max(1, st.session_state.get("questions_asked", 0))


# -------------------------
# Google Places city photo helpers
# -------------------------
def _safe_get_google_api_key() -> Optional[str]:
    return getattr(Config, "GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")


def get_city_place_photo_url(city: str, maxwidth: int = 1200) -> Optional[str]:
    api_key = _safe_get_google_api_key()
    if not api_key or not city:
        return None

    try:
        find_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        find_params = {
            "input": city,
            "inputtype": "textquery",
            "fields": "place_id",
            "key": api_key,
        }
        r = requests.get(find_url, params=find_params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("candidates"):
            return None
        place_id = data["candidates"][0].get("place_id")
        if not place_id:
            return None

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {"place_id": place_id, "fields": "photos", "key": api_key}
        r2 = requests.get(details_url, params=details_params, timeout=10)
        r2.raise_for_status()
        details = r2.json().get("result", {})
        photos = details.get("photos") or []
        if not photos:
            return None

        photo_ref = photos[0].get("photo_reference")
        if not photo_ref:
            return None

        return (
            "https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth={maxwidth}&photoreference={photo_ref}&key={api_key}"
        )
    except Exception:
        return None


# -------------------------
# Normalize + render attractions (WITH PRICE)
# -------------------------
def normalize_attractions_list(items: Any) -> List[Dict[str, Any]]:
    """
    Converts Attraction objects (pydantic/dataclass) into dicts.
    Ensures approx_price_per_person survives. :contentReference[oaicite:3]{index=3}
    """
    if not items:
        return []
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        else:
            if hasattr(item, "dict"):
                out.append(item.dict())
            elif hasattr(item, "__dict__"):
                out.append(dict(item.__dict__))
            else:
                out.append({"name": str(item)})
    return out


def _safe_price(item: Dict[str, Any]) -> Optional[float]:
    """
    Price field is approx_price_per_person per your agent prompt + model. :contentReference[oaicite:4]{index=4}
    """
    raw = item.get("approx_price_per_person")
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def render_attractions_with_prices(attractions: List[Dict[str, Any]]):
    if not attractions:
        st.error(
            "No attractions were returned to the UI.\n\n"
            "If the terminal shows them, it means they exist in the agent, but they aren't reaching this UI state."
        )
        return

    for idx, item in enumerate(attractions[:10], start=1):
        name = (item.get("name") or item.get("title") or f"Attraction {idx}").strip()
        desc = (item.get("short_description") or item.get("description") or "").strip()
        tags = item.get("tags") or []
        reason = (item.get("reason_for_user") or item.get("reason") or "").strip()

        price = _safe_price(item)
        price_txt = f"{price:.0f}€ / person" if price is not None else "—"

        with st.container(border=True):
            # Title row with price clearly visible
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{idx}. {name}**")
            with c2:
                st.markdown(f"**{price_txt}**")

            if desc:
                st.write(desc)

            if tags and isinstance(tags, list):
                st.caption("Tags: " + ", ".join([str(t) for t in tags][:6]))

            if reason:
                st.caption(reason)


# -------------------------
# Results page switch
# -------------------------
def _find_results_page_path() -> Optional[str]:
    candidates = [
        "pages/2_📊_Results.py",
        "pages/2_Results.py",
        "pages/📊_Results.py",
        "pages/Results.py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _switch_to_results_page():
    results_page = _find_results_page_path()
    if results_page:
        try:
            st.switch_page(results_page)
        except Exception:
            st.info("Results page detected, but automatic switch isn't supported in this Streamlit version.")
            st.caption(f"Open it from the sidebar: {results_page}")
    else:
        st.info("Couldn't find a Results page file in /pages. Open Results from the sidebar.")


# -------------------------
# Inputs UI
# -------------------------
tab1, tab2 = st.tabs(["📝 Trip Info", "🎯 Preferences"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("💰 Total Budget (EUR)", min_value=100.0, max_value=10000.0, value=600.0, step=100.0)
        people = st.number_input("👥 Travelers", min_value=1, max_value=20, value=2, step=1)
        days = st.number_input("📅 Days", min_value=1, max_value=30, value=3, step=1)
    with col2:
        with_children = st.checkbox("👶 Children", value=False)
        with_disabled = st.checkbox("♿ Accessibility needs", value=False)

with tab2:
    st.markdown("### Pick a template or write your own")
    templates = {
        "Custom": "",
        "Nature + Food": "Include hiking trails, forests, parks, outdoor nature areas; Include local food experiences, restaurants, markets",
        "Culture + Museums": "Museums, historical sites, art galleries; local cuisine; relaxed pace",
        "Beach + Relax": "Beaches, sea views, spas, relaxing walks; not too many activities per day",
        "City + Nightlife": "Shopping, nightlife, city exploration, modern attractions, food tours",
    }
    choice = st.selectbox("Template", list(templates.keys()), index=0)
    preferences = st.text_area("Your preferences", value=templates[choice], height=160)

st.markdown("---")
colA, colB = st.columns([3, 2])
start = colA.button("✨ Start Dialogue (max 3 questions)", type="primary", use_container_width=True)
clear = colB.button("🔄 Clear", use_container_width=True)

if clear:
    st.session_state.chat = []
    st.session_state.dialogue_stage = "idle"
    st.session_state.profile = None
    st.session_state.itinerary_data = None
    st.session_state.questions_asked = 0
    st.session_state.user_inputs = {}
    st.session_state.basic_info = None
    st.session_state.llm_attractions_10 = []
    st.session_state.preview = {
        "ready": False,
        "shown": False,
        "auto_redirect": True,
        "redirect_in_sec": 2.0,
        "data": None,
        "city": None,
        "city_photo_url": None,
        "attractions_10": [],
        "refined_attractions": [],
    }
    st.success("Cleared.")
    st.rerun()

if start:
    if not preferences.strip():
        st.error("Please enter your preferences first.")
        st.stop()

    st.session_state.user_inputs = {
        "budget": float(budget),
        "people": int(people),
        "days": int(days),
        "with_children": bool(with_children),
        "with_disabled": bool(with_disabled),
        "preferences": preferences.strip(),
    }
    st.session_state.basic_info = {
        "budget": float(budget),
        "people": int(people),
        "days": int(days),
        "with_children": bool(with_children),
        "with_disabled": bool(with_disabled),
    }

    planner = get_planner()
    planner.reset(st.session_state.basic_info)

    st.session_state.chat = []
    st.session_state.profile = None
    st.session_state.itinerary_data = None
    st.session_state.questions_asked = 0
    st.session_state.dialogue_stage = "refine"
    st.session_state.llm_attractions_10 = []

    st.session_state.preview.update({
        "ready": False,
        "shown": False,
        "data": None,
        "city": None,
        "city_photo_url": None,
        "attractions_10": [],
        "refined_attractions": [],
    })

    with st.spinner("Generating first question..."):
        ensure_first_question_exists()

    st.rerun()

if st.session_state.get("dialogue_stage") == "refine":
    ensure_first_question_exists()


# -------------------------
# Dialogue UI
# -------------------------
if st.session_state.get("dialogue_stage") in ("refine", "llm_attractions", "running_pipeline", "preview", "done"):
    st.markdown("### 💬 Refinement Dialogue")
    st.caption(f"Questions asked: {st.session_state.get('questions_asked', 0)}/3")

    for i, m in enumerate(st.session_state.get("chat", [])):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and i == len(st.session_state.chat) - 1:
                st.write_stream(stream_words(m["content"], delay=0.02))
            else:
                st.write(m["content"])


# -------------------------
# Continue refinement
# -------------------------
if st.session_state.get("dialogue_stage") == "refine" and st.session_state.get("chat"):
    user_text = st.chat_input("Answer…")
    if user_text:
        add_msg("user", user_text)

        with st.spinner("Refining..."):
            planner = get_planner()
            step = planner.process_refinement_turn(user_text)

        if step.get("action") == "ask_question":
            if st.session_state.questions_asked >= 3:
                st.session_state.dialogue_stage = "llm_attractions"
                st.rerun()

            q = normalize_question(step.get("question", ""))
            if not q:
                step = planner.process_refinement_turn(user_text)
                q = normalize_question(step.get("question", ""))

            if q:
                add_msg("assistant", q)
                st.session_state.questions_asked += 1

            st.rerun()

        # finalized profile
        st.session_state.profile = step.get("profile")
        st.session_state.dialogue_stage = "llm_attractions"

        # store city early for UI
        city_from_profile = get_profile_city()
        if city_from_profile:
            st.session_state.preview["city"] = city_from_profile
            if st.session_state.preview.get("city_photo_url") is None:
                st.session_state.preview["city_photo_url"] = get_city_place_photo_url(city_from_profile)

        st.rerun()


# -------------------------
# Step: Generate Top 10 Attractions immediately after profile finalization
# -------------------------
if st.session_state.get("dialogue_stage") == "llm_attractions":
    if st.session_state.get("profile") is None:
        st.error("Profile not finalized. Please Clear → Start again.")
        st.stop()

    city = get_profile_city()
    refined_profile_text = get_profile_refined_profile_text()
    constraints = get_profile_constraints_dict()

    st.markdown("## 🧠 LLM Steps")

    # 1) City + photo
    st.markdown("### 1) ✅ Chosen city")
    if city:
        st.subheader(city)
        photo = st.session_state.preview.get("city_photo_url")
        if not photo:
            photo = get_city_place_photo_url(city)
            st.session_state.preview["city_photo_url"] = photo
        if photo:
            st.image(photo, caption=f"{city} (Google Places Photo)", use_container_width=True)
    else:
        st.warning("City is missing (profile.chosen_city is empty).")

    st.markdown("---")

    # 2) Generate attractions ONCE and display WITH PRICE
    st.markdown("### 2) 🗺️ Top 10 attractions (LLM) — with prices")

    if not st.session_state.llm_attractions_10:
        with st.spinner("Generating 10 attractions with the LLM..."):
            agent = LocationScoutAgent()
            raw_list = agent.generate_attractions(
                city=city or "",
                refined_profile=refined_profile_text,
                constraints=constraints,
            )
            # normalize object -> dict
            st.session_state.llm_attractions_10 = normalize_attractions_list(raw_list)

    a10 = st.session_state.llm_attractions_10[:10]
    st.session_state.preview["attractions_10"] = a10

    # Render list
    render_attractions_with_prices(a10)

    # Debug (helps you see why price might be missing)
    with st.expander("Debug: attraction payload (first item keys)"):
        st.write(f"Count: {len(a10)}")
        if a10:
            st.write("Keys:", sorted(list(a10[0].keys())))
            st.write(a10[0])

    st.markdown("---")
    proceed = st.button("➡️ Continue to build full itinerary", type="primary", use_container_width=True)
    if proceed:
        st.session_state.dialogue_stage = "running_pipeline"
        st.rerun()


# -------------------------
# Run full pipeline AFTER we displayed top 10
# -------------------------
if st.session_state.get("dialogue_stage") == "running_pipeline":
    st.markdown("### ⚙️ Building full itinerary…")
    with st.spinner("Multi-agent system working..."):
        planner = get_planner()
        results = planner.run_pipeline_from_profile(st.session_state.profile)

    st.session_state.itinerary_data = results
    st.session_state.preview["data"] = results
    st.session_state.preview["ready"] = True
    st.session_state.dialogue_stage = "preview"
    st.rerun()


# -------------------------
# Preview + redirect
# -------------------------
if st.session_state.get("dialogue_stage") == "preview" and st.session_state.preview.get("ready"):
    city = st.session_state.preview.get("city") or get_profile_city()
    city_photo_url = st.session_state.preview.get("city_photo_url")
    a10 = st.session_state.preview.get("attractions_10") or st.session_state.llm_attractions_10 or []

    st.markdown("## 🧠 LLM Steps")
    st.caption("Showing intermediate results before opening the full itinerary page.")

    st.markdown("### 1) ✅ Chosen city")
    if city:
        st.subheader(city)
        if city_photo_url:
            st.image(city_photo_url, caption=f"{city} (Google Places Photo)", use_container_width=True)

    st.markdown("---")

    st.markdown("### 2) 🗺️ Top 10 attractions (LLM) — with prices")
    render_attractions_with_prices(a10)

    st.markdown("---")
    st.markdown("### 3) 📊 Open full itinerary")
    col1, col2 = st.columns([2, 3])
    go_now = col1.button("➡️ View results page", type="primary", use_container_width=True)

    auto_redirect = st.session_state.preview.get("auto_redirect", True)
    delay = float(st.session_state.preview.get("redirect_in_sec", 2.0))

    if go_now:
        st.session_state.dialogue_stage = "done"
        _switch_to_results_page()
        st.stop()

    if auto_redirect:
        col2.caption(f"Auto-opening results in ~{delay:.1f}s…")
        time.sleep(delay)
        st.session_state.dialogue_stage = "done"
        _switch_to_results_page()
        st.stop()


if st.session_state.get("dialogue_stage") == "done" and st.session_state.get("itinerary_data"):
    st.success("Done! Open **📊 Results** from the sidebar (or it should auto-open).")
