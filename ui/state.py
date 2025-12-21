import streamlit as st

def ensure_session_state() -> None:
    """Ensure all keys used across pages exist.

    Multi-page Streamlit runs each page script independently, so session_state
    is our shared store across pages.
    """
    defaults = {
        "page": "home",
        "itinerary_data": None,
        "generation_in_progress": False,
        "user_inputs": {},
        "agent_logs": [],
        "itinerary_generated": False,
        # optional extras used by some backends
        "evaluation": None,
        "evaluation_file": None,
        "run_file": None,
        "planner": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
