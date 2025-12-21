import streamlit as st

from ui.style import inject_global_css
from ui.state import ensure_session_state
from ui.sidebar import render_sidebar
from ui.components import (
    to_plain,
    budget_overview,
    display_daily_schedule,
    display_attraction_card,
    display_evaluation,
    display_detailed_view,
    display_map_view,
)

inject_global_css()
ensure_session_state()

# Sidebar
render_sidebar()

st.markdown('<h1 class="main-header">📊 Results</h1>', unsafe_allow_html=True)

data = st.session_state.itinerary_data
if not data:
    st.warning("No itinerary yet. Go to **✈️ Plan Trip** first.")
    st.stop()

profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
city = profile.get("chosen_city", "Destination")

attractions = data.get("attractions_budget_filtered") or data.get("attractions_enriched") or data.get("attractions_generated") or []
itinerary = data.get("itinerary") or {}
evaluation = data.get("evaluation") or st.session_state.evaluation or {}

# Metrics
total_cost = sum((a.get("final_price_estimate") or 0) for a in attractions)
total_budget = (profile.get("constraints") or {}).get("budget", 0)
remaining = (total_budget - total_cost) if total_budget else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📍 Destination", city)
c2.metric("💰 Total Cost", f"€{total_cost:.2f}")
c3.metric("🎯 Remaining", f"€{remaining:.2f}")
if isinstance(evaluation, dict):
    scores = [evaluation.get("interest_match"), evaluation.get("budget_realism"), evaluation.get("logistics"), evaluation.get("suitability_for_constraints")]
    scores = [s for s in scores if isinstance(s, (int, float))]
    overall = sum(scores) / len(scores) if scores else 0
else:
    overall = 0
c4.metric("⭐ Overall", f"{overall:.1f}/5")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily Schedule", "🏞️ Attractions", "⭐ Evaluation", "🗺️ Map"])

with tab1:
    display_daily_schedule(itinerary, attractions)

with tab2:
    st.markdown(f"### 🏞️ {len(attractions)} Attractions")

    colA, colB = st.columns(2)
    with colA:
        sort_by = st.selectbox("Sort by:", ["Name", "Price", "Rating"])
    with colB:
        tag_filter = st.multiselect("Filter by tags:", display_detailed_view(attractions), default=[])

    filtered = attractions
    if tag_filter:
        filtered = [a for a in attractions if any(t in (a.get("tags") or []) for t in tag_filter)]

    if sort_by == "Price":
        filtered.sort(key=lambda x: x.get("final_price_estimate") or 0)
    elif sort_by == "Rating":
        filtered.sort(key=lambda x: x.get("google_rating") or 0, reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("name") or "")

    for a in filtered:
        display_attraction_card(a, compact=False)

with tab3:
    display_evaluation(evaluation if isinstance(evaluation, dict) else {})

with tab4:
    # ✅ FIX: pass the required 'city' argument
    display_map_view(attractions, city)
