#!/usr/bin/env python3
import streamlit as st

from ui.style import inject_global_css
from ui.state import ensure_session_state
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="AI Travel Itinerary Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
ensure_session_state()
render_sidebar()
st.markdown(
    """
    <style>
      /* Hide the default pages navigation in the sidebar */
      [data-testid="stSidebarNav"] { display: none; }
      
      /* Optional: remove extra top padding space that the nav used */
      [data-testid="stSidebarNav"] + div { margin-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 class="main-header">✈️ AI Travel Itinerary Planner</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Multi-Agent AI System")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🤖 7 AI Agents")
    st.markdown("""
- Semantic Understanding  
- Interest Refinement  
- Location Scouting  
- Budget Planning  
- Schedule Optimization  
- Quality Evaluation  
- Google Places Integration  
""")

with col2:
    st.markdown("### 🌍 Real-time Data")
    st.markdown("""
- Live Google Places API  
- Current Ratings & Prices  
- Opening Hours  
- Accessibility Info  
- User Reviews  
""")

with col3:
    st.markdown("### 📊 Smart Features")
    st.markdown("""
- Budget Optimization  
- Personalized Matching  
- Constraint Handling  
- Quality Evaluation  
- Multi-day Planning  
""")

st.markdown("---")
st.markdown('<h2 class="sub-header">🚀 How It Works</h2>', unsafe_allow_html=True)

steps = [
    ("1️⃣", "Input Preferences", "Tell us about your dream trip, budget, and constraints"),
    ("2️⃣", "AI Analysis", "7 specialized agents work together to understand your needs"),
    ("3️⃣", "Destination Selection", "AI finds the perfect city matching your preferences"),
    ("4️⃣", "Attraction Discovery", "Real-time Google Places data enriches recommendations"),
    ("5️⃣", "Itinerary Creation", "Smart scheduling and budget optimization"),
    ("6️⃣", "Quality Evaluation", "AI evaluates and scores the final itinerary"),
]
for icon, title, desc in steps:
    st.markdown(f"**{icon} {title}:** {desc}")

st.markdown("---")
if st.button("🚀 Start Planning Your Trip", use_container_width=True):
    if hasattr(st, "switch_page"):
        st.switch_page("pages/1_✈️_Plan_Trip.py")
    else:
        st.info("Use the sidebar page menu to open ✈️ Plan Trip.")
