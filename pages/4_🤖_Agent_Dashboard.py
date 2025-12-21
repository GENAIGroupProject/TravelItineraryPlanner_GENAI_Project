import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

from ui.style import inject_global_css
from ui.state import ensure_session_state
from ui.sidebar import render_sidebar

inject_global_css()
ensure_session_state()
render_sidebar()

st.markdown('<h1 class="main-header">🤖 Agent Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### 📊 Agent Performance")

agents = [
    {"name": "Semantic Agent", "icon": "🤖", "purpose": "Understands preferences"},
    {"name": "Interest Refinement", "icon": "🎯", "purpose": "Finds destination"},
    {"name": "Location Scout", "icon": "📍", "purpose": "Discovers attractions"},
    {"name": "Google Places", "icon": "🌐", "purpose": "Gets real-time data"},
    {"name": "Budget Agent", "icon": "💰", "purpose": "Optimizes budget"},
    {"name": "Scheduler Agent", "icon": "📅", "purpose": "Creates schedule"},
    {"name": "Evaluation Agent", "icon": "⭐", "purpose": "Assesses quality"},
]

cols = st.columns(3)
for idx, agent in enumerate(agents):
    with cols[idx % 3]:
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1rem;border-radius:10px;margin:0.5rem 0;'>
                <h3>{agent['icon']} {agent['name']}</h3>
                <p>{agent['purpose']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

st.markdown("### 📋 Recent Agent Logs")

LOG_FILE = "travel_planner.log"


def read_agent_logs(log_file: str, max_lines: int = 300):
    if not os.path.exists(log_file):
        return []

    logs = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]

    for line in lines:
        line = line.strip()

        # Match: [AGENT] message
        bracket_match = re.search(r"\[(.*?)\]\s+(.*)", line)

        # Match structured agent communications
        if "AGENT_COMM:" in line:
            logs.append({
                "timestamp": line[:19],
                "agent": "AGENT_COMM",
                "level": "INFO",
                "message": line.split("AGENT_COMM:", 1)[1].strip(),
            })

        elif "AGENT_OUTPUT:" in line:
            logs.append({
                "timestamp": line[:19],
                "agent": "AGENT_OUTPUT",
                "level": "INFO",
                "message": line.split("AGENT_OUTPUT:", 1)[1].strip(),
            })

        elif bracket_match:
            agent = bracket_match.group(1)
            message = bracket_match.group(2)

            logs.append({
                "timestamp": line[:19],
                "agent": agent,
                "level": "INFO",
                "message": message,
            })

    return logs


agent_logs = read_agent_logs(LOG_FILE)

if agent_logs:
    agent_logs = list(reversed(agent_logs))
    df = pd.DataFrame(agent_logs)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No agent logs found in travel_planner.log")

st.markdown("### 🖥️ System Status")
col1, col2 = st.columns(2)

with col1:
    try:
        from utils.llm_client import LLMClient  # type: ignore
        llm = LLMClient()
        if llm.check_health():
            st.success("✅ LLM Server: Connected")
        else:
            st.error("❌ LLM Server: Disconnected")
    except Exception:
        st.warning("⚠️ LLM Status: Unknown")

with col2:
    try:
        from config import Config  # type: ignore
        if getattr(Config, "GOOGLE_API_KEY", None) and Config.GOOGLE_API_KEY == "AIzaSyANxjWJzD0BetncqmnBp069mfnawH9xO6g":
            st.success("✅ Google API: Configured")
        else:
            st.warning("⚠️ Google API: Using demo key")
    except Exception:
        st.warning("⚠️ Google API: Unknown")
