import json
import streamlit as st
import pandas as pd
from datetime import datetime

from ui.style import inject_global_css
from ui.state import ensure_session_state
from ui.sidebar import render_sidebar

inject_global_css()
ensure_session_state()
render_sidebar()

st.markdown('<h1 class="main-header">📥 Export</h1>', unsafe_allow_html=True)

data = st.session_state.itinerary_data
if not data:
    st.warning("No itinerary to export. Generate one first.")
    st.stop()

profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
city = profile.get("chosen_city", "destination")

st.markdown("### 📄 Download")

col1, col2 = st.columns(2)

with col1:
    json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        "📥 Download JSON",
        data=json_str,
        file_name=f"itinerary_{city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

with col2:
    attractions = data.get("attractions_budget_filtered") or data.get("attractions_enriched") or data.get("attractions_generated") or []
    if attractions:
        rows = []
        for a in attractions:
            rows.append({
                "Name": a.get("name", ""),
                "Price (€)": a.get("final_price_estimate", 0),
                "Rating": a.get("google_rating", ""),
                "Tags": ", ".join(a.get("tags") or []),
            })
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button(
            "📊 Download Attractions CSV",
            data=csv,
            file_name=f"attractions_{city}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No attractions to export.")
