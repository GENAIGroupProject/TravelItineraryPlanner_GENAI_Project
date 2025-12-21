from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def to_plain(obj: Any) -> Any:
    """Convert pydantic-ish objects into plain python types for UI."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj

def get_all_tags(attractions: List[Dict[str, Any]]) -> List[str]:
    tags = set()
    for attr in attractions or []:
        for t in (attr.get("tags") or []):
            if t:
                tags.add(str(t))
    return sorted(tags)

def display_agent_logs(max_items: int = 5) -> None:
    if not st.session_state.get("agent_logs"):
        return
    st.markdown("### 📋 Agent Activity Log")
    logs = st.session_state.agent_logs[-max_items:]
    for log in logs:
        status = log.get("status", "processing")
        status_emoji = "🔄" if status == "processing" else "✅" if status == "success" else "❌"
        st.markdown(
            f"""
            <div class="agent-card">
                <strong>{status_emoji} {log.get('agent','Agent')}</strong><br>
                <small>🕐 {log.get('timestamp','')}</small><br>
                {log.get('message','')}
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_attraction_card(attr: Dict[str, Any], compact: bool = False) -> None:
    name = attr.get("name", "Unknown")
    description = attr.get("short_description", "")
    price = float(attr.get("final_price_estimate", 0) or 0)
    rating = attr.get("google_rating")
    opening_hours = attr.get("opening_hours")
    tags = attr.get("tags", []) or []

    if compact:
        st.markdown(f"**{name}** - €{price:.2f}")
        if isinstance(rating, (int, float)):
            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            st.markdown(f"<span class='rating-stars'>{stars} ({rating}/5)</span>", unsafe_allow_html=True)
        return

    with st.container():
        st.markdown('<div class="attraction-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {name}")
            if description:
                st.markdown(f"*{description}*")
        with col2:
            st.metric("Price", f"€{price:.2f}")
            if isinstance(rating, (int, float)):
                st.metric("Rating", f"{rating}/5")

        if tags:
            tag_html = " ".join([f"<span class='pill'>{t}</span>" for t in tags[:5]])
            st.markdown(f"**Tags:** {tag_html}", unsafe_allow_html=True)

        if opening_hours and isinstance(opening_hours, dict):
            with st.expander("🕒 Opening Hours"):
                if opening_hours.get("open_now") is not None:
                    st.write("**Status:**", "✅ Open Now" if opening_hours["open_now"] else "❌ Closed")
                if opening_hours.get("weekday_text"):
                    for line in opening_hours["weekday_text"]:
                        st.write(line)
                elif opening_hours.get("periods"):
                    st.write("**Schedule:**")
                    for period in opening_hours["periods"][:3]:
                        open_time = period.get("open", {}).get("time", "")
                        close_time = period.get("close", {}).get("time", "")
                        if open_time and close_time:
                            open_formatted = f"{open_time[:2]}:{open_time[2:]}"
                            close_formatted = f"{close_time[:2]}:{close_time[2:]}"
                            st.write(f"{open_formatted} - {close_formatted}")

        st.markdown("</div>", unsafe_allow_html=True)

def display_daily_schedule(itinerary: Dict[str, Any], attractions: List[Dict[str, Any]]) -> None:
    if not itinerary:
        st.info("No itinerary available.")
        return

    def find_attr(name: str) -> Optional[Dict[str, Any]]:
        return next((a for a in attractions if a.get("name") == name), None)

    # Handle both {"days": [...]} and {"day1": {...}}
    if isinstance(itinerary.get("days"), list):
        day_items = [(f"day{i+1}", d or {}) for i, d in enumerate(itinerary["days"])]
    else:
        def daynum(k: str) -> int:
            return int("".join(c for c in k if c.isdigit()) or 999)

        keys = sorted(
            [k for k in itinerary if str(k).lower().startswith("day")],
            key=daynum
        )
        day_items = [(k, itinerary.get(k) or {}) for k in keys]

    for day_key, day in day_items:
        st.markdown(
            f"<div class='day-schedule'><h3>📅 {day_key.upper().replace('DAY','DAY ')}</h3>",
            unsafe_allow_html=True,
        )

        for slot in ["morning", "afternoon", "evening"]:
            st.markdown(f"### ⏰ {slot.title()}")
            items = (day or {}).get(slot, []) or []

            if not items:
                st.caption("—")
                continue

            for item in items:
                name = item.get("name") if isinstance(item, dict) else str(item)
                if not name:
                    continue

                attr = find_attr(name)

                # Two-column layout: text | image
                col_text, col_img = st.columns([3, 1])

                with col_text:
                    if attr:
                        display_attraction_card(attr, compact=True)
                    else:
                        st.write(f"• {name}")

                with col_img:
                    if attr and attr.get("google_photo_reference"):
                        photo_url = (
                            "https://maps.googleapis.com/maps/api/place/photo"
                            f"?maxwidth=400"
                            f"&photo_reference={attr['google_photo_reference']}"
                            f"&key={st.secrets.get('GOOGLE_API_KEY', '')}"
                        )
                        st.image(photo_url, use_container_width=True)

def _display_one_day(day_key: str, day_data: Dict[str, Any], attractions: List[Dict[str, Any]]) -> None:
    if not day_data:
        return
    st.markdown(
        f"<div class='day-schedule'><h3>📅 {day_key.upper().replace('DAY','DAY ')}</h3>",
        unsafe_allow_html=True,
    )

    for slot in ["morning", "afternoon", "evening"]:
        slot_attractions = day_data.get(slot, []) or []
        if slot_attractions:
            st.markdown(f"### ⏰ {slot.title()}")
            for attr_name in slot_attractions:
                attr = next((a for a in attractions if a.get("name") == attr_name), None)
                if attr:
                    display_attraction_card(attr, compact=True)

    st.markdown("</div>", unsafe_allow_html=True)

def display_attractions(attractions: List[Dict[str, Any]]) -> None:
    st.markdown(f"### 🏞️ {len(attractions)} Attractions")

    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox("Sort by:", ["Name", "Price", "Rating"])
    with col2:
        tag_filter = st.multiselect("Filter by tags:", get_all_tags(attractions), default=[])

    filtered = attractions
    if tag_filter:
        filtered = [
            a for a in attractions
            if any(t in (a.get("tags", []) or []) for t in tag_filter)
        ]

    if sort_by == "Price":
        filtered.sort(key=lambda x: float(x.get("final_price_estimate", 0) or 0))
    elif sort_by == "Rating":
        filtered.sort(key=lambda x: float(x.get("google_rating", 0) or 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("name", ""))

    for attr in filtered:
        display_attraction_card(attr, compact=False)

def display_evaluation(evaluation: Dict[str, Any]) -> None:
    if not evaluation:
        st.info("No evaluation data available.")
        return

    st.markdown("### ⭐ Itinerary Evaluation")

    scores = [
        evaluation.get("interest_match", 0),
        evaluation.get("budget_realism", 0),
        evaluation.get("logistics", 0),
        evaluation.get("suitability_for_constraints", 0),
    ]
    scores = [float(s) for s in scores if isinstance(s, (int, float))]
    overall = sum(scores) / len(scores) if scores else 0

    stars = "★" * int(overall) + "☆" * (5 - int(overall)) + f" ({overall:.1f}/5)"
    st.markdown(f"### {stars}")

    metrics = [
        ("Interest Match", evaluation.get("interest_match", 0)),
        ("Budget Realism", evaluation.get("budget_realism", 0)),
        ("Schedule Flow", evaluation.get("logistics", 0)),
        ("Suitability", evaluation.get("suitability_for_constraints", 0)),
    ]

    for name, score in metrics:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{name}:**")
        with col2:
            score = float(score or 0)
            bar_html = "<div style='background:#E5E7EB;border-radius:5px;height:20px;width:100%;'>"
            bar_html += f"<div style='background:#3B82F6;border-radius:5px;height:100%;width:{score*20}%;'></div>"
            bar_html += f"<div style='position:relative;top:-20px;text-align:center;color:black;font-weight:bold;'>{score}/5</div>"
            bar_html += "</div>"
            st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("### 💬 Expert Feedback")
    st.info(evaluation.get("comment", "No comment available"))

    st.markdown("### 📊 Evaluation Radar Chart")
    categories = ["Interest Match", "Budget Realism", "Schedule Flow", "Suitability"]
    values = [
        float(evaluation.get("interest_match", 0) or 0),
        float(evaluation.get("budget_realism", 0) or 0),
        float(evaluation.get("logistics", 0) or 0),
        float(evaluation.get("suitability_for_constraints", 0) or 0),
    ]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#3B82F6"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

def display_map_view(attractions: List[Dict[str, Any]], city: str) -> None:
    st.markdown(f"### 🗺️ Attractions in {city}")

    locations: List[Dict[str, Any]] = []
    for attr in attractions:
        loc = attr.get("location")
        if isinstance(loc, dict) and "lat" in loc and "lng" in loc:
            locations.append(
                {
                    "name": attr.get("name", "Unknown"),
                    "lat": loc["lat"],
                    "lon": loc["lng"],
                    "price": float(attr.get("final_price_estimate", 0) or 0),
                    "rating": float(attr.get("google_rating", 0) or 0),
                }
            )

    if not locations:
        st.info("No location data available for attractions.")
        return

    df = pd.DataFrame(locations)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="name",
        hover_data=["price", "rating"],
        color="price",
        size="rating",
        color_continuous_scale=px.colors.cyclical.IceFire,
        zoom=12,
        height=500,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📍 Location Details"):
        for loc in locations:
            st.write(f"**{loc['name']}**")
            st.write(f"Coordinates: {loc['lat']:.4f}, {loc['lon']:.4f}")
            st.write(f"Price: €{loc['price']:.2f} | Rating: {loc['rating']}/5")
            st.write("---")

def display_detailed_view(data: Any) -> None:
    st.markdown("### 📋 Detailed Data View")
    display_data = to_plain(data)

    with st.expander("📄 View Raw JSON"):
        st.json(display_data)

    st.markdown("### 📊 Statistics")
    attractions = display_data.get("attractions", []) if isinstance(display_data, dict) else []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Attractions", len(attractions))
    with col2:
        avg_price = sum(float(a.get("final_price_estimate", 0) or 0) for a in attractions) / max(len(attractions), 1)
        st.metric("Average Price", f"€{avg_price:.2f}")
    with col3:
        ratings = [float(a.get("google_rating", 0) or 0) for a in attractions if a.get("google_rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        st.metric("Average Rating", f"{avg_rating:.1f}/5")

    st.markdown("### 🏷️ Tags Distribution")
    all_tags = get_all_tags(attractions)
    tag_counts = {t: sum(1 for a in attractions if t in (a.get("tags", []) or [])) for t in all_tags}

    if tag_counts:
        fig = px.bar(
            x=list(tag_counts.keys()),
            y=list(tag_counts.values()),
            labels={"x": "Tag", "y": "Count"},
            color=list(tag_counts.values()),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

def budget_overview(profile: Dict[str, Any], attractions: List[Dict[str, Any]]) -> Tuple[float, float, float, go.Figure]:
    total_cost = sum(float(a.get("final_price_estimate", 0) or 0) for a in attractions)
    total_budget = float((profile.get("constraints", {}) or {}).get("budget", 1) or 1)
    remaining_budget = total_budget - total_cost
    usage_percent = (total_cost / total_budget * 100) if total_budget > 0 else 0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=usage_percent,
            title={"text": "Budget Utilization"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(height=250)

    return total_cost, remaining_budget, usage_percent, fig
