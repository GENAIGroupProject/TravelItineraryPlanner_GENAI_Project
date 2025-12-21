import streamlit as st

def _switch_page(path: str) -> None:
    """Switch page if supported; otherwise show a hint."""
    if hasattr(st, "switch_page"):
        st.switch_page(path)
    else:
        # Older Streamlit: can't programmatically switch
        st.warning("This Streamlit version doesn't support st.switch_page(). Use the sidebar page menu.")

def render_sidebar() -> None:
    with st.sidebar:
        # Branding
        try:
            st.image("https://img.icons8.com/color/96/000000/airplane-take-off.png", width=80)
        except Exception:
            st.write("✈️")
        st.title("🌍 Navigation")

        # Navigation buttons (to match the original single-file UI)
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            _switch_page("app.py")
        if st.button("✈️ Plan Trip", key="nav_plan", use_container_width=True):
            _switch_page("pages/1_✈️_Plan_Trip.py")
        if st.button("📊 View Results", key="nav_results", use_container_width=True):
            _switch_page("pages/2_📊_Results.py")
        if st.button("🤖 Agent Dashboard", key="nav_dashboard", use_container_width=True):
            _switch_page("pages/4_🤖_Agent_Dashboard.py")
        if st.button("📥 Export", key="nav_export", use_container_width=True):
            _switch_page("pages/5_📥_Export.py")

        st.markdown("---")

        # If we have itinerary_data, show readiness + destination
        data = st.session_state.get("itinerary_data") or {}
        if data:
            st.success("✅ Itinerary Ready!")

            profile = data.get("profile") if isinstance(data, dict) else None
            if isinstance(profile, dict):
                destination = profile.get("chosen_city", "Unknown")
            elif profile is not None:
                destination = getattr(profile, "chosen_city", "Unknown")
            else:
                destination = "Unknown"

            st.info(f"**Destination:** {destination}")

        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")

        if not data:
            st.caption("Run **✈️ Plan Trip** to see stats here.")
            return

        # ✅ Match Results.py attraction selection logic exactly
        attractions = (
            data.get("attractions_budget_filtered")
            or data.get("attractions_enriched")
            or data.get("attractions_generated")
            or []
        )

        # Safe numeric extraction (dict or object)
        def _price(a) -> float:
            try:
                if isinstance(a, dict):
                    return float(a.get("final_price_estimate") or 0)
                return float(getattr(a, "final_price_estimate", 0) or 0)
            except Exception:
                return 0.0

        total_cost = sum(_price(a) for a in (attractions or []))

        # Budget/remaining (same structure as Results.py)
        profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
        constraints = (profile.get("constraints") or {}) if isinstance(profile, dict) else {}
        try:
            total_budget = float((constraints.get("budget") or 0))
        except Exception:
            total_budget = 0.0

        remaining = (total_budget - total_cost) if total_budget else 0.0

        # Overall score (same as Results.py)
        evaluation = data.get("evaluation") or st.session_state.get("evaluation") or {}
        if isinstance(evaluation, dict):
            scores = [
                evaluation.get("interest_match"),
                evaluation.get("budget_realism"),
                evaluation.get("logistics"),
                evaluation.get("suitability_for_constraints"),
            ]
            scores = [s for s in scores if isinstance(s, (int, float))]
            overall = (sum(scores) / len(scores)) if scores else 0.0
        else:
            overall = 0.0

        # ✅ Sidebar metrics
        st.metric("💰 Total Cost", f"€{total_cost:.2f}")
        st.metric("🎯 Remaining", f"€{remaining:.2f}")
        st.metric("🏞️ Attractions", len(attractions or []))
        st.metric("⭐ Overall", f"{overall:.1f}/5")
