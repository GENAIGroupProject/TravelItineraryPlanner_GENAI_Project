import time
import streamlit as st
from main import TravelPlanner

st.set_page_config(page_title="Planner", layout="wide")
st.title("Travel Itinerary Planner")

# -------------------------
# Session init
# -------------------------
if "planner" not in st.session_state:
    st.session_state.planner = TravelPlanner()

if "stage" not in st.session_state:
    # collect_basic -> ask_pref -> refine -> run_chain -> done
    st.session_state.stage = "collect_basic"

if "basic_info" not in st.session_state:
    st.session_state.basic_info = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "profile" not in st.session_state:
    st.session_state.profile = None

if "results" not in st.session_state:
    st.session_state.results = None


# -------------------------
# Helpers
# -------------------------
def stream_text_words(text: str, delay: float = 0.02):
    """Simulated word-by-word streaming for short assistant questions."""
    if not text:
        return
    parts = text.split(" ")
    for i, w in enumerate(parts):
        yield (w + (" " if i < len(parts) - 1 else ""))
        if delay:
            time.sleep(delay)

def as_tags(tags):
    if not tags:
        return ""
    return " ".join([f"`{t}`" for t in tags])

def format_opening_hours(opening_hours):
    if not opening_hours or not isinstance(opening_hours, dict):
        return None
    weekday = opening_hours.get("weekday_text")
    if isinstance(weekday, list) and weekday:
        return " | ".join(weekday[:2]) + (" | ..." if len(weekday) > 2 else "")
    open_now = opening_hours.get("open_now")
    if open_now is True:
        return "Open now"
    if open_now is False:
        return "Closed now"
    return None

def attraction_card(attr: dict):
    name = attr.get("name", "Unknown")
    tags = attr.get("tags") or []
    price_pp = attr.get("approx_price_per_person", None)
    final_price = attr.get("final_price_estimate", None)

    rating = attr.get("google_rating", None)
    rating_count = attr.get("google_user_ratings_total", None)
    opening_hours = format_opening_hours(attr.get("opening_hours"))

    with st.container(border=True):
        st.markdown(f"### {name}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Price / person", f"{price_pp} €" if price_pp is not None else "—")
        with c2:
            st.metric("Est. total", f"{final_price} €" if final_price is not None else "—")
        with c3:
            if rating is not None:
                st.metric("Rating", f"{rating} ⭐" + (f" ({rating_count})" if rating_count else ""))
            else:
                st.metric("Rating", "—")

        if opening_hours:
            st.caption(f"🕒 {opening_hours}")

        if tags:
            st.write(as_tags(tags))

def render_itinerary_cards(itinerary: dict):
    """Renders a day-by-day itinerary into cards."""
    if not itinerary or not isinstance(itinerary, dict):
        st.warning("No itinerary to display.")
        return

    days_obj = itinerary.get("days")
    if isinstance(days_obj, list):
        day_items = [(f"Day {i+1}", d) for i, d in enumerate(days_obj)]
    else:
        # fallback: day1/day2 keys
        day_keys = [k for k in itinerary.keys() if str(k).lower().startswith("day")]
        if day_keys:
            def daynum(k):
                digits = "".join([c for c in k if c.isdigit()])
                return int(digits) if digits.isdigit() else 999
            day_keys = sorted(day_keys, key=daynum)
            day_items = [(k.capitalize(), itinerary[k]) for k in day_keys]
        else:
            st.json(itinerary)
            return

    for day_title, day in day_items:
        with st.container(border=True):
            st.subheader(day_title)
            morning = day.get("morning", []) if isinstance(day, dict) else []
            afternoon = day.get("afternoon", []) if isinstance(day, dict) else []
            evening = day.get("evening", []) if isinstance(day, dict) else []

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Morning**")
                if morning:
                    for item in morning:
                        st.write("• " + (item.get("name") if isinstance(item, dict) else str(item)))
                else:
                    st.caption("—")
            with c2:
                st.markdown("**Afternoon**")
                if afternoon:
                    for item in afternoon:
                        st.write("• " + (item.get("name") if isinstance(item, dict) else str(item)))
                else:
                    st.caption("—")
            with c3:
                st.markdown("**Evening**")
                if evening:
                    for item in evening:
                        st.write("• " + (item.get("name") if isinstance(item, dict) else str(item)))
                else:
                    st.caption("—")

def render_evaluation(evaluation: dict):
    if not evaluation:
        st.warning("No evaluation available.")
        return
    if not isinstance(evaluation, dict):
        st.json(evaluation)
        return

    keys = [k for k in ["overall_score", "score", "final_score", "budget_score", "variety_score"] if k in evaluation]
    if keys:
        cols = st.columns(min(4, len(keys)))
        for i, k in enumerate(keys[:4]):
            with cols[i]:
                st.metric(k.replace("_", " ").title(), str(evaluation.get(k)))

    st.json(evaluation)


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("Trip Inputs")
    budget = st.number_input("Budget (total €)", min_value=0, value=800, step=50)
    people = st.number_input("People", min_value=1, value=2, step=1)
    days = st.number_input("Days", min_value=1, value=3, step=1)
    with_children = st.checkbox("Traveling with children", value=False)
    with_disabled = st.checkbox("Accessibility / mobility needs", value=False)

    if st.button("Start / Reset"):
        st.session_state.basic_info = {
            "budget": float(budget),
            "people": int(people),
            "days": int(days),
            "with_children": bool(with_children),
            "with_disabled": bool(with_disabled),
        }
        st.session_state.planner.reset(st.session_state.basic_info)

        st.session_state.chat = []
        st.session_state.profile = None
        st.session_state.results = None
        st.session_state.stage = "ask_pref"

# -------------------------
# Guard
# -------------------------
if st.session_state.stage == "collect_basic":
    st.info("Set trip inputs in the sidebar, then click **Start / Reset**.")
    st.stop()

# -------------------------
# Render chat history
# -------------------------
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# -------------------------
# Stage: ask_pref
# -------------------------
if st.session_state.stage == "ask_pref":
    if len(st.session_state.chat) == 0:
        msg = "Tell me your travel preferences (activities you like, pace, food, constraints)."
        with st.chat_message("assistant"):
            st.write_stream(stream_text_words(msg, delay=0.02))
        st.session_state.chat.append({"role": "assistant", "content": msg})

    user_text = st.chat_input("Type your preferences…")
    if user_text:
        with st.chat_message("user"):
            st.write(user_text)
        st.session_state.chat.append({"role": "user", "content": user_text})

        with st.spinner("Thinking..."):
            step = st.session_state.planner.start_refinement(user_text)

        if step["action"] == "ask_question":
            q = step["question"]
            with st.chat_message("assistant"):
                st.write_stream(stream_text_words(q, delay=0.02))
            st.session_state.chat.append({"role": "assistant", "content": q})
            st.session_state.stage = "refine"
        else:
            st.session_state.profile = step["profile"]
            msg = f"Great — I’ll build your itinerary for **{st.session_state.profile.chosen_city}** now."
            with st.chat_message("assistant"):
                st.write_stream(stream_text_words(msg, delay=0.015))
            st.session_state.chat.append({"role": "assistant", "content": msg})
            st.session_state.stage = "run_chain"

# -------------------------
# Stage: refine (ask up to 3 questions)
# -------------------------
if st.session_state.stage == "refine":
    user_text = st.chat_input("Answer the question…")
    if user_text:
        with st.chat_message("user"):
            st.write(user_text)
        st.session_state.chat.append({"role": "user", "content": user_text})

        with st.spinner("Refining..."):
            step = st.session_state.planner.process_refinement_turn(user_text)

        if step["action"] == "ask_question":
            q = step["question"]
            with st.chat_message("assistant"):
                st.write_stream(stream_text_words(q, delay=0.02))
            st.session_state.chat.append({"role": "assistant", "content": q})
        else:
            st.session_state.profile = step["profile"]
            msg = f"Perfect. Now generating attractions and itinerary for **{st.session_state.profile.chosen_city}**."
            with st.chat_message("assistant"):
                st.write_stream(stream_text_words(msg, delay=0.015))
            st.session_state.chat.append({"role": "assistant", "content": msg})
            st.session_state.stage = "run_chain"

# -------------------------
# Stage: run_chain (slow part starts here)
# -------------------------
if st.session_state.stage == "run_chain" and st.session_state.profile is not None:
    with st.spinner("Generating attractions, itinerary and evaluation..."):
        results = st.session_state.planner.run_pipeline_from_profile(st.session_state.profile)
        st.session_state.results = results

    # optional: final streaming summary
    with st.chat_message("assistant"):
        st.write_stream(
            st.session_state.planner.stream_final_summary_words(
                results.get("profile", {}),
                results.get("itinerary", {})
            )
        )

    st.session_state.stage = "done"

# -------------------------
# Stage: done (show list 10 + cards + itinerary cards + evaluation)
# -------------------------
if st.session_state.stage == "done" and st.session_state.results:
    results = st.session_state.results

    st.subheader("10 attractions (generated list)")
    gen = results.get("attractions_generated", [])
    if gen:
        for i, a in enumerate(gen, 1):
            name = a.get("name", "Attraction")
            price = a.get("approx_price_per_person", "—")
            st.write(f"{i}. **{name}** — {price} € / person")
    else:
        st.warning("No attractions generated.")

    st.subheader("Refined attractions (cards)")
    refined = results.get("attractions_budget_filtered", [])
    if not refined:
        st.warning("No attractions after budget filtering.")
    else:
        for a in refined:
            attraction_card(a)

    st.subheader("Final itinerary (cards)")
    render_itinerary_cards(results.get("itinerary", {}))

    st.subheader("Evaluation")
    render_evaluation(results.get("evaluation", {}))
