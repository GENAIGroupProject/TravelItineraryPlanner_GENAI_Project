import json
from glob import glob
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation Report")

# -------------------------
# Helpers
# -------------------------
def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

def as_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default

def format_dt_from_filename(path: str):
    # supports logs/evaluation_YYYYMMDD_HHMMSS.json or similar
    name = path.split("/")[-1]
    base = name.replace("evaluation_", "").replace(".json", "")
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S", "%Y%m%d-%H%M%S"):
        try:
            return datetime.strptime(base, fmt)
        except Exception:
            pass
    return None

def score_block(label: str, value: int, max_score: int = 5):
    # metric + progress bar style
    if value is None:
        st.metric(label, "—")
        return
    st.metric(label, f"{value} / {max_score}")
    st.progress(min(max(value / max_score, 0.0), 1.0))

def overall_score_from_fields(data: dict):
    keys = ["interest_match", "budget_realism", "logistics", "suitability_for_constraints"]
    vals = [as_int(data.get(k)) for k in keys if as_int(data.get(k)) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)

# -------------------------
# Find evaluation files
# -------------------------
log_files = sorted(glob("logs/evaluation_*.json"), reverse=True)

with st.sidebar:
    st.header("Select report")
    if not log_files:
        st.warning("No evaluation files found in ./logs")
        st.stop()

    # show file with optional parsed timestamp
    options = []
    for p in log_files:
        dt = format_dt_from_filename(p)
        label = p if dt is None else f"{p}  ({dt.strftime('%Y-%m-%d %H:%M:%S')})"
        options.append((label, p))

    selected_label = st.selectbox("Evaluation file", [o[0] for o in options], index=0)
    selected = dict(options)[selected_label]

# -------------------------
# Load data
# -------------------------
data, err = safe_load_json(selected)
if err:
    st.error(f"Could not read {selected}: {err}")
    st.stop()

if not isinstance(data, dict):
    st.error("Evaluation file is not a JSON object.")
    st.json(data)
    st.stop()

# -------------------------
# Header
# -------------------------
st.caption(f"Source: `{selected}`")

comment = data.get("comment") or data.get("notes") or data.get("summary") or ""
overall = overall_score_from_fields(data)

# -------------------------
# Top summary (nice)
# -------------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Summary")
    if comment:
        st.info(comment)
    else:
        st.info("No comment/summary was provided in this evaluation.")

with c2:
    st.subheader("Overall")
    if overall is None:
        st.metric("Overall score", "—")
    else:
        st.metric("Overall score", f"{overall} / 5")
        st.progress(min(max(overall / 5.0, 0.0), 1.0))

st.divider()

# -------------------------
# Scorecards
# -------------------------
st.subheader("Scores")

colA, colB, colC, colD = st.columns(4)

with colA:
    score_block("Interest match", as_int(data.get("interest_match")))
with colB:
    score_block("Budget realism", as_int(data.get("budget_realism")))
with colC:
    score_block("Logistics", as_int(data.get("logistics")))
with colD:
    score_block("Constraints suitability", as_int(data.get("suitability_for_constraints")))

st.divider()

# -------------------------
# Report style sections
# -------------------------
st.subheader("Report")

left, right = st.columns([1, 1])

with left:
    with st.container(border=True):
        st.markdown("### What looks strong")
        bullets = []
        if as_int(data.get("logistics")) is not None and as_int(data.get("logistics")) >= 4:
            bullets.append("Plan sequencing and timing seem practical.")
        if as_int(data.get("suitability_for_constraints")) is not None and as_int(data.get("suitability_for_constraints")) >= 4:
            bullets.append("Good fit for stated constraints (kids / accessibility).")
        if as_int(data.get("interest_match")) is not None and as_int(data.get("interest_match")) >= 4:
            bullets.append("Activities align well with preferences.")

        if bullets:
            for b in bullets:
                st.write(f"✅ {b}")
        else:
            st.caption("No strong positives were inferred from the numeric scores.")

with right:
    with st.container(border=True):
        st.markdown("### What to improve")
        bullets = []
        if as_int(data.get("budget_realism")) is not None and as_int(data.get("budget_realism")) <= 3:
            bullets.append("Adjust attraction costs, dining assumptions, or transportation choices to match budget.")
        if as_int(data.get("interest_match")) is not None and as_int(data.get("interest_match")) <= 3:
            bullets.append("Add more activities matching the user’s stated interests.")
        if as_int(data.get("logistics")) is not None and as_int(data.get("logistics")) <= 3:
            bullets.append("Reduce travel time between stops or avoid overpacking a day.")

        if bullets:
            for b in bullets:
                st.write(f"⚠️ {b}")
        else:
            st.caption("No improvement areas were inferred from the numeric scores.")

st.divider()

# -------------------------
# Raw JSON fallback
# -------------------------
with st.expander("Raw evaluation JSON"):
    st.json(data)
