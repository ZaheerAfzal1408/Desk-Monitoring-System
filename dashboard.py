import streamlit as st
import requests
import textwrap

st.set_page_config(page_title="Smart Office Monitor", layout="wide", page_icon="🏙️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
.stApp { background: #0d0f14; }

h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    letter-spacing: -1px;
    color: #f0f4ff !important;
}
.subtitle {
    color: #4a5568;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: -8px;
    margin-bottom: 32px;
}
.desk-card {
    padding: 22px 24px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #1e2433;
    background: #141720;
    position: relative;
    overflow: hidden;
    min-height: 180px;
}
.desk-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
}
.working              { border-color: #14532d; }
.working::before      { background: #22c55e; }
.sitting-idle         { border-color: #78350f; }
.sitting-idle::before { background: #f59e0b; }
.using-mobile         { border-color: #7f1d1d; }
.using-mobile::before { background: #ef4444; }
.walking              { border-color: #0c4a6e; }
.walking::before      { background: #38bdf8; }
.standing             { border-color: #4a1d96; }
.standing::before     { background: #a855f7; }
.vacant               { border-color: #1e2433; opacity: 0.5; }
.vacant::before       { background: #374151; }

.desk-label    { font-size: 0.68rem; letter-spacing: 0.15em; color: #4a5568; text-transform: uppercase; margin-bottom: 4px; }
.person-name   { font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 800; color: #e2e8f0; margin-bottom: 2px; }
.activity-text { font-size: 0.82rem; color: #94a3b8; margin-bottom: 16px; }

.pill              { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.pill-working      { background: #14532d; color: #86efac; }
.pill-sitting-idle { background: #78350f; color: #fcd34d; }
.pill-using-mobile { background: #7f1d1d; color: #fca5a5; }
.pill-walking      { background: #0c4a6e; color: #7dd3fc; }
.pill-standing     { background: #4a1d96; color: #c4b5fd; }
.pill-vacant       { background: #1f2937; color: #6b7280; }

.timer { float: right; font-size: 0.75rem; color: #374151; }

.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 5px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ─ Employee Stats Section ────────────────────────────────────────── */
.emp-card {
    background: #141720;
    border: 1px solid #1e2433;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.emp-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #e2e8f0;
    margin-bottom: 2px;
}
.emp-total {
    font-size: 0.68rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
}
.stat-row { margin-bottom: 10px; }
.stat-label-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 0.74rem;
    color: #94a3b8;
}
.stat-bar-track {
    background: #1e2433;
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
}
.stat-bar-fill {
    height: 7px;
    border-radius: 4px;
    transition: width 0.6s ease;
}
[data-testid="stStatusWidget"] { visibility: hidden; height: 0%; position: fixed; }
.stMetric label { color: #4a5568 !important; font-size: 0.7rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Static header — rendered ONCE, never re-rendered ──────────────────────────
st.markdown("# 🏙️ Smart Office Monitor")
st.markdown('<p class="subtitle">Real-time edge-computing employee monitoring</p>',
            unsafe_allow_html=True)

BACKEND_URL  = "http://localhost:8000/logs"
NUM_COLUMNS  = 3


def activity_to_css(activity: str) -> str:
    return activity.lower().replace(" ", "-")


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s"


def activity_icon(activity: str) -> str:
    return {"Working": "💻", "Using Mobile": "📱", "Walking": "🚶",
            "Sitting Idle": "💤", "Standing": "🧍", "—": "🪑"}.get(activity, "❓")


def build_card_html(desk_id: str, data: dict) -> str:
    """Return the full HTML string for a single desk card."""
    activity    = data.get("activity", "—")
    person_name = data.get("person_name", "—")
    duration    = data.get("time", 0)
    is_vacant   = data.get("status") == "Vacant"

    css_cls  = "vacant" if is_vacant else activity_to_css(activity)
    pill_cls = f"pill-{css_cls}"
    icon     = activity_icon(activity)
    label    = "Vacant" if is_vacant else activity
    live_dot = "" if is_vacant else '<span class="live-dot"></span>'

    return textwrap.dedent(f"""
        <div class="desk-card {css_cls}">
            <div class="desk-label">{live_dot}Desk {desk_id}</div>
            <div class="person-name">{person_name}</div>
            <div class="activity-text">{icon} {label}</div>
            <span class="pill {pill_cls}">{label}</span>
            <span class="timer">{fmt_duration(duration)}</span>
        </div>
    """)


ACTIVITY_PALETTE = {
    "Working":      ("#22c55e", "💻"),
    "Sitting Idle": ("#f59e0b", "💤"),
    "Using Mobile": ("#ef4444", "📱"),
    "Walking":      ("#38bdf8", "🚶"),
    "Standing":     ("#a855f7", "🧍"),
}


def build_employee_stats_html(name: str, stats: dict) -> str:
    """Return HTML for one employee activity breakdown card."""
    total = sum(stats.values())
    if total == 0:
        return ""

    rows = ""
    for act, (color, icon) in ACTIVITY_PALETTE.items():
        secs = stats.get(act, 0)
        if secs == 0:
            continue
        pct = min(secs / total * 100, 100)
        rows += f"""
<div class="stat-row">
    <div class="stat-label-row">
        <span>{icon} {act}</span>
        <span style="color:#64748b;">{fmt_duration(int(secs))}</span>
    </div>
    <div class="stat-bar-track">
        <div class="stat-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
    </div>
</div>"""

    return textwrap.dedent(f"""
        <div class="emp-card">
            <div class="emp-name">{name}</div>
            <div class="emp-total">Total session: {fmt_duration(int(total))}</div>
            {rows}
        </div>
    """)

@st.fragment(run_every=3)
def live_desk_grid():
    try:
        resp = requests.get(BACKEND_URL, timeout=1.5)
        resp.raise_for_status()
        desks: dict = resp.json()
    except Exception as e:
        st.error(f"Cannot reach backend: {e}")
        return

    if not desks:
        st.info("⏳ Waiting for camera data…")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    total     = len(desks)
    occupied  = sum(1 for d in desks.values() if d.get("status") == "Occupied")
    working   = sum(1 for d in desks.values() if d.get("activity") == "Working")
    on_phone  = sum(1 for d in desks.values() if d.get("activity") == "Using Mobile")
    standing  = sum(1 for d in desks.values() if d.get("activity") == "Standing")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Desks", total)
    m2.metric("Occupied",    occupied)
    m3.metric("Working",     working)
    m4.metric("Standing",    standing)
    m5.metric("On Mobile",   on_phone)

    st.divider()

    # ── Desk cards — ONE st.markdown call per column, not per card ────────────
    sorted_desks  = sorted(desks.items())
    col_html      = [""] * NUM_COLUMNS   # one HTML string per column

    for idx, (desk_id, data) in enumerate(sorted_desks):
        col_html[idx % NUM_COLUMNS] += build_card_html(desk_id, data)

    cols = st.columns(NUM_COLUMNS)
    for col_idx, col in enumerate(cols):
        with col:
            if col_html[col_idx]:
                st.markdown(col_html[col_idx], unsafe_allow_html=True)

    # ── Employee Activity Stats ───────────────────────────────────────────────
    try:
        stats_resp = requests.get("http://localhost:8000/employee_stats", timeout=1.5)
        emp_stats: dict = stats_resp.json() if stats_resp.ok else {}
    except Exception:
        emp_stats = {}

    if emp_stats:
        st.divider()
        st.markdown("### 📊 Employee Activity Breakdown")
        st.markdown(
            '<p class="subtitle">Accumulated time per activity this session</p>',
            unsafe_allow_html=True,
        )
        stat_cols = st.columns(min(len(emp_stats), NUM_COLUMNS))
        for col_idx, (name, stats) in enumerate(sorted(emp_stats.items())):
            with stat_cols[col_idx % NUM_COLUMNS]:
                html = build_employee_stats_html(name, stats)
                if html:
                    st.markdown(html, unsafe_allow_html=True)


live_desk_grid()