import asyncio
import time
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Desk Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ───────────────────────────────────────────────────────────────
_state_lock: asyncio.Lock | None = None   
desk_states:     Dict[str, Dict[str, Any]]   = {}
employee_stats:  Dict[str, Dict[str, float]] = {}  

# How long (seconds) before an unseen desk is marked Vacant
VACANCY_TIMEOUT = 5.0


@app.on_event("startup")
async def _startup():
    global _state_lock
    _state_lock = asyncio.Lock()


def _get_lock() -> asyncio.Lock:
    """Return the lock, raising clearly if startup hasn't run yet."""
    if _state_lock is None:
        raise RuntimeError("State lock not initialised — did startup run?")
    return _state_lock


# ── Models ─────────────────────────────────────────────────────────────────────
class StatusUpdate(BaseModel):
    desk_id:     str
    status:      str
    activity:    str
    person_name: str = "Unknown"
    time:        int = 0


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/update")
async def update_status(data: List[StatusUpdate]):
    """
    Receive the current frame's desk data.
    Desks NOT in this payload are NOT immediately removed —
    they are marked Vacant after VACANCY_TIMEOUT seconds.
    """
    now      = time.time()
    seen_ids = {d.desk_id for d in data}

    async with _get_lock():
        # Update / insert desks reported in this frame
        for d in data:
            desk_states[d.desk_id] = {
                "status":      d.status,
                "activity":    d.activity,
                "person_name": d.person_name,
                "time":        d.time,
                "last_seen":   now,
            }

        for desk_id in list(desk_states.keys()):
            if desk_id not in seen_ids:
                state = desk_states[desk_id]
                age = now - state.get("last_seen", now)
                if age > VACANCY_TIMEOUT:
                    state["status"]      = "Vacant"
                    state["activity"]    = "—"
                    state["person_name"] = "—"
                    state["time"]        = 0

    return {"msg": "Updated", "active_desks": len(seen_ids)}


@app.get("/logs")
async def get_logs():
    """Return all known desks (occupied + recently vacated)."""
    async with _get_lock():
        return {
            desk_id: {k: v for k, v in state.items() if k != "last_seen"}
            for desk_id, state in desk_states.items()
        }


@app.post("/activity_log")
async def update_activity_log(data: Dict[str, Dict[str, float]]):
    """
    Receive per-employee accumulated activity durations (in seconds) from main.py.
    Each key is a person name; each value maps activity name → total seconds.
    We store the MAXIMUM value seen for each (person, activity) pair so that
    accumulated totals never go backwards even if main.py restarts.
    """
    async with _get_lock():
        for person, activities in data.items():
            if person not in employee_stats:
                employee_stats[person] = {}
            for activity, seconds in activities.items():
                if seconds > employee_stats[person].get(activity, 0):
                    employee_stats[person][activity] = round(seconds, 1)
    return {"msg": "Activity log updated"}


@app.get("/employee_stats")
async def get_employee_stats():
    """Return accumulated per-employee activity durations."""
    async with _get_lock():
        return dict(employee_stats)


@app.delete("/reset")
async def reset():
    """Clear all desk state and activity logs (useful for testing)."""
    async with _get_lock():
        desk_states.clear()
        employee_stats.clear()
    return {"msg": "Reset complete"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)