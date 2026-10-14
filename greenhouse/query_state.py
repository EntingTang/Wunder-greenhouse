"""Restore and share the active view without replacing crop-specific state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st

from greenhouse.data import DUALPAM_LABEL, LIGHT_RESPONSE_GROUP, VARIABLE_GROUPS


QUERY_KEYS = ("crop", "analysis", "variable", "group")
SNAPSHOT_KEY = "greenhouse__query_snapshot"
PENDING_KEY = "greenhouse__query_selection"


def _query_snapshot() -> tuple[str | None, ...]:
    return tuple(st.query_params.get(key) for key in QUERY_KEYS)


def _choice(value: str | None, options: Sequence[str], default: str) -> str:
    requested = (value or "").strip().casefold()
    return next((option for option in options if option.casefold() == requested), default)


def parse_view_query(
    params: Mapping[str, str], crops: Sequence[str], analyses: Sequence[str],
) -> dict[str, str | None]:
    """Accept known labels/columns and infer the group for response-only links."""

    variable = params.get("variable")
    if variable in ("Fv/Fm_LSA", "FvFm1_LSA"):
        variable = DUALPAM_LABEL
    inferred_group = LIGHT_RESPONSE_GROUP
    for group, variables in VARIABLE_GROUPS.items():
        if any((variable or "").strip().casefold() in (label.casefold(), column.casefold())
               for label, column in variables):
            inferred_group = group
            break
    return {
        "crop": _choice(params.get("crop"), crops, crops[0]),
        "analysis": _choice(params.get("analysis"), analyses, analyses[0]),
        "variable": variable,
        "group": _choice(params.get("group"), list(VARIABLE_GROUPS), inferred_group),
    }


def restore_query_selection(crops: Sequence[str], analyses: Sequence[str]) -> None:
    """Read incoming links once; URL writes made by this app don't rehydrate widgets."""

    snapshot = _query_snapshot()
    previous = st.session_state.get(SNAPSHOT_KEY)
    if previous == snapshot:
        return
    st.session_state[SNAPSHOT_KEY] = snapshot
    if previous is None and not any(value is not None for value in snapshot):
        return
    selection = parse_view_query(st.query_params.to_dict(), crops, analyses)
    st.session_state["greenhouse__crop"] = selection["crop"]
    st.session_state["greenhouse__analysis"] = selection["analysis"]
    st.session_state[PENDING_KEY] = selection


def _pending_selection(crop: str) -> dict:
    pending = st.session_state.get(PENDING_KEY, {})
    if (pending.get("crop") == crop
            and pending.get("analysis") == st.session_state.get("greenhouse__analysis")):
        return pending
    return {}


def restore_query_group(crop: str, state_key: str) -> None:
    pending = _pending_selection(crop)
    if pending:
        st.session_state[state_key] = pending["group"]


def restore_query_variable(crop: str, state_key: str, labels_by_column: Mapping[str, str]) -> None:
    pending = _pending_selection(crop)
    if not pending:
        return
    requested = str(pending.get("variable") or "").strip().casefold()
    for column, label in labels_by_column.items():
        if requested in (column.casefold(), label.casefold()):
            st.session_state[state_key] = column
            break
    st.session_state.pop(PENDING_KEY, None)


def sync_query_selection(
    crop: str, analysis: str, variable: str | None = None, group: str | None = None,
) -> None:
    """Write current choices while preserving unrelated/repeated URL parameters."""

    values = {"crop": crop, "analysis": analysis}
    if variable is not None:
        values["variable"] = variable
        if group is not None:
            values["group"] = group
        elif "group" in st.query_params:
            del st.query_params["group"]
    changed = {key: value for key, value in values.items() if st.query_params.get(key) != value}
    if changed:
        st.query_params.update(changed)
    st.session_state[SNAPSHOT_KEY] = _query_snapshot()
