"""Streamlit entry point for the greenhouse light-response curve dashboard."""

from pathlib import Path

import streamlit as st

from greenhouse.ui import render_crop_panel


APP_ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "Potato": APP_ROOT / "data" / "Potato_LC_data_for_ts_plots.xlsx",
    "Tomato": APP_ROOT / "data" / "Tomato_LC_data_for_ts_plots.xlsx",
}


st.set_page_config(
    page_title="Greenhouse curve explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep crop-specific widget values when only the selected crop panel is rendered.
for state_key in list(st.session_state):
    if state_key.startswith("greenhouse__"):
        st.session_state[state_key] = st.session_state[state_key]

st.markdown(
    """
    <style>
      :root {
        --ink: #1d2924;
        --muted: #5f6f67;
        --line: #dce5df;
        --surface: #f7faf7;
        --accent: #34735a;
      }
      html, body, .stApp {
        font-family: "Geist", "Aptos", "Segoe UI", sans-serif;
        color: var(--ink);
      }
      .material-symbols-rounded,
      [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
      }
      .stApp { background: #fbfdfb; }
      .block-container {
        max-width: 1680px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
      }
      h1, h2, h3, h4 { letter-spacing: -0.025em; }
      h1 { font-size: clamp(2.1rem, 4vw, 3.65rem) !important; line-height: 1 !important; }
      h3 { margin-top: 0.2rem !important; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      div[role="radiogroup"] {
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 0.2rem 0 0.55rem;
        scrollbar-width: thin;
      }
      div[role="radiogroup"] > label {
        flex: 0 0 auto;
        border-radius: 999px;
        transition: transform 180ms cubic-bezier(0.16, 1, 0.3, 1), border-color 180ms ease;
      }
      div[role="radiogroup"] > label:active { transform: translateY(1px); }
      .section-rule {
        border-top: 1px solid var(--line);
        margin: 0.25rem 0 1rem;
      }
      [data-testid="stDownloadButton"] button {
        border-color: #b9c9bf;
        transition: transform 180ms cubic-bezier(0.16, 1, 0.3, 1), background 180ms ease;
      }
      [data-testid="stDownloadButton"] button:active { transform: translateY(1px); }
      [data-testid="stAlertContainer"] { border-radius: 0.75rem; }
      [data-baseweb="tag"] {
        background-color: #e5eee8 !important;
        color: #20352b !important;
      }
      [data-baseweb="tag"] span { color: #20352b !important; }
      .date-period-key {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.2rem 0 0.55rem;
      }
      .date-period-key span {
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.26rem 0.62rem;
      }
      .date-period-before { background: #d9eaf4; color: #214c68; }
      .date-period-after { background: #f6e4c9; color: #704515; }
      @media (max-width: 768px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([1.5, 1], vertical_alignment="bottom")
with header_left:
    st.title("Greenhouse curve explorer")
with header_right:
    st.markdown(
        "Explore plant-level light-response curves or compare treatment means with "
        "sample-SD uncertainty. Filters and selections stay independent for each crop."
    )

with st.sidebar:
    st.header("View controls")
    st.caption("Choose a crop first, then select an analysis mode.")
    selected_crop = st.radio(
        "Crop",
        list(DATA_FILES),
        key="greenhouse__crop",
    )
    selected_analysis = st.radio(
        "Analysis",
        ["Explore Data", "Compare Data"],
        key=f"greenhouse__{selected_crop.lower()}__function",
    )

render_crop_panel(selected_crop, DATA_FILES[selected_crop], selected_analysis)

