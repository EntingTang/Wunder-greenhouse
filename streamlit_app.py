"""Streamlit entry point for the greenhouse data dashboard."""

from pathlib import Path

import streamlit as st

from greenhouse.ui import render_crop_panel
from greenhouse.model_validation import render_model_validation
from greenhouse.query_state import restore_query_selection, sync_query_selection


APP_ROOT = Path(__file__).resolve().parent
DATA_FILES = {
    "Potato": APP_ROOT / "data" / "Potato_LC_data_for_ts_plots.xlsx",
    "Tomato": APP_ROOT / "data" / "Tomato_LC_data_for_ts_plots.xlsx",
}
MEASUREMENT_FILES = {
    "Potato": APP_ROOT / "data" / "Potato_meteo_HH2_LSA_biomass_OJIP_tillHarvest_LSA_updated.xlsx",
    "Tomato": APP_ROOT / "data" / "LMM_tomato_input_tillHarvest_LSA_updated.xlsx",
}
MODEL_FILES = {
    "Potato": APP_ROOT / "data" / "Potato_ye_ETR_YII_model_results.xlsx",
    "Tomato": APP_ROOT / "data" / "Tomato_ye_ETR_YII_results.xlsx",
}
ANALYSES = ["Explore Data", "Compare Data", "Model Validation"]
READ_ONLY_STATE_SUFFIXES = (
    "__interactive_plot",
    "_button",
    "_download",
)


st.set_page_config(
    page_title="Greenhouse data explorer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
restore_query_selection(list(DATA_FILES), ANALYSES)

# Keep crop-specific widget values when only the selected crop panel is rendered.
for state_key in list(st.session_state):
    is_crop_state = state_key.startswith("greenhouse__")
    is_read_only = state_key.endswith(READ_ONLY_STATE_SUFFIXES)
    if is_crop_state and not is_read_only:
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
      .st-key-response_variables div[role="radiogroup"] {
        flex-wrap: wrap;
        overflow-x: visible;
      }
      .st-key-response_variables div[role="radiogroup"] label {
        flex: 0 0 auto;
        width: auto;
        min-width: max-content;
        white-space: nowrap;
      }
      .st-key-response_variables [data-testid="stMarkdownContainer"] p {
        white-space: nowrap;
      }
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
      @media (max-width: 768px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Greenhouse data explorer")

with st.sidebar:
    st.header("View controls")
    selected_crop = st.radio(
        "Crop",
        list(DATA_FILES),
        key="greenhouse__crop",
    )
    selected_analysis = st.radio(
        "Analysis",
        ANALYSES,
        key="greenhouse__analysis",
    )

sync_query_selection(selected_crop, selected_analysis)

if selected_analysis == "Model Validation":
    render_model_validation(selected_crop, MODEL_FILES[selected_crop])
else:
    render_crop_panel(
        selected_crop, DATA_FILES[selected_crop], MEASUREMENT_FILES[selected_crop], selected_analysis
    )

