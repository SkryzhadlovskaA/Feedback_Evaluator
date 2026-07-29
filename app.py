"""Streamlit UI for the Feedback Evaluator pipeline.

Run from project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.pipeline import run_pipeline
from src.taxonomy import PROJECT_TOPICS

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
UPLOAD_DIR = RAW_DIR / "uploads"


def list_raw_files() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.csv"))


def render_section(title: str, items: list):
    st.subheader(title)
    if not items:
        st.info("No items found.")
        return

    for item in items:
        with st.expander(f"{item['label']} ({item['count']})"):
            for quote in item.get("representative_quotes", []):
                st.write(f"• {quote}")


st.set_page_config(page_title="Feedback Evaluator", page_icon="📋", layout="wide")

st.title("Feedback Evaluator")
st.caption("Analyze Erasmus+ participant feedback and generate a structured summary.")

with st.sidebar:
    st.header("Input data")
    source = st.radio("Choose input", ["Existing file", "Upload CSV"])

    input_path = None

    if source == "Existing file":
        files = list_raw_files()
        if not files:
            st.warning("No CSV files in data/raw/")
        else:
            labels = [f.name for f in files]
            choice = st.selectbox("File", labels)
            input_path = str(RAW_DIR / choice)
    else:
        uploaded = st.file_uploader("Upload feedback CSV", type=["csv"])
        if uploaded:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            save_path = UPLOAD_DIR / uploaded.name
            save_path.write_bytes(uploaded.getvalue())
            input_path = str(save_path)

    st.divider()
    st.markdown("**Required columns:**")
    st.code("project_id, response_id, question, text")

    with st.expander("Configured project topics"):
        if PROJECT_TOPICS:
            for pid, topic in PROJECT_TOPICS.items():
                st.write(f"**{pid}**")
                st.caption(topic["label"])
        else:
            st.caption("None — add entries in src/taxonomy.py")

run_clicked = st.button("Analyze feedback", type="primary", disabled=not input_path)

if run_clicked and input_path:
    status = st.status("Running pipeline...", expanded=True)

    try:
        result = run_pipeline(input_path, on_step=status.write)
        status.update(label="Done", state="complete", expanded=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Strengths groups", len(result.get("strengths", [])))
        col2.metric("Improvement groups", len(result.get("improvements", [])))
        col3.metric("Learning outcomes", len(result.get("learning_outcomes", [])))

        if result.get("project_id"):
            st.write(f"**Project ID:** `{result['project_id']}`")
        if result.get("project_topic_category"):
            st.success(f"Project topic: {result['project_topic_category']}")
        elif result.get("project_id"):
            st.warning(
                f"No PROJECT_TOPICS entry for `{result['project_id']}`. "
                "Add it in src/taxonomy.py for the 11th learning category."
            )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Strengths", "Improvements", "Learning outcomes", "JSON"]
        )

        with tab1:
            render_section("Strengths", result.get("strengths", []))
        with tab2:
            render_section("Improvements", result.get("improvements", []))
        with tab3:
            render_section("Learning outcomes", result.get("learning_outcomes", []))
        with tab4:
            display = {k: v for k, v in result.items() if not k.startswith("_")}
            st.json(display)
            st.download_button(
                "Download JSON",
                data=json.dumps(display, indent=2, ensure_ascii=False),
                file_name=f"{result.get('project_name', 'output')}_final_output.json",
                mime="application/json",
            )

        st.caption(f"Saved to: `{result.get('_output_path', '')}`")

    except Exception as e:
        status.update(label="Failed", state="error", expanded=True)
        st.error(str(e))

elif not input_path:
    st.info("Select or upload a CSV file, then click **Analyze feedback**.")
