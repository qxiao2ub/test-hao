\
import io
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

from pubg_lib import train_pipeline, predict_winplace, make_submission

# Notebook rendering deps
import nbformat
from nbconvert import HTMLExporter

APP_TITLE = "PUBG WinPlacePerc Prediction App"

COPYRIGHT_LINE = "© 2026 Hao Ning. All rights reserved."
SUPERVISOR_LINE = "Supervisor: Qingyang Xiao"


def render_footer() -> None:
    st.markdown("---")
    st.caption(f"{COPYRIGHT_LINE}  •  {SUPERVISOR_LINE}")


@st.cache_data(show_spinner=False)
def ipynb_bytes_to_html(ipynb_bytes: bytes) -> str:
    nb = nbformat.reads(ipynb_bytes.decode("utf-8"), as_version=4)
    exporter = HTMLExporter()
    # We only want charts/tables, not the code (change to False if you want code too)
    exporter.exclude_input = True
    body, _ = exporter.from_notebook_node(nb)
    return body


def page_home():
    st.title(APP_TITLE)
    st.write(
        "Upload **train.csv** and **test.csv** to train a model and generate **submission.csv**. "
        "You can also upload a pre-trained model for prediction only."
    )

    st.info(
        "Tip: For the full PUBG dataset, **train online may be slow** on free hosting. "
        "For production, train offline once and use **Prediction Only** mode."
    )

    with st.expander("Required columns", expanded=False):
        st.markdown(
            """
- **train.csv** must contain the target column: `winPlacePerc`  
- **test.csv** must contain: `Id`  
- Other PUBG original features are recommended (`matchId`, `groupId`, numeric stats, etc.)
            """
        )

    render_footer()


def page_train_predict():
    st.header("Train & Predict")
    st.write("Upload train/test CSV, train a model, and download a Kaggle-style submission file.")

    train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_csv")
    test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_csv")

    model_name = st.selectbox("Model", ["Ridge"], index=0)
    valid_size = st.slider("Validation split ratio", 0.05, 0.4, 0.2, 0.05)

    run = st.button("Run training + prediction", type="primary", disabled=not (train_file and test_file))

    if run:
        with st.spinner("Reading CSV files..."):
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

        st.subheader("Data preview")
        st.write("Train sample:")
        st.dataframe(train_df.head(10), use_container_width=True)
        st.write("Test sample:")
        st.dataframe(test_df.head(10), use_container_width=True)

        with st.spinner("Training model..."):
            result = train_pipeline(train_df, model_name=model_name, valid_size=valid_size)
            st.success(f"Training complete. Validation MAE: {result.valid_mae:.5f}")

        with st.spinner("Predicting & building submission..."):
            preds = predict_winplace(result.pipe, test_df)
            sub = make_submission(test_df, preds)

        st.subheader("submission.csv preview")
        st.dataframe(sub.head(20), use_container_width=True)

        st.download_button(
            "Download submission.csv",
            data=sub.to_csv(index=False).encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )

        # Optional: download the trained model
        buf = io.BytesIO()
        joblib.dump(result.pipe, buf)
        st.download_button(
            "Download trained model (model.joblib)",
            data=buf.getvalue(),
            file_name="model.joblib",
            mime="application/octet-stream",
        )

    render_footer()


def page_predict_only():
    st.header("Prediction Only")
    st.write("Upload a trained model file (joblib) and a test.csv to generate submission.csv.")

    model_file = st.file_uploader("Upload model.joblib", type=["joblib", "pkl"], key="model_joblib")
    test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_csv_only")

    run = st.button("Run prediction", type="primary", disabled=not (model_file and test_file))

    if run:
        with st.spinner("Loading model & reading test.csv..."):
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            pipe = joblib.load(tmp_path)
            test_df = pd.read_csv(test_file)

        with st.spinner("Predicting & building submission..."):
            preds = predict_winplace(pipe, test_df)
            sub = make_submission(test_df, preds)

        st.subheader("submission.csv preview")
        st.dataframe(sub.head(20), use_container_width=True)

        st.download_button(
            "Download submission.csv",
            data=sub.to_csv(index=False).encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )

    render_footer()


def page_notebook_visuals():
    st.header("Notebook Visualizations & Tables")
    st.write(
        "This page renders the original notebook outputs (charts and tables) inside the web app.\n\n"
        "You can either use the bundled notebook file or upload your updated notebook."
    )

    default_nb_path = Path(__file__).parent / "notebook_report.ipynb"
    source = st.radio(
        "Notebook source",
        ["Use bundled notebook (notebook_report.ipynb)", "Upload a notebook (.ipynb)"],
        horizontal=True,
    )

    ipynb_bytes = None
    if source.startswith("Use bundled"):
        if default_nb_path.exists():
            ipynb_bytes = default_nb_path.read_bytes()
            st.caption(f"Using: {default_nb_path.name}")
        else:
            st.error("Bundled notebook file not found. Please upload a notebook.")
    else:
        nb_file = st.file_uploader("Upload .ipynb", type=["ipynb"], key="nb_upload")
        if nb_file is not None:
            ipynb_bytes = nb_file.read()

    if ipynb_bytes:
        with st.spinner("Rendering notebook to HTML..."):
            html = ipynb_bytes_to_html(ipynb_bytes)

        height = st.slider("Viewer height", 600, 2000, 1000, 100)
        components.html(html, height=height, scrolling=True)

        st.info(
            "If some charts are missing, make sure your notebook is **executed and saved with outputs** "
            "(so figures/tables exist in the .ipynb file)."
        )

    render_footer()


def page_about():
    st.header("About")
    st.write(APP_TITLE)
    st.write(COPYRIGHT_LINE)
    st.write(SUPERVISOR_LINE)

    st.markdown(
        """
### What this app does
- Trains a tabular regression model for PUBG `winPlacePerc`
- Generates `submission.csv` for Kaggle-style evaluation
- Renders your original notebook's charts & tables inside the app

### Tech stack
- Python, pandas, scikit-learn
- Streamlit for the web UI
- nbconvert to render `.ipynb` as HTML
        """
    )

    render_footer()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Train & Predict", "Prediction Only", "Notebook Visuals", "About"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(COPYRIGHT_LINE)
    st.sidebar.caption(SUPERVISOR_LINE)

    if page == "Home":
        page_home()
    elif page == "Train & Predict":
        page_train_predict()
    elif page == "Prediction Only":
        page_predict_only()
    elif page == "Notebook Visuals":
        page_notebook_visuals()
    else:
        page_about()


if __name__ == "__main__":
    main()
