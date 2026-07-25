import os
import tempfile

import streamlit as st

from inference.predict import FloodPredictor
from reports.recommendations import get_recommendation

from ui.sidebar import show_sidebar
from ui.metrics import show_metrics
from ui.display import show_images


st.set_page_config(
    page_title="Flood Mapping",
    layout="wide"
)

show_sidebar()

st.title("🌊 Flood Mapping using Sentinel-1 SAR")

uploaded = st.file_uploader(
    "Upload Sentinel-1 TIFF",
    type=["tif", "tiff"]
)

if uploaded is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".tif"
    ) as tmp:

        tmp.write(uploaded.read())

        path = tmp.name

    predictor = FloodPredictor()

    with st.spinner("Running AI models..."):

        result = predictor.predict(path)

    os.remove(path)

    if not result["flood_detected"]:

        st.success("No Flood Detected")

    else:

        severity, rec = get_recommendation(
            result["statistics"]["flood_percentage"]
        )

        show_images(result)

        show_metrics(result, severity)

        st.subheader("Recovery Recommendations")

        for r in rec:

            st.write("✅", r)