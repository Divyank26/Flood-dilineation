import streamlit as st
import numpy as np


def show_images(result):

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        if result["mask"] is not None:

            st.image(
                result["mask"],
                caption="Predicted Flood Mask",
                use_container_width=True
            )

    with col2:

        if result["overlay"] is not None:

            st.image(
                result["overlay"],
                caption="Flood Overlay",
                use_container_width=True
            )