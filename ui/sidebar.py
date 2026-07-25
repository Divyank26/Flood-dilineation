import streamlit as st


def show_sidebar():

    st.sidebar.title("Flood Mapping")

    st.sidebar.markdown("---")

    st.sidebar.info(
        """
AI-based Flood Mapping System

• CNN Flood Classification

• Attention U-Net

• Sentinel-1 SAR

• Recovery Recommendation
"""
    )

    st.sidebar.markdown("---")

    st.sidebar.write("Version 2.0")