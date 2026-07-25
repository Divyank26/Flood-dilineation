import streamlit as st


def severity_color(severity):

    if severity == "LOW":
        return "🟢"

    if severity == "MODERATE":
        return "🟡"

    if severity == "HIGH":
        return "🟠"

    return "🔴"


def show_metrics(result, severity):

    stats = result["statistics"]

    st.subheader("Flood Statistics")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Flood Presence Confidence",
            f"{result['classifier_probability']*100:.2f}%"
        )

        st.metric(
            "Flood Area",
            f"{stats['area_km2']:.3f} km²"
        )

    with c2:

        st.metric(
            "Flood Coverage",
            f"{stats['flood_percentage']:.2f}%"
        )

        st.metric(
            "Inference Time",
            f"{result['inference_time']:.2f} sec"
        )

    st.markdown(
        f"## {severity_color(severity)} Severity : **{severity}**"
    )