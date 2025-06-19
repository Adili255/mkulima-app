import streamlit as st

def display_navigation():
    selected = st.selectbox(
        "Navigate", ["Home", "Analysis", "Prediction"],
        index=0,
        key="nav_top",
    )
    return selected