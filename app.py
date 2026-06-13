import streamlit as st
from core.config_user import PROJECT_TITLE, COPYRIGHT_TEXT, CUSTOM_ALGOS

st.set_page_config(page_title="FYP Dashboard", layout="wide")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

st.title("🎓 Student FYP Dashboard")
st.header(f"Project: {PROJECT_TITLE}")

if CUSTOM_ALGOS:
    st.success(f"Custom Algorithms Loaded: {', '.join(CUSTOM_ALGOS)}")

st.info("👈 Use the Sidebar to open **Training Lab**.")
