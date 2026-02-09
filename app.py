import streamlit as st

st.set_page_config(layout="wide")

st.title("⚽ Player Tracking Analytics")

st.markdown("""
Welcome to the analytics dashboard.

Use the sidebar to navigate to different reports:

- Heatmap
- Goalkeeper intervals
- Speed zones
- Fatigue analysis
""")

st.markdown("---")

st.info("All reports are generated from the same session data.")
