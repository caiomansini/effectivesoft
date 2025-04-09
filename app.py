import streamlit as st
from ui.chat_ui import chatbot

# Optional: Set up page metadata
st.set_page_config(
    page_title="EffectiveSoft Sales Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Launch the chatbot UI
if __name__ == "__main__":
    chatbot()