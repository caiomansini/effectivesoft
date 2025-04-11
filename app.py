import streamlit as st
from ui.chat_ui import chatbot
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# App-level configuration
st.set_page_config(
    page_title="EffectiveSoft Assistant",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Entry point
if __name__ == "__main__":
    chatbot()