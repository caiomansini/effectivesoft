print("✅ chat_ui.py loaded")
import streamlit as st
from ui.sidebar import sidebar_and_documentChooser
from utils.llm_handler import get_response_from_LLM

def chatbot():
    st.title("🤖 EffectiveSoft Sales Assistant")

    # Sidebar upload and config
    sidebar_and_documentChooser()

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about a customer, opportunity, or pitch..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call the full fallback logic from handler
        answer, sources, source_type = get_response_from_LLM(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown("### 🤖 Answer")
            st.markdown(answer)

            # Show source docs if available
        if source_type:
            label_map = {
                "internal": "Internal documents",
                "web": "Web search",
                "internal+web": "Internal + Web sources"
            }
            readable_label = label_map.get(source_type, source_type.capitalize())
            st.markdown(f"#### 📚 Source: **{readable_label}**")

        # Save assistant reply in history
        st.session_state.messages.append({"role": "assistant", "content": answer})