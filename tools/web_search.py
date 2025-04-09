import streamlit as st
from langchain.chat_models import ChatOpenAI

def get_direct_llm_response(prompt):
    """Fallback to direct LLM response (no retrieval)."""
    llm = ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        model=st.session_state.selected_model,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        model_kwargs={"top_p": st.session_state.top_p},
    )
    return llm.invoke(prompt).content