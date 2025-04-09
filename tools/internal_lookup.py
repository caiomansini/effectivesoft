import streamlit as st

def safe_internal_lookup(query):
    """RAG-based lookup tool with hallucination filtering."""
    if not st.session_state.chain:
        return "Internal knowledge base is not loaded."

    result = st.session_state.chain.invoke({"question": query})
    answer = result.get("answer", "")

    fallback_phrases = [
        "no relevant", "i do not know", "context does not",
        "i'm sorry", "not found", "not enough info", "cannot provide"
    ]

    if any(phrase in answer.lower() for phrase in fallback_phrases):
        return "No relevant internal data found."
    return answer