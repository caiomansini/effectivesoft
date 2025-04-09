print("✅ chat_ui.py loaded")
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from ui.sidebar import sidebar_and_documentChooser, clear_chat_history
from utils.config import DEFAULT_MODEL
from chains.rag_chain import chain_RAG_blocks
from memory.memory import create_memory
from retrievers.setup import select_embeddings_model
from tools.internal_lookup import safe_internal_lookup
from tools.web_search import get_direct_llm_response
from retrievers.setup import is_question_covered_by_docs


def get_response_from_LLM(prompt):
    """
    Handles logic for querying:
    1. RAG chain (if relevant)
    2. Sales Agent (for proposal or reasoning)
    3. Direct LLM fallback
    """
    if not prompt:
        return "", []

    # Priority 1: If RAG chain has relevant data, use it
    if "chain" in st.session_state and "retriever" in st.session_state:
        from retrievers.setup import is_question_covered_by_docs
        if is_question_covered_by_docs(prompt):
            try:
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result.get("answer", "I couldn't find an answer.")
                source_docs = result.get("source_documents", [])
                return answer, source_docs
            except Exception as e:
                st.error(f"Error during RAG: {e}")

    # Priority 2: Try the agent
    if "sales_agent" in st.session_state:
        try:
            result = st.session_state.sales_agent.invoke({"input": prompt})
            return result.get("output", "Agent could not generate an answer."), []
        except Exception as e:
            st.warning(f"Agent error: {e}")

    # Priority 3: Direct LLM fallback
    try:
        from tools.web_search import get_direct_llm_response
        return get_direct_llm_response(prompt), []
    except Exception as e:
        return f"Error using direct LLM fallback: {e}", []

def chatbot():
    """
    Streamlit UI for interacting with the chatbot.
    """
    sidebar_and_documentChooser()
    st.title("EffectiveSoft Chatbot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_response_from_LLM(prompt)
                st.markdown(answer)

                if sources:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.code(doc.page_content[:1000])

        st.session_state.chat_history.append({"role": "assistant", "content": answer})