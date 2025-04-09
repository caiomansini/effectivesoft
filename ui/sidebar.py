import streamlit as st
from utils.config import OPENAI_API_KEY, TAVILY_API_KEY
from utils.file_loader import delte_temp_files
from chains.rag_chain import create_vectorstore_from_uploaded_documents, chain_RAG_blocks
from retrievers.setup import select_embeddings_model
from memory.memory import create_memory
from langchain.chat_models import ChatOpenAI

def sidebar_and_documentChooser():
    st.sidebar.title("⚙️ Settings")

    # Load keys from .env fallback
    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", value=OPENAI_API_KEY, type="password"
    )
    st.session_state.tavily_api_key = st.sidebar.text_input(
        "Tavily API Key", value=TAVILY_API_KEY, type="password"
    )

    # Model settings
    st.session_state.selected_model = st.sidebar.selectbox(
        "Model", ["gpt-4o", "gpt-3.5-turbo"]
    )

    # ✅ NEW: Embedding model selector
    st.session_state.embeddings_model = st.sidebar.selectbox(
        "Embedding Model", ["openai", "huggingface"], index=1
    )

    st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
    st.session_state.max_tokens = st.sidebar.slider("Max Tokens", 256, 4096, 1024)
    st.session_state.top_p = st.sidebar.slider("Top P", 0.1, 1.0, 1.0)

    st.sidebar.markdown("---")

    # Upload multiple files
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents", 
        type=["pdf", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        delte_temp_files()
        uploaded_paths = []
        for uploaded_file in uploaded_files:
            temp_path = f"data/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_paths.append(temp_path)
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        
        st.session_state.uploaded_file_paths = uploaded_paths

    # Button to build vectorstore and RAG chain
    if st.sidebar.button("🛠️ Build Vectorstore"):
        vectorstore = create_vectorstore_from_uploaded_documents()
        if vectorstore:
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

            # Build LLM
            llm = ChatOpenAI(
                model=st.session_state.selected_model,
                api_key=st.session_state.openai_api_key,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                model_kwargs={"top_p": st.session_state.top_p}
            )

            # Build memory
            memory = create_memory(
                model_name=st.session_state.selected_model,
                api_key=st.session_state.openai_api_key
            )

            # Build the full chain
            st.session_state.chain = chain_RAG_blocks(llm, st.session_state.retriever, memory)
            st.sidebar.success("✅ RAG chain is ready!")

def clear_chat_history():
    st.session_state.chat_history = []
