import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

def select_embeddings_model():
    if st.session_state.embeddings_model == "openai":
        return OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def is_question_covered_by_docs(query):
    retriever = st.session_state.retriever
    results = retriever.get_relevant_documents(query)
    return any(results)

def is_low_relevance(query, threshold=0.35):
    retriever = st.session_state.retriever
    results = retriever.get_relevant_documents(query)
    return all(doc.metadata.get("score", 1.0) < threshold for doc in results)
