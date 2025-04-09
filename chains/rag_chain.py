from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from utils.file_loader import load_and_split_all_documents
from retrievers.setup import select_embeddings_model
import streamlit as st

def answer_template():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for answering questions about customer documents.
Use the following context to answer the question. If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}
Answer:"""
    )

def chain_RAG_blocks(llm, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": answer_template()},
        return_source_documents=True,
        verbose=False
    )

def create_vectorstore_from_uploaded_documents(persist_dir="data/vectorstore"):
    if "uploaded_file_paths" not in st.session_state or not st.session_state.uploaded_file_paths:
        st.warning("No uploaded files found.")
        return None

    # Load and chunk all uploaded docs
    docs = load_and_split_all_documents(st.session_state.uploaded_file_paths)

    # Select embedding model
    embedding_model = select_embeddings_model()

    # Save to Chroma
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    vectorstore.persist()
    st.success(f"Vectorstore created with {len(docs)} chunks.")
    return vectorstore