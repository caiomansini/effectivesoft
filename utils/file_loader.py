import os
import shutil
import pandas as pd
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def delte_temp_files(path="data/tmp"):
    """
    Clears the temporary document folder.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def langchain_document_loader(file_path):
    """
    Loads a document based on file type.
    """
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def split_documents_to_chunks(documents, chunk_size=512, chunk_overlap=64):
    """
    Splits loaded documents into manageable chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def load_and_split_all_documents(file_paths):
    """
    Loads and chunks all uploaded documents.
    """
    all_docs = []
    for file_path in file_paths:
        docs = langchain_document_loader(file_path)
        all_docs.extend(docs)
    
    return split_documents_to_chunks(all_docs)