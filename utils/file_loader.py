import os
import shutil
import re
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

def extract_company_from_filename(file_path):
    """
    Dynamically extracts company name from filename using common delimiters.
    E.g., 'Boli AI - Project.pdf' → 'Boli Ai'
    """
    filename = os.path.splitext(os.path.basename(file_path))[0]
    parts = re.split(r'[_\-\s]+', filename)
    company_name = " ".join(parts[:2]).strip().title()
    return company_name if company_name else "Unknown"

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
    Loads and chunks all uploaded documents with automatic metadata.
    """
    all_docs = []
    for file_path in file_paths:
        docs = langchain_document_loader(file_path)
        company_name = extract_company_from_filename(file_path)
        for doc in docs:
            doc.metadata["company"] = company_name
            doc.metadata["source_file"] = os.path.basename(file_path)
        all_docs.extend(docs)
    return split_documents_to_chunks(all_docs)