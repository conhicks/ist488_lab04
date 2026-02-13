__import__('pysqlite3')
import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib
import Path
from PyPDF2
import PdfReader
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chromadb.utils import embedding_functions
import os
from pathlib import Path

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def create_vector_db():
    """Create and populate ChromaDB collection with PDF documents"""
    
    chroma_client = chromadb.EphemeralClient()
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=openai_ef
    )
    
    pdf_folder = Path("pdfs")
    
    if not pdf_folder.exists():
        st.error("PDF folder not found.  Create a 'pdfs' folder with your PDF files.")
        st.stop()
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        st.error("No PDF files found in the pdfs folder!")
        st.stop()
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, pdf_path in enumerate(pdf_files):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            documents.append(text)
            metadatas.append({
                "filename": pdf_path.name,
                "page_count": len(pdf_reader.pages)
            })
            ids.append(f"doc_{idx}")
        except Exception as e:
            st.warning(f"Error reading {pdf_path.name}: {e}")
    
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    return collection

st.title("Course Information Chatbot")
st.markdown("Ask questions about course materials and get answers powered by RAG")

if 'Lab4_VectorDB' not in st.session_state:
    with st.spinner("Loading course documents into vector database"):
        st.session_state.Lab4_VectorDB = create_vector_db()
    st.success("Vector database loaded successfully")

if 'lab4_messages' not in st.session_state:
    st.session_state.lab4_messages = []

for message in st.session_state.lab4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the course materials..."):
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    results = st.session_state.Lab4_VectorDB.query(
        query_texts=[prompt],
        n_results=3
    )
    
    relevant_docs = results['documents'][0]
    relevant_files = [meta['filename'] for meta in results['metadatas'][0]]
    
    context_parts = []
    for file, doc in zip(relevant_files, relevant_docs):
        truncated_doc = doc[:2000] if len(doc) > 2000 else doc
        context_parts.append(f"Document: {file}\n{truncated_doc}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    enhanced_prompt = f"""You are a helpful course information assistant. Use the following course materials to answer the user's question."""