import streamlit as st
import os
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq



# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to extract text from all PDFs in a folder
def extract_text_from_folder(folder_path):
    all_text = {}
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"Processing: {pdf_path}")
            all_text[file] = extract_text_from_pdf(pdf_path)
    return all_text


# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'Page\s+\d+\s+(of\s+\d+)?', '', text, flags=re.IGNORECASE)  # Remove page numbers
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess spaces
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if len(line.strip()) > 10]  # Remove short lines
    return "\n".join(cleaned_lines)


# Function to clean all texts
def clean_all_texts(pdf_texts):
    cleaned_texts = {}
    for pdf_name, text in pdf_texts.items():
        print(f"Cleaning text for: {pdf_name}")
        cleaned_texts[pdf_name] = clean_text(text)
    return cleaned_texts



# Function to chunk text into smaller sections
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_token = 0

    for word in words:
        if current_token + len(word) + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token = 0
        current_chunk.append(word)
        current_token += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks



# Function to chunk all cleaned texts
def chunk_all_texts(cleaned_texts, max_length=500):
    chunked_texts = {}
    for pdf_name, text in cleaned_texts.items():
        print(f"Chunking text for: {pdf_name}")
        chunked_texts[pdf_name] = chunk_text(text, max_length)
    return chunked_texts


# Load and process PDF folder
folder_path = r"C:\Users\soura\OneDrive\Desktop\Projects\GenerativeAI-Projects\ResearchPaper-Query-RAG\ResearchPaper-Data"
pdf_texts = extract_text_from_folder(folder_path)
cleaned_pdf_texts = clean_all_texts(pdf_texts)
chunked_texts = chunk_all_texts(cleaned_pdf_texts)


# Initialize models and FAISS index
@st.cache_resource
def load_models_and_index():
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    flat_chunks = [chunk for pdf_chunks in chunked_texts.values() for chunk in pdf_chunks]

    # Initialize ChatGroq LLM
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    # Define FAISS index 
    dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'
    index = faiss.IndexFlatL2(dimension)

    embeddings = embedding_model.encode(flat_chunks)
    index.add(np.array(embeddings))

    return embedding_model, llm, index, flat_chunks


embedding_model, llm, index, flat_chunks = load_models_and_index()


# Helper functions
def retrieve_relevant_chunks(query, model, index, chunks, top_k=5):
    """
    Retrieve the most relevant text chunks for a given query.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    return [chunks[i] for i in indices[0]]


def combine_chunks(relevant_chunks, max_length=3000):
    """
    Combine text chunks into a single context string within a token limit.
    """
    combined_text = ""
    for chunk in relevant_chunks:
        if len(combined_text) + len(chunk) <= max_length:
            combined_text += chunk + "\n"
        else:
            break
    return combined_text


def generate_response(query, context, llm):
    """
    Generate a response using the LLM with the provided context.
    """
    prompt = f"""
    You are a helpful AI assistant. Use the context provided to answer the question accurately. 
    If you do not have enough information to answer the question, say 'I don't have enough information to answer this question.'
    
    Context:
    {context}
    
    Question: {query}
    Answer:
    """
    # Invoke the language model
    response = llm.invoke(input=prompt, max_tokens=300)

    # Access the content attribute of the response
    return response.content.strip() if hasattr(response, 'content') else str(response).strip()

def query_rag_system(query, embedding_model, llm, max_context_length=3000):
    """
    Full RAG system pipeline: retrieves context and generates a response.
    """
    # Step 1: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, embedding_model, index, flat_chunks, top_k=5)
    
    # Step 2: Combine retrieved chunks into context
    context = combine_chunks(relevant_chunks, max_length=max_context_length)
    
    # Step 3: Generate a response
    response = generate_response(query, context, llm)
    
    return response
