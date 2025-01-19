import streamlit as st
from rag import *

# Load models and index
st.title("Research Paper Query Assistant")
st.write("This tool helps you retrieve and answer questions from research papers.")

# Initialize models and index
with st.spinner("Loading models and embeddings..."):
    embedding_model, llm, index, flat_chunks = load_models_and_index()

# Input form
query = st.text_input("Enter your query:", placeholder="e.g., Write here")

if st.button("Get Response"):
    if query.strip():
        with st.spinner("Processing your query..."):
            try:
                # Use the RAG system pipeline
                response = query_rag_system(query, embedding_model, llm)
                st.success("Response Generated!")
                st.write("### Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid query!")
