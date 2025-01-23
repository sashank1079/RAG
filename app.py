"""
Streamlit Chat Interface for RAG Assistant

This module provides a web interface for the RAG-based chat assistant using Streamlit.
It handles user interactions, message history, and displays responses with source attribution.

Features:
- Interactive chat interface
- Message history management
- Source attribution with relevance scores
- Cached RAG initialization
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rag import answer_question

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_rag():
    """Initialize and cache RAG components.
    
    This function is cached by Streamlit to avoid reinitializing
    expensive components on each rerun.
    
    Returns:
        PineconeVectorStore: Initialized vector store
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name="sashank-trial",
        embedding=embeddings,
        namespace="sashank-3"
    )
    return vectorstore

def main():
    """Main application function."""
    st.title("🤖 RAG Assistant")
    
    # Initialize RAG components (cached)
    vectorstore = initialize_rag()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources", expanded=False):
                    # Display top 3 most relevant sources
                    for doc, score in message["sources"][:3]:
                        st.markdown(f"**Score:** {score:.3f}")
                        st.markdown(f"{doc.page_content[:150]}...")
    
    # Chat input and response generation
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response and sources from RAG
                response, sources = answer_question(vectorstore, prompt)
                st.markdown(response)
                
                # Show sources in expandable section
                if sources:
                    with st.expander("View Sources"):
                        for doc, score in sources:
                            st.markdown(f"**Score:** {score:.3f}")
                            st.markdown(f"{doc.page_content[:150]}...")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()
