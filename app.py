import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rag import answer_question

# Load environment variables
load_dotenv()

# Set up the page configuration
st.set_page_config(
    page_title="teleCalm RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_rag():
    """Initialize RAG components with caching"""
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name="sashank-trial",
        embedding=embeddings,
        namespace="sashank-3"
    )
    return vectorstore

def main():
    st.title("🤖 teleCalm Assistant")
    
    # Initialize RAG components (now cached)
    vectorstore = initialize_rag()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:  # Only show if sources exist
                with st.expander("View Sources", expanded=False):  # Default collapsed
                    for doc, score in message["sources"][:3]:  # Limit to top 3 sources
                        st.markdown(f"**Score:** {score:.3f}")
                        st.markdown(f"{doc.page_content[:150]}...")  # Show less content
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about teleCalm?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = answer_question(vectorstore, prompt)
                st.markdown(response)
                
                # Show sources in an expander
                if sources:
                    with st.expander("View Sources"):
                        for doc, score in sources:
                            st.markdown(f"**Relevance Score:** {score:.3f}")
                            st.markdown(f"**Content:** {doc.page_content[:200]}...")
                            st.markdown("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()
