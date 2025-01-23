"""
RAG (Retrieval Augmented Generation) Core Implementation

This module implements the core RAG functionality, combining document retrieval with
language model generation. It uses Pinecone for vector storage and GPT-4 for response generation.

Key components:
- Vector similarity search with configurable thresholds
- Cached embeddings for performance
- Dynamic prompt enhancement based on query type
- Multi-level error handling and logging
"""

import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.globals import set_verbose, set_debug
from langchain.cache import InMemoryCache
from typing import List, Tuple
import logging
from functools import lru_cache
from langchain_core.documents import Document
from langchain.globals import set_llm_cache
from sklearn.metrics.pairwise import cosine_similarity
from semantic_chunker import BertSemanticChunker

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable caching to improve response times
cache = InMemoryCache()
set_llm_cache(cache)

# Load environment variables and configure LangChain settings
load_dotenv()
set_verbose(False)
set_debug(False)

# Initialize core components
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model="gpt-4-turbo-preview",
    temperature=0.7,  # Controls response creativity
    cache=True  # Enable LLM response caching
)
semantic_chunker = BertSemanticChunker()

@lru_cache(maxsize=1000)
def get_embeddings(text: str) -> List[float]:
    """Cache embeddings for frequently asked questions to improve performance.
    
    Args:
        text: The text to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_query(text)

def get_relevant_context(
    vectorstore: PineconeVectorStore,
    query: str,
    threshold: float = 0.75,
    k: int = 3
) -> Tuple[str, List[Tuple[Document, float]]]:
    """Retrieve and filter relevant context from the vector store.
    
    Args:
        vectorstore: Pinecone vector store instance
        query: User's question
        threshold: Minimum similarity score (0-1)
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (combined context string, list of (document, score) pairs)
    """
    # Perform similarity search
    response = vectorstore.similarity_search_with_score(query, k=k)
    
    # Filter results by similarity threshold
    filtered_responses = [(doc, score) for doc, score in response if score > threshold]
    
    if not filtered_responses:
        return "", []
    
    # Combine context from filtered documents
    context_parts = [doc.page_content for doc, _ in filtered_responses]
    return "\n\n".join(context_parts), filtered_responses

@lru_cache(maxsize=100)
def get_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    embeddings = OpenAIEmbeddings()
    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

def enhance_prompt(query: str, context: str) -> str:
    """Dynamically enhance the prompt based on query type for better responses.
    
    Args:
        query: User's question
        context: Retrieved context
        
    Returns:
        Enhanced prompt template
    """
    base_template = """
    You are a helpful AI assistant answering questions based on the provided context.
    Please provide accurate, concise answers based only on the context provided.
    
    Context:
    {context}
    
    Question: {question}
    
    Additional Instructions:
    1. Only use information from the provided context
    2. If uncertain, acknowledge the limitation
    3. Format lists and prices clearly
    4. Highlight key features and benefits
    5. Keep responses clear and concise
    
    Answer:"""
    
    # Add specialized instructions based on query type
    if "cost" in query.lower() or "price" in query.lower():
        base_template += "\nInclude all relevant pricing details and any available discounts."
    elif "service" in query.lower():
        base_template += "\nClearly differentiate between different service tiers and features."
        
    return base_template

def answer_question(
    vectorstore: PineconeVectorStore,
    question: str,
    threshold: float = 0.78
) -> Tuple[str, List[Tuple[Document, float]]]:
    """Process a question and generate an answer using RAG.
    
    Args:
        vectorstore: Pinecone vector store instance
        question: User's question
        threshold: Minimum similarity score for context filtering
        
    Returns:
        Tuple of (generated answer, list of source documents with scores)
    """
    try:
        # Get relevant context
        context, responses = get_relevant_context(vectorstore, question, threshold=threshold)
        if not context:
            return ("I apologize, but I don't have enough relevant information to answer your question accurately.", [])
        
        # Generate response using enhanced prompt
        enhanced_template = enhance_prompt(question, context)
        prompt = ChatPromptTemplate.from_template(enhanced_template)
        prompt_formatted_str = prompt.format(context=context, question=question)
        answer = model.predict(prompt_formatted_str)
        
        return answer, responses
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return ("I encountered an error while processing your question. Please try again.", [])

def main():
    # Initialize vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name="sashank-trial",
        embedding=embeddings,
        namespace="sashank-3"
    )
    
    # Example questions
    questions = [
        "What is caregiver"
    ]
    
    # Clean output format
    print("\n=== teleCalm Q&A System ===\n")
    
    for question in questions:
        print(f"Q: {question}")
        answer, responses = answer_question(vectorstore, question)
        print(f"A: {answer}\n")
        
        print("Relevant Sources:")
        for doc, score in responses:
            print(f"Score: {score:.3f}")
            print(f"Content: {doc.page_content[:200]}...")
            print("-" * 80 + "\n")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()