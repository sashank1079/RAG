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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable caching
cache = InMemoryCache()
set_llm_cache(cache)

load_dotenv()
set_verbose(False)
set_debug(False)

# Initialize models and chunker
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model="gpt-4-turbo-preview",
    temperature=0.7,
    cache=True
)
semantic_chunker = BertSemanticChunker()

@lru_cache(maxsize=1000)
def get_embeddings(text: str) -> List[float]:
    """Cache embeddings for frequently asked questions."""
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_query(text)

def get_relevant_context(
    vectorstore: PineconeVectorStore,
    query: str,
    threshold: float = 0.75,
    k: int = 3
) -> Tuple[str, List[Tuple[Document, float]]]:
    """Get relevant context with improved filtering and reranking."""
    # Get initial results
    response = vectorstore.similarity_search_with_score(query, k=k)
    
    # Filter by similarity threshold
    filtered_responses = [(doc, score) for doc, score in response if score > threshold]
    
    if not filtered_responses:
        return "", []
    
    # Combine context
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
    """Dynamically enhance the prompt based on query type."""
    base_template = """
    You are a helpful AI assistant answering questions about teleCalm's services and products.
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
    """Process question and return answer with relevant context."""
    try:
        context, responses = get_relevant_context(vectorstore, question, threshold=threshold)
        if not context:
            return ("I apologize, but I don't have enough relevant information to answer your question accurately.", [])
        
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