import os
import logging
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from semantic_chunker import BertSemanticChunker

# Constants
BATCH_SIZE = 100

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_chunks(text: str) -> List[Document]:
    """Create semantically meaningful chunks from text."""
    chunker = BertSemanticChunker()
    semantic_chunks = chunker.create_chunks(text)
    
    return [
        Document(
            page_content=chunk.text,
            metadata={"semantic_score": chunk.score}
        ) for chunk in semantic_chunks
    ]

def batch_upsert_to_pinecone(docs: List[Document], batch_size: int = BATCH_SIZE) -> None:
    """Batch upload documents to Pinecone with error handling."""
    embeddings = OpenAIEmbeddings()
    index_name = "sashank-trial"
    namespace = "sashank-3"
    
    # Process in batches
    for i in tqdm(range(0, len(docs), batch_size), desc="Uploading to Pinecone"):
        batch = docs[i:i + batch_size]
        try:
            PineconeVectorStore.from_documents(
                batch, 
                embeddings, 
                index_name=index_name, 
                namespace=namespace
            )
        except Exception as e:
            logger.error(f"Error uploading batch {i//batch_size}: {str(e)}")
            # Continue with next batch instead of failing completely
            continue

def process_documents(input_dir: str) -> List[Document]:
    """Process all documents in the input directory."""
    all_docs = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = create_chunks(content)
                all_docs.extend(chunks)
                logger.info(f"Processed {filename}: {len(chunks)} chunks created")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                
    return all_docs

def main():
    # Get the current working directory
    raw_file_dir = os.getcwd()
    input_parsed_file_dir = os.path.join(raw_file_dir, "parsed_docs")
    
    # Process all documents
    logger.info("Starting document processing...")
    all_docs = process_documents(input_parsed_file_dir)
    logger.info(f"Total chunks created: {len(all_docs)}")
    
    # Upload to Pinecone
    logger.info("Starting Pinecone upload...")
    batch_upsert_to_pinecone(all_docs)
    logger.info("Upload complete!")

if __name__ == "__main__":
    main()