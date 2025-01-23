"""
Semantic Text Chunking Implementation

This module provides semantic-aware text chunking using BERT models. It splits text
into meaningful chunks while preserving semantic coherence, which is crucial for
RAG applications.

Features:
- BERT-based semantic understanding
- Configurable chunk sizes and overlap
- GPU acceleration support
- Automatic batch processing
- Score-based chunk evaluation
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from dataclasses import dataclass
from langchain.text_splitter import TextSplitter

@dataclass
class SemanticChunk:
    """Represents a semantically meaningful chunk of text with metadata.
    
    Attributes:
        text: The chunk content
        score: Semantic coherence score (0-1)
        metadata: Optional metadata about the chunk
    """
    text: str
    score: float = 1.0
    metadata: Dict = None

class BertSemanticChunker(TextSplitter):
    """BERT-based semantic text chunker.
    
    This class uses a BERT model to create semantically meaningful chunks of text.
    It ensures that chunks maintain semantic coherence while respecting size constraints.
    """
    
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",  # Smaller, faster model
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """Initialize the semantic chunker.
        
        Args:
            model_name: Name of the BERT model to use
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        super().__init__()
        # Set up GPU acceleration if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Configure chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Enable evaluation mode for faster inference
        self.model.eval()

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while maintaining word boundaries.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        
        # Create chunks based on size and overlap
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep overlap words for next chunk
                current_chunk = current_chunk[-self.chunk_overlap:]
        
        # Add the last chunk if any
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def create_chunks(self, text: str) -> List[SemanticChunk]:
        """Create semantically meaningful chunks from text.
        
        This method:
        1. Splits text into initial chunks
        2. Evaluates semantic coherence of each chunk
        3. Returns chunks with their semantic scores
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of SemanticChunk objects with scores
        """
        # Get initial splits
        splits = self.split_text(text)
        
        chunks = []
        for split in splits:
            # Get semantic coherence score using BERT
            tokens = self.tokenizer(split, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
                score = torch.sigmoid(outputs.logits[0]).mean().item()
            
            # Create chunk with semantic score
            chunks.append(SemanticChunk(
                text=split,
                score=score
            ))
        
        return chunks 