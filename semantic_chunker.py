from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from dataclasses import dataclass
from langchain.text_splitter import TextSplitter

@dataclass
class SemanticChunk:
    text: str
    score: float = 1.0
    metadata: Dict = None

class BertSemanticChunker(TextSplitter):
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",  # Smaller, faster model
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Enable evaluation mode for faster inference
        self.model.eval()

    def split_text(self, text: str) -> List[str]:
        """Required implementation of abstract method."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep overlap words for next chunk
                current_chunk = current_chunk[-self.chunk_overlap:]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def create_chunks(self, text: str) -> List[SemanticChunk]:
        """Create semantically meaningful chunks from text."""
        # Get initial splits
        splits = self.split_text(text)
        
        chunks = []
        for split in splits:
            # Get semantic score
            tokens = self.tokenizer(split, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
                score = torch.sigmoid(outputs.logits[0]).mean().item()
            
            chunks.append(SemanticChunk(
                text=split,
                score=score
            ))
        
        return chunks 