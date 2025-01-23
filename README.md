# RAG Chat Assistant

A Retrieval-Augmented Generation (RAG) based chat assistant built with Streamlit, LangChain, and Pinecone. This application enables intelligent document retrieval and question answering using state-of-the-art language models.

## Features

- 🔍 Semantic search with optimized BERT-based chunking
- 💾 Vector storage using Pinecone
- 🤖 GPT-4 Turbo for response generation
- 🚀 Streamlit-based chat interface
- 📝 Source attribution for responses
- 🔄 Caching for improved performance
- 📚 Parallel document processing with LlamaParse

## Technical Architecture

### RAG Implementation
- Uses LangChain for orchestrating the RAG pipeline
- Implements semantic chunking using BERT for better context understanding
- Utilizes Pinecone for efficient vector similarity search
- Employs GPT-4 Turbo for generating contextually relevant responses

### Components
1. **Knowledge Base Parser** (`rag_kb_parsing.py`):
   - Parallel document processing using ThreadPoolExecutor
   - Support for multiple file formats via LlamaParse
   - Batch processing with configurable batch size
   - Robust error handling and logging
   - Converts documents to markdown format for consistency
   - Place your documents in the `input_docs` folder for parsing

2. **Semantic Chunker** (`semantic_chunker.py`):
   - BERT-based text chunking for semantic understanding
   - Configurable chunk size and overlap
   - GPU acceleration support
   - Automatic batch processing

3. **RAG Core** (`rag.py`):
   - Context retrieval with configurable thresholds
   - Multi-level caching system
   - Dynamic prompt enhancement
   - Error handling and logging

4. **Web Interface** (`app.py`):
   - Streamlit-based chat interface
   - Session state management
   - Source attribution display
   - Response caching

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

### Document Processing
1. Place your documents in the `input_docs` directory
2. Run the knowledge base parser:
```bash
python rag_kb_parsing.py
```
3. Processed documents will be saved in `parsed_docs` directory
4. Supported file formats include PDF, DOCX, PPTX, and more

### Configuring the Vector Store
1. Create a Pinecone index with 1536 dimensions (OpenAI embedding size)
2. Update the index name and namespace in `rag.py`
3. Add your documents to the vector store using the provided utilities

### Customizing the Model
- Adjust the `temperature` parameter in `rag.py` for response creativity
- Modify the chunk size and overlap in `semantic_chunker.py`
- Configure the similarity threshold in `get_relevant_context()`
- Adjust `BATCH_SIZE` in `rag_kb_parsing.py` for parallel processing

### Advanced Features
- **Caching**: Implement custom caching strategies by modifying the `@lru_cache` decorators
- **Prompt Templates**: Customize response formats by editing the `enhance_prompt()` function
- **Logging**: Configure logging levels in `rag.py` for debugging
- **Document Processing**: Customize file filtering and batch processing in `rag_kb_parsing.py`

## Performance Optimizations

- Uses `bert-tiny` model for faster semantic chunking
- Implements caching at multiple levels:
  - Embedding cache
  - LLM response cache
  - Streamlit component cache
- GPU acceleration when available
- Optimized vector search parameters
- Batch processing for document ingestion
- Parallel document processing with ThreadPoolExecutor

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for embeddings and chat completion
- `PINECONE_API_KEY`: Pinecone API key for vector storage
- `LLAMA_CLOUD_API_KEY`: LlamaParse API key for document processing
- Optional environment variables:
  - `CHUNK_SIZE`: Override default chunk size
  - `CHUNK_OVERLAP`: Override default chunk overlap
  - `SIMILARITY_THRESHOLD`: Override default similarity threshold
  - `BATCH_SIZE`: Override default document processing batch size

## Troubleshooting

Common issues and solutions:
1. **Memory Issues**: Reduce chunk size or batch size
2. **Slow Response**: Adjust caching parameters or reduce context window
3. **GPU Errors**: Check CUDA installation or fall back to CPU
4. **Token Limits**: Adjust chunk size or implement token counting
5. **Document Processing Errors**: Check file permissions and supported formats


