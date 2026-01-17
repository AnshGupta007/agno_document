# Local Document Q&A with Agno Framework Setup Guide

This guide will help you set up the local document Q&A system using the Agno framework with Ollama models.

## Prerequisites

### 1. Install Ollama
Follow the [Ollama installation guide](https://ollama.ai/) for your operating system:

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

### 2. Pull Required Models
After installing Ollama, download the required models:

```bash
# Main language model for Q&A
ollama pull qwen3:8b

# Alternative models you can try:
# ollama pull llama3.1:8b
# ollama pull mistral:7b
# ollama pull phi3:mini
```

### 3. Install Python Dependencies
Install the required Python packages:

```bash
pip install -U agno ollama sentence-transformers
```

Or install all dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Setup Steps

### 1. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Verify Ollama Installation
Make sure Ollama is running and the model is available:
```bash
ollama list
ollama run qwen3:8b "Hello, how are you?"
```

### 3. Add Documents
Place your documents in the `agno_document_qa/documents/` folder:
- Supported formats: `.txt`, `.md`, `.json`, `.docx`, `.py`, `.js`, `.html`, `.css`
- The system will automatically process and index them

### 4. Run the Agent
```bash
cd agno_document_qa
python agno_agent.py
```

## Key Features

### 🔒 **Complete Privacy**
- All processing happens locally
- No data sent to external APIs
- Works completely offline

### 🤖 **Local AI Models**
- **Language Model**: Ollama (qwen3:8b by default)
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Search**: FAISS for fast similarity search

### 🌍 **Multilingual Support**
- Ask questions in any language
- Responses in the same language as your query
- Local models handle multiple languages

### 📚 **Smart Document Processing**
- Automatic text chunking with overlap
- Semantic search across all documents
- Local LLM selects most relevant chunks

## Usage Examples

```
🧑 You: What is this document about?
🤖 Assistant: [Analyzes documents and provides summary with sources]

🧑 You: ¿Cuáles son las características principales?
🤖 Assistant: [Responds in Spanish with relevant information]

🧑 You: このシステムはどのように動作しますか？
🤖 Assistant: [Responds in Japanese with technical details]
```

## Configuration Options

### Change the Language Model
Edit `agno_agent.py` and modify the model name:
```python
agent = create_agno_document_agent(model_name="llama3.1:8b")
```

### Change the Embedding Model
Edit `local_document_search.py` and modify the embedding model:
```python
search_engine = LocalDocumentSemanticSearch(
    embedding_model="all-mpnet-base-v2",  # More accurate but slower
    llm_model="qwen3:8b"
)
```

### Adjust Chunk Settings
Modify chunk size and overlap in `local_document_search.py`:
```python
search_engine = LocalDocumentSemanticSearch(
    chunk_size=1500,    # Larger chunks
    chunk_overlap=300   # More overlap
)
```

## Troubleshooting

### Common Issues

**1. "Model not found" error:**
```bash
ollama pull qwen3:8b
```

**2. "Connection refused" error:**
Make sure Ollama is running:
```bash
ollama serve
```

**3. Slow embedding generation:**
The first run will be slower as it downloads the embedding model. Subsequent runs will be faster.

**4. Out of memory errors:**
Try using a smaller model:
```bash
ollama pull qwen2.5:1.5b
```

### Performance Tips

1. **Use GPU acceleration** if available (Ollama will automatically use GPU)
2. **Adjust chunk size** based on your document types
3. **Use smaller models** for faster responses on limited hardware
4. **Pre-process documents** by running the system once to generate embeddings

## Model Recommendations

### For Different Hardware:

**High-end systems (16GB+ RAM):**
- `qwen2.5:7b` or `llama3.1:8b`
- `all-mpnet-base-v2` embeddings

**Mid-range systems (8-16GB RAM):**
- `qwen3:8b` (default)
- `all-MiniLM-L6-v2` embeddings (default)

**Low-end systems (4-8GB RAM):**
- `qwen2.5:1.5b` or `phi3:mini`
- `all-MiniLM-L6-v2` embeddings

## File Structure

```
agno_document_qa/
├── agno_agent.py              # Main Agno agent
├── local_document_search.py   # Local RAG implementation
├── documents/                 # Your documents go here
├── embeddings/                # Generated embeddings (auto-created)
└── setup_guide.md            # This guide
```

## Next Steps

1. Add your documents to the `documents/` folder
2. Run `python agno_agent.py`
3. Start asking questions about your documents!

The system will automatically:
- Process and chunk your documents
- Generate embeddings using local models
- Create a searchable index
- Provide accurate answers with source citations
