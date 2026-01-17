# Agno Document Q&A System

A powerful document question-answering system that combines Groq's fast language models with local embeddings for privacy and performance.

## 🚀 Features

- **Fast AI Models**: Uses Groq's high-speed language models (openai/gpt-oss-20b)
- **Local Embeddings**: SentenceTransformers for privacy-preserving document processing
- **Multiple Interfaces**: Web UI (Streamlit) and Command-line interface
- **Multi-format Support**: .txt, .md, .json, .docx, .py, .js, .html, .css files
- **RAG Architecture**: Retrieval-Augmented Generation for accurate answers
- **Source Attribution**: Always cites document sources for answers

## 📋 Quick Start

### 1. Environment Setup

```bash
# Set your Groq API key
export GROQ_API_KEY="your-groq-api-key-here"

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Documents

Place your documents in the `documents/` folder:
```
agno_document_qa/
├── documents/
│   ├── your_document.txt
│   ├── research_paper.md
│   └── code_file.py
```

### 3. Run the System

**Option A: Web Interface (Recommended)**
```bash
python main.py --web
```

**Option B: Command Line Interface**
```bash
python main.py --cli
```

**Option C: Setup Check**
```bash
python main.py --setup
```

## 🖥️ Web Interface

The Streamlit web interface provides:

- **Interactive Chat**: Ask questions about your documents
- **Document Management**: View loaded documents and their status
- **Model Selection**: Choose from different Groq models
- **Source Attribution**: See which documents were used for each answer
- **Sample Questions**: Quick-start examples

Access at: `http://localhost:8501`

## 🛠️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Agno Agent     │    │  Groq API       │
│                 │◄──►│                  │◄──►│                 │
│  - Chat Interface│    │ - Tool Calling   │    │ - LLM Inference │
│  - File Upload   │    │ - Response Gen   │    │ - Fast Models   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Local Search     │
                       │                  │
                       │ - SentenceTransf │
                       │ - FAISS Index    │
                       │ - Document Chunks│
                       └──────────────────┘
```

## 📁 File Structure

```
agno_document_qa/
├── main.py                 # Main entry point
├── streamlit_app.py        # Web interface
├── agno_agent.py          # Agno agent configuration
├── local_document_search.py # Search engine
├── requirements.txt        # Dependencies
├── README.md              # This file
├── documents/             # Your documents go here
└── embeddings/            # Generated embeddings (auto-created)
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)

### Model Options

Available Groq models:
- `openai/gpt-oss-20b` (default)
- `llama-3.3-70b-versatile`
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

### Supported File Types

- Text files: `.txt`, `.md`
- Code files: `.py`, `.js`, `.html`, `.css`
- Documents: `.docx`, `.json`

## 🚀 Usage Examples

### Web Interface
1. Start the web app: `python main.py --web`
2. Upload documents to the `documents/` folder
3. Initialize the agent
4. Ask questions like:
   - "What is the main topic of these documents?"
   - "Can you summarize the key points?"
   - "What technologies are mentioned?"

### Command Line
```bash
python main.py --cli
# Then type your questions interactively
```

### Programmatic Usage
```python
from agno_agent import create_agno_document_agent

agent = create_agno_document_agent()
response = agent.run("What are the main features described?")
print(response.content)
```

## 🔍 How It Works

1. **Document Processing**: Documents are chunked and embedded using local SentenceTransformers
2. **Semantic Search**: User queries are embedded and matched against document chunks using FAISS
3. **Context Selection**: Groq LLM selects the most relevant chunks for answering
4. **Response Generation**: Agno agent generates responses using selected context
5. **Source Attribution**: System provides citations for all answers

## 🛠️ Troubleshooting

### Common Issues

**"GROQ_API_KEY not set"**
```bash
export GROQ_API_KEY="your-key-here"
```

**"No documents found"**
- Add documents to the `documents/` folder
- Ensure files have supported extensions

**"Agent initialization failed"**
- Check internet connection
- Verify API key is valid
- Install all requirements

### Performance Tips

- Use smaller documents for faster processing
- The system automatically caches embeddings
- Web interface is recommended for better UX

## 📊 System Requirements

- Python 3.8+
- 4GB+ RAM (for embeddings)
- Internet connection (for Groq API)
- ~2GB disk space (for models)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check this README
2. Run `python main.py --setup` for diagnostics
3. Check the console output for error messages
4. Ensure all requirements are installed
