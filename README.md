# RAG Study Assistant

An AI-powered study tool that lets you upload course PDFs and ask questions about them using Retrieval-Augmented Generation (RAG).

## How It Works
1. Upload a PDF (textbook, lecture notes, etc.)
2. The document is chunked and embedded using `nomic-embed-text` via Ollama
3. Embeddings are stored in a ChromaDB vector store
4. When you ask a question, relevant chunks are retrieved and passed to `llama3.2` to generate an answer

## Tech Stack
- **LLM**: llama3.2 (via Ollama, runs locally)
- **Embeddings**: nomic-embed-text (via Ollama)
- **Vector Store**: ChromaDB
- **Orchestration**: LangChain
- **UI**: Streamlit

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- Python 3.10+

### Install
```bash
git clone https://github.com/mukhilt/RAG-Study-Assistant.git
cd RAG-Study-Assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pull Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Run
```bash
streamlit run app.py
```