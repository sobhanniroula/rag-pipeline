# RAG Pipeline

A hands-on learning repository for building **Retrieval-Augmented Generation (RAG)** pipelines from scratch. Built as a personal reference and sandbox to explore the core components of modern AI document search and question-answering systems.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a pattern where:

1. Documents are split into chunks and converted into vector embeddings.
2. Those embeddings are stored in a vector database.
3. At query time, the most relevant chunks are retrieved via semantic similarity search.
4. A Large Language Model (LLM) uses the retrieved context to generate a grounded answer.

---

## Project Structure

```
rag-pipeline/
├── main.py                        # Entry point (placeholder)
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Pip-installable dependencies
├── data/
│   ├── pdf_files/                 # Place PDF documents here for ingestion
│   ├── text_files/                # Sample plain text documents
│   │   ├── sample.txt
│   │   └── sample2.txt
│   └── vector_store/              # Persisted ChromaDB vector store
│       └── chroma.sqlite3
└── notebook/
    ├── document.ipynb             # Tutorial: document loaders & text splitters
    ├── pdf_loader.ipynb           # Tutorial: embeddings + ChromaDB vector store
    └── typesense.ipynb            # Full RAG app: Typesense + Groq LLM
```

---

## Notebooks Overview

### 1. `document.ipynb` — Data Ingestion

Covers the first stage of a RAG pipeline: loading and splitting documents.

- Loading `.txt` files with `TextLoader`
- Loading entire directories with `DirectoryLoader`
- Loading `.pdf` files with `PyPDFLoader` and `PyMuPDFLoader`
- Understanding the `Document` object structure (page content + metadata)

### 2. `pdf_loader.ipynb` — Embeddings & Vector Store

Covers converting documents into vectors and storing them locally.

- Splitting documents with `RecursiveCharacterTextSplitter`
- Generating embeddings with `SentenceTransformer` (`all-MiniLM-L6-v2`)
- Storing and querying embeddings in **ChromaDB** (local, persistent)
- Cosine similarity search using HNSW index

### 3. `typesense.ipynb` — Full RAG Application

End-to-end RAG app using a cloud vector store and a hosted LLM.

- Loading and splitting text documents
- Generating embeddings (configurable; `FakeEmbeddings` used during development)
- Storing vectors in **Typesense Cloud** via LangChain's `Typesense` vector store
- Performing semantic search against the vector store
- Answering questions using the **Groq** LLM API (fast hosted inference)

---

## Tech Stack

### Framework

| Library                 | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `langchain`             | Core RAG orchestration framework                              |
| `langchain-core`        | Base abstractions (Document, Runnable, etc.)                  |
| `langchain-community`   | Third-party integrations (loaders, vector stores, embeddings) |
| `langchain-groq`        | LangChain integration for the Groq LLM API                    |
| `langchain-huggingface` | LangChain integration for HuggingFace embeddings              |

### LLM

| Service  | Notes                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Groq** | Cloud-hosted ultra-fast inference. Used in `typesense.ipynb`. Requires `GROQ_API_KEY` in `.env`. Supports models like LLaMA 3. |

### Embeddings

| Model / Class                            | Notes                                                               |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `sentence-transformers/all-MiniLM-L6-v2` | Lightweight, 384-dim HuggingFace model. Used in `pdf_loader.ipynb`. |
| `FakeEmbeddings` (langchain-community)   | Random vectors for development/testing. No API key needed.          |

### Vector Databases

| Database            | Type                                                | Used In                  |
| ------------------- | --------------------------------------------------- | ------------------------ |
| **ChromaDB**        | Local, file-persisted (`data/vector_store/`)        | `pdf_loader.ipynb`       |
| **Typesense Cloud** | Cloud, hosted vector + keyword search               | `typesense.ipynb`        |
| **FAISS**           | In-memory, CPU-based (installed, available for use) | Available as alternative |

### Document Loading & Processing

| Library                                 | Purpose                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| `langchain-community` `TextLoader`      | Load `.txt` files                                          |
| `langchain-community` `DirectoryLoader` | Bulk-load from a directory                                 |
| `pypdf` / `PyPDFLoader`                 | Load and parse PDF files                                   |
| `pymupdf` / `PyMuPDFLoader`             | Faster/richer PDF parsing alternative                      |
| `CharacterTextSplitter`                 | Split by character with chunk size/overlap                 |
| `RecursiveCharacterTextSplitter`        | Smarter splitting respecting paragraph/sentence boundaries |

### Other

| Library         | Purpose                               |
| --------------- | ------------------------------------- |
| `python-dotenv` | Load secrets from `.env` file         |
| `tqdm`          | Progress bars during batch operations |
| `numpy`         | Vector math and array operations      |
| `scikit-learn`  | Cosine similarity calculations        |

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TYPESENSE_HOST=your-cluster.a1.typesense.net
TYPESENSE_API_KEY=your_typesense_api_key
```

---

## Setup

**Requirements:** Python 3.13+

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Key Concepts Learned

- **Document** — LangChain's base unit: `page_content` (str) + `metadata` (dict)
- **Text Splitting** — Chunking large documents so they fit in LLM context windows
- **Embeddings** — Dense vector representations of text that encode semantic meaning
- **Vector Store** — Database optimised for similarity search over embedding vectors
- **Retriever** — Component that takes a query and returns the top-k relevant chunks
- **RAG Chain** — Combines retriever + LLM prompt + LLM into a single callable pipeline

---

## Notes

- `huggingface_hub` 1.x has a known import bug (`cannot import name 'logging'`) that breaks `sentence-transformers` in certain environments. Use `FakeEmbeddings` for local development or pin `huggingface_hub==1.12.0`.
- Typesense requires a live cloud cluster. The free tier at [cloud.typesense.org](https://cloud.typesense.org) is enough for learning.
- Groq offers a free tier with generous rate limits — ideal for experimentation.
