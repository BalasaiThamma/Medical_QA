# Medical Q&A RAG System using Weaviate, BGE Embeddings & Ollama

A complete Retrieval-Augmented Generation (RAG) application built using FastAPI, Weaviate, Sentence Transformers, and Ollama. This project retrieves semantically relevant medical Question-Answer pairs from a vector database and generates grounded answers using locally hosted Large Language Models.

## 🚀 Features

- 🔍 Semantic Search using **BAAI/bge-base-en-v1.5**
- 🗄️ Vector Database with **Weaviate**
- 🤖 Local LLM inference using **Ollama**
- ⚖️ Compare multiple LLMs
  - Llama 3.2:1B
  - Gemma3:270M
  - Qwen2.5:0.5B
- ⚡ FastAPI REST API
- 📚 Row-Level Semantic Chunking
- 🧠 Retrieval-Augmented Generation (RAG)

---

# 🏗️ Architecture

```
                User Query
                    │
                    ▼
     BAAI/bge-base-en-v1.5 Embedding
                    │
                    ▼
         Weaviate Vector Database
                    │
                    ▼
      Top-K Similar Medical Q&A Pairs
                    │
                    ▼
            Context Builder
                    │
                    ▼
     Ollama (Qwen / Llama / Gemma)
                    │
                    ▼
        Grounded Medical Answer
```

---

# 📂 Project Structure

```
Medical-QA-RAG/
│
├── Data/
│   └── train.csv
│
├── ingestion.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Vector Database | Weaviate |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| LLM Runtime | Ollama |
| Models | Llama3.2:1B, Gemma3:270M, Qwen2.5:0.5B |
| Language | Python |
| Dataset | Comprehensive Medical Q&A Dataset |

---

# 📖 Dataset

The project uses the **Comprehensive Medical Q&A Dataset**.

Each record contains:

- Question
- Answer
- Question Type (`qtype`)

Example:

```text
Question:
What causes Asthma?

Answer:
The exact cause of asthma is unknown...

Type:
causes
```

---

# ✂️ Chunking Strategy

## Row-Level Semantic Chunking

Each Question-Answer pair is treated as a single semantic chunk.

Example:

```text
Type:
Symptoms

Question:
What are the symptoms of Parkinson's Disease?

Answer:
Early symptoms include tremor, rigidity...
```

**One Medical Record = One Semantic Chunk**

---

# 🧠 Embedding Model

**Model Used**

```
BAAI/bge-base-en-v1.5
```

### Purpose

- Converts text into dense vector embeddings
- Captures semantic meaning
- Enables semantic similarity search

---

# 🗄️ Vector Database

**Weaviate** stores:

- Question
- Answer
- Question Type
- Embedding Vector

The vector database retrieves the **Top-K most similar Question-Answer pairs** using semantic similarity.

---

# 🔄 RAG Workflow

### Step 1

User asks a question.

```
What causes hypertension?
```

↓

### Step 2

Generate embedding using

```
BAAI/bge-base-en-v1.5
```

↓

### Step 3

Search Weaviate

Retrieve Top-K similar Question-Answer pairs.

↓

### Step 4

Build Context

```
Reference 1

Question:
...

Answer:
...

Reference 2

Question:
...

Answer:
...
```

↓

### Step 5

Send Context + User Query to Ollama

↓

### Step 6

LLM generates a grounded answer.

---

# 🌐 API Endpoints

## Home

### GET /

Returns

- Project Information
- Available Models
- Total Records

---

## Search

### GET /search

Example

```
/search?query=What causes asthma?
```

Purpose

- Retrieval only
- Returns Top-K similar Question-Answer pairs

---

## Ask

### GET /ask

Example

```
/ask?query=What causes hypertension?
```

Workflow

```
User Query
      │
      ▼
Retrieve Top-K Documents
      │
      ▼
Build Context
      │
      ▼
Qwen2.5:0.5B
      │
      ▼
Generated Answer
```

Example Response

```json
{
  "query": "What causes hypertension?",
  "answer": "Hypertension can result from genetic and lifestyle factors..."
}
```

---

## Compare

### GET /compare

Example

```
/compare?query=What causes asthma?
```

Purpose

Compare responses from multiple LLMs using the same retrieved context.

Example Response

```json
{
  "query": "What causes asthma?",
  "retrieved_documents": 5,
  "models": {
    "llama3.2:1b": {
      "answer": "..."
    },
    "gemma3:270m": {
      "answer": "..."
    },
    "qwen2.5:0.5b": {
      "answer": "..."
    }
  }
}
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Medical-QA-RAG.git

cd Medical-QA-RAG
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start Weaviate

```bash
docker compose up -d
```

---

## Ingest Dataset

```bash
python ingestion.py
```

This will

- Read the CSV dataset
- Generate embeddings
- Store vectors in Weaviate

---

## Run FastAPI

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```
http://localhost:8000/docs
```

---

# 🤖 Models Used

| Model | Purpose |
|--------|----------|
| BAAI/bge-base-en-v1.5 | Embedding |
| Llama3.2:1B | Response Generation |
| Gemma3:270M | Response Generation |
| Qwen2.5:0.5B | Response Generation |

---

# 📈 Future Improvements

- Hybrid Search
- Similarity Score Display
- Cross-Encoder Re-ranking
- PDF RAG
- Streamlit Chat UI
- LangChain Integration
- LangGraph Agents
- Conversation Memory
- Source Citations

---

# 📚 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Weaviate Vector Database
- FastAPI
- Ollama Integration
- Prompt Engineering
- Model Comparison
- Row-Level Semantic Chunking

---

# Acknowledgements

- Sentence Transformers
- Weaviate
- Ollama
- FastAPI
- BAAI

---
