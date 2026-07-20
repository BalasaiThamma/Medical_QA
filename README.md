# Medical Q&A RAG System using Weaviate, BGE Embeddings & Ollama

A complete Retrieval-Augmented Generation (RAG) application built using FastAPI, Weaviate, Sentence Transformers, and Ollama. This project retrieves semantically relevant medical Question-Answer pairs from a vector database and generates grounded answers using locally hosted Large Language Models.

🚀 Features
Semantic Search using BAAI/bge-base-en-v1.5
Vector Database with Weaviate
Local LLM Inference using Ollama
Compare multiple LLMs
Llama 3.2 1B
Gemma 3 270M
Qwen 2.5 0.5B
FastAPI REST API
Row-Level Semantic Chunking
Retrieval + Generation (Complete RAG Pipeline)
Production-ready project structure
📌 Project Architecture
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
📂 Project Structure
Medical_QA_RAG/
│
├── Data/
│   └── train.csv
│
├── ingestion.py
├── main.py
├── requirements.txt
├── README.md
│
└── venv/
🛠 Tech Stack
Component	Technology
Backend	FastAPI
Vector Database	Weaviate
Embedding Model	BAAI/bge-base-en-v1.5
LLM Runtime	Ollama
LLM Models	Llama 3.2 1B, Gemma 3 270M, Qwen 2.5 0.5B
Language	Python
Dataset	Comprehensive Medical Q&A Dataset
📖 Dataset

This project uses the Comprehensive Medical Q&A Dataset containing structured medical Question-Answer pairs.

Each record contains:

Question
Answer
Question Type (qtype)

Example:

Question:
What causes Asthma?

Answer:
The exact cause of asthma is unknown...

Type:
causes

Each Question-Answer pair is treated as one semantic knowledge unit.

✂️ Chunking Strategy
Row-Level Semantic Chunking

Instead of splitting documents into paragraphs or sentences, each Question-Answer pair is treated as an independent semantic chunk.

Example:

Type:
Symptoms

Question:
What are the symptoms of Parkinson's Disease?

Answer:
Early symptoms include tremor, rigidity...

One Medical Record = One Chunk

This approach preserves the complete medical context for retrieval.

🧠 Embedding Model

Model Used

BAAI/bge-base-en-v1.5

Purpose

Converts text into dense vector embeddings
Captures semantic meaning instead of keyword matching
Improves retrieval accuracy

Example

"What causes hypertension?"

↓

768-dimensional vector
🗄 Vector Database

We use Weaviate to store vector embeddings.

Each stored object contains:

Question
Answer
Question Type
Embedding Vector

Weaviate performs fast semantic similarity search using vector distance.

🔄 RAG Workflow
Step 1

User submits a medical question.

Example:

What causes hypertension?
Step 2

The query is converted into an embedding using

BAAI/bge-base-en-v1.5
Step 3

Weaviate retrieves the Top-K Similar Medical Q&A Pairs.

Example:

1. What causes High Blood Pressure?
2. What is Hypertension?
3. What causes Essential Hypertension?
Step 4

Retrieved documents are combined into a context prompt.

Example:

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
Step 5

The context and user query are passed to a local LLM via Ollama.

Supported Models

Llama 3.2 1B
Gemma 3 270M
Qwen 2.5 0.5B
Step 6

The LLM generates a grounded answer based only on the retrieved medical knowledge.

🌐 API Endpoints
Home
GET /

Returns

Project Information
Embedding Model
Record Count
Available Models
Search
GET /search?query=What causes asthma?

Purpose

Retrieval Only

Returns the most semantically similar medical Question-Answer pairs.

Ask
GET /ask?query=What causes hypertension?

Purpose

Complete RAG

Flow

User Query
↓

Retrieve Similar Documents

↓

Build Context

↓

Qwen 2.5

↓

Generated Answer

Example Response

{
    "query":"What causes hypertension?",
    "answer":"Hypertension can result from genetic, environmental and lifestyle factors..."
}
Compare
GET /compare?query=What causes asthma?

Purpose

Compare multiple LLMs using the same retrieved context.

Models Compared

Llama 3.2 1B
Gemma 3 270M
Qwen 2.5 0.5B

Example Response

{
    "query":"What causes asthma?",
    "models":{
        "llama3.2:1b":{
            "answer":"..."
        },
        "gemma3:270m":{
            "answer":"..."
        },
        "qwen2.5:0.5b":{
            "answer":"..."
        }
    }
}
⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/Medical-QA-RAG.git

cd Medical-QA-RAG

Create virtual environment

python -m venv venv

Activate

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

Install dependencies

pip install -r requirements.txt
🗄 Start Weaviate
docker compose up -d
📥 Ingest Dataset
python ingestion.py

This will

Read Medical Dataset
Generate Embeddings
Store vectors in Weaviate
▶️ Run FastAPI
uvicorn main:app --reload

Swagger UI

http://localhost:8000/docs
📊 Models Used
Model	Purpose
BAAI/bge-base-en-v1.5	Embedding Generation
Llama 3.2 1B	Response Generation
Gemma 3 270M	Response Generation
Qwen 2.5 0.5B	Response Generation
📈 Future Improvements
Hybrid Search
Similarity Score Display
Cross-Encoder Re-ranking
PDF Document RAG
Multi-document Knowledge Base
Streamlit Chat Interface
LangChain Integration
LangGraph Agents
Conversation Memory
Source Citation in Responses
🎯 Learning Outcomes

This project demonstrates:

Retrieval-Augmented Generation (RAG)
Semantic Search
Vector Embeddings
Weaviate Vector Database
FastAPI API Development
Ollama Local LLM Integration
Prompt Engineering
Model Comparison
Row-Level Semantic Chunking
🤝 Acknowledgements
Sentence Transformers for semantic embeddings
Weaviate for vector search
Ollama for local LLM inference
FastAPI for API development
BAAI for the BGE embedding model
