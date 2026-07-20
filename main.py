import ollama
import weaviate

from fastapi import FastAPI
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

COLLECTION_NAME = "MedicalQA"

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50051

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

MODELS = {
    "llama": "llama3.2:1b",
    "gemma": "gemma3:270m",
    "qwen": "qwen2.5:0.5b"
}

# =====================================================
# EMBEDDINGS
# =====================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

# =====================================================
# WEAVIATE
# =====================================================

client = weaviate.connect_to_local(
    host=WEAVIATE_HOST,
    port=WEAVIATE_HTTP_PORT,
    grpc_port=WEAVIATE_GRPC_PORT
)

collection = client.collections.get(
    COLLECTION_NAME
)

# =====================================================
# LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Medical RAG Started")

    yield

    client.close()

app = FastAPI(
    title="Medical Q&A RAG",
    lifespan=lifespan
)

# =====================================================
# RETRIEVE
# =====================================================

def retrieve(query, k=5):

    emb = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    response = collection.query.near_vector(
        near_vector=emb.tolist(),
        limit=k
    )

    return [
        obj.properties
        for obj in response.objects
    ]

# =====================================================
# BUILD CONTEXT
# =====================================================

def build_context(results):

    context = ""

    for i, r in enumerate(results, start=1):

        context += f"""
Reference {i}

Question:
{r.get('question')}

Answer:
{r.get('answer')}

"""

    return context

# =====================================================
# GENERATE
# =====================================================

def ask_model(model_name, query, context):

    prompt = f"""
You are a medical knowledge assistant.

Answer the user's question ONLY using the information provided
in the medical references below.

Medical References:
{context}

User Question:
{query}

Instructions:
- Give a concise and accurate medical answer.
- Do not mention the references.
- Give only the answer, do not include any additional information.
- Give the answer in a single paragraph.
- If the answer is not found in the references,
  say "The information is unavailable in the knowledge base."
"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2,
            "num_predict": 300
        }
    )

    return response["message"]["content"]


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    count = collection.aggregate.over_all(
        total_count=True
    ).total_count

    return {
        "project": "Medical Q&A RAG",
        "embedding_model": EMBEDDING_MODEL,
        "records": count,
        "models": MODELS
    }

# =====================================================
# SEARCH
# =====================================================

@app.get("/search")
def search(query: str):

    results = retrieve(
        query=query,
        k=5
    )

    return {
        "query": query,
        "results": results
    }

# =====================================================
# ASK (PRODUCTION RAG)
# =====================================================

@app.get("/ask")
def ask(query: str):

    docs = retrieve(
        query=query,
        k=5
    )

    context = build_context(
        docs
    )

    answer = ask_model(
        MODELS["qwen"],
        query,
        context
    )

    return {
        "query": query,
        "answer": answer
    }

# =====================================================
# COMPARE MODELS
# =====================================================

@app.get("/compare")
def compare(query: str):

    docs = retrieve(
        query=query,
        k=5
    )

    context = build_context(
        docs
    )

    llama = ask_model(
        MODELS["llama"],
        query,
        context
    )

    gemma = ask_model(
        MODELS["gemma"],
        query,
        context
    )

    qwen = ask_model(
        MODELS["qwen"],
        query,
        context
    )

    sources = []

    for i, doc in enumerate(
        docs,
        start=1
    ):
        sources.append({
            "rank": i,
            "question": doc.get("question"),
            "qtype": doc.get("qtype")
        })

    return {
        "query": query,

        "retrieved_documents": len(docs),

        "sources": sources,

        "models": {

            "llama3.2:1b": {
                "answer": llama
            },

            "gemma3:270m": {
                "answer": gemma
            },

            "qwen2.5:0.5b": {
                "answer": qwen
            }
        }
    }