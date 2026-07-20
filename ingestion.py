import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "Data/train.csv"

COLLECTION_NAME = "MedicalQA"

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50051

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

CSV_CHUNK_SIZE = 1000
EMBED_BATCH_SIZE = 64

MAX_ROWS = None

# ============================================================
# CHUNK PREVIEW
# ============================================================

def show_chunk_preview(chunk_text, chunk_no):
    print("\n" + "=" * 80)
    print(f"SEMANTIC CHUNK {chunk_no}")
    print("=" * 80)
    print(chunk_text[:1000])
    print("=" * 80)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(f"\nLoading Embedding Model: {EMBEDDING_MODEL}")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded successfully.\n")

# ============================================================
# CONNECT TO WEAVIATE
# ============================================================

client = weaviate.connect_to_local(
    host=WEAVIATE_HOST,
    port=WEAVIATE_HTTP_PORT,
    grpc_port=WEAVIATE_GRPC_PORT
)

try:

    # ============================================================
    # CREATE COLLECTION
    # ============================================================

    if not client.collections.exists(COLLECTION_NAME):

        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(
                    name="qtype",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="question",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="answer",
                    data_type=DataType.TEXT
                )
            ]
        )

        print(f"Created collection: {COLLECTION_NAME}")

    else:
        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

    collection = client.collections.get(
        COLLECTION_NAME
    )

    # ============================================================
    # CHECK CSV COLUMNS
    # ============================================================

    sample = pd.read_csv(
        CSV_PATH,
        nrows=5
    )

    print("\nDetected Columns:")
    print(sample.columns.tolist())

    # ============================================================
    # READ CSV
    # ============================================================

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CSV_CHUNK_SIZE
    )

    total_inserted = 0
    total_failed = 0
    preview_count = 0

    # ============================================================
    # PROCESS DATA
    # ============================================================

    for chunk_number, df_chunk in enumerate(
        reader,
        start=1
    ):

        if MAX_ROWS is not None:

            if total_inserted >= MAX_ROWS:
                break

            remaining = MAX_ROWS - total_inserted

            if remaining < len(df_chunk):
                df_chunk = df_chunk.head(
                    remaining
                )

        texts_to_embed = []
        rows_data = []

        for _, row in df_chunk.iterrows():

            qtype = str(
                row.get("qtype", "")
            ).strip()

            question = str(
                row.get("Question", "")
            ).strip()

            answer = str(
                row.get("Answer", "")
            ).strip()

            if question == "nan":
                question = ""

            if answer == "nan":
                answer = ""

            chunk_text = f"""
Type: {qtype}

Question:
{question}

Answer:
{answer}
"""

            # Show first 3 chunks

            if preview_count < 3:

                show_chunk_preview(
                    chunk_text,
                    preview_count + 1
                )

                preview_count += 1

            texts_to_embed.append(
                chunk_text
            )

            rows_data.append({
                "qtype": qtype,
                "question": question,
                "answer": answer
            })

        # ========================================================
        # EMBEDDINGS
        # ========================================================

        embeddings = embedding_model.encode(
            texts_to_embed,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # ========================================================
        # INSERT INTO WEAVIATE
        # ========================================================

        with collection.batch.fixed_size(
            batch_size=200
        ) as batch:

            for row_data, embedding in zip(
                rows_data,
                embeddings
            ):

                batch.add_object(
                    properties=row_data,
                    vector=embedding.tolist()
                )

        failed_count = len(
            collection.batch.failed_objects
        )

        total_failed += failed_count
        total_inserted += len(
            rows_data
        )

        print(
            f"[Chunk {chunk_number}] "
            f"Inserted={len(rows_data)} "
            f"Total={total_inserted} "
            f"Failed={total_failed}"
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Records Inserted : {total_inserted}")
    print(f"Failed Records   : {total_failed}")
    print(f"Embedding Model  : {EMBEDDING_MODEL}")
    print("Chunking Method  : Row-Level Semantic Chunking")
    print("=" * 60)

finally:

    print("\nClosing Weaviate Connection...")
    client.close()