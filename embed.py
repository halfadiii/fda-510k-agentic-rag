"""
Day 2, Part 1: Turn chunks into searchable vectors.

WHAT AN EMBEDDING IS
    A model reads a piece of text and outputs a list of numbers (here, 384 of
    them) that represents its meaning. Texts about similar things get similar
    number-lists, even when they share no words.

        "how is the dressing sterilized?"
        "the device is terminally sterilized by gamma irradiation"

    Almost no words in common, but the meaning is close -- so their vectors
    land near each other. That's what makes meaning-based search possible.

WHY LOCAL
    DeepSeek has no embeddings API, so we run BAAI/bge-small-en-v1.5 on your
    own machine via sentence-transformers. It's free, private, and fast enough.
    Cost for the whole corpus: $0.

WHERE THEY GO
    Qdrant, a vector database -- storage built for "find me the nearest
    vectors" instead of "find me the exact match". We use its embedded mode,
    which is just a folder on disk. No Docker needed.

Run:  python embed.py
Out:  data/qdrant/  (the database)  +  data/bm25.pkl  (keyword index)
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
import pickle
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNKS_FILE = Path("data/chunks.json")
QDRANT_PATH = Path("data/qdrant")
BM25_FILE = Path("data/bm25.pkl")

COLLECTION = "fda_wound_dressings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384          # bge-small outputs 384 numbers. Must match Qdrant.
BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found -- run ingest.py first")
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


# ---------------------------------------------------------------------------
# Dense vectors
# ---------------------------------------------------------------------------

def build_vector_index(chunks: list[dict]) -> None:
    print(f"\nLoading embedding model ({MODEL_NAME})...")
    print("  First run downloads ~130MB. After that it's cached.")
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=str(QDRANT_PATH))

    # Recreate the collection from scratch so reruns are clean
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Indexes on these fields let us filter searches later, e.g. "only
    # guidance documents" or "only this device". Without an index, filtering
    # works but is slow.
    for field in ("doc_type", "doc_id", "section"):
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print("  On a laptop CPU this takes a few minutes.\n")

    start = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # NOTE: document text gets NO prefix. Questions do (see search.py).
        # That asymmetry is how bge models were trained.
        texts = [c["text"] for c in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,   # required for cosine similarity
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=i + j,                       # Qdrant needs an int or UUID
                vector=vec.tolist(),
                payload=chunk,                  # everything from ingest.py
            )
            for j, (chunk, vec) in enumerate(zip(batch, vectors))
        ]
        client.upsert(collection_name=COLLECTION, points=points)

        done = min(i + BATCH_SIZE, len(chunks))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0
        eta = (len(chunks) - done) / rate if rate else 0
        print(f"  {done:>5}/{len(chunks)}  ({100*done/len(chunks):5.1f}%)  "
              f"eta {eta:4.0f}s", end="\r")

    print(f"\n\nVector index built in {time.time() - start:.0f}s")
    info = client.get_collection(COLLECTION)
    print(f"  Collection '{COLLECTION}': {info.points_count} points")
    client.close()


# ---------------------------------------------------------------------------
# Keyword index
# ---------------------------------------------------------------------------

def build_keyword_index(chunks: list[dict]) -> None:
    """
    BM25 is classic keyword search -- it scores documents by how many of your
    exact search words they contain, weighted so rare words count for more.

    Why bother, when we have semantic search? Because meaning-based search is
    bad at exact strings. Ask for "K212521" or "ISO 10993-1" and the embedding
    model has no idea those are precise identifiers; it'll return things that
    are vaguely about standards. BM25 nails them.

    The two methods fail in different ways, which is exactly why combining
    them works.
    """
    from rank_bm25 import BM25Okapi

    print("\nBuilding keyword (BM25) index...")

    def tokenize(text: str) -> list[str]:
        import re
        # Keep alphanumerics and internal punctuation that matters here:
        # "10993-1" and "807.92(a)" should survive as units
        return re.findall(r"[a-z0-9][a-z0-9\.\-\(\)]*", text.lower())

    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    with BM25_FILE.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}, f)

    print(f"  Saved to {BM25_FILE}")


# ---------------------------------------------------------------------------

def smoke_test(chunks: list[dict]) -> None:
    """Prove it works before moving on."""
    print("\n" + "=" * 60)
    print("SMOKE TEST -- do these results look sensible?")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))

    queries = [
        "How are wound dressings sterilized?",
        "What biocompatibility testing is required for devices contacting broken skin?",
        "What is a predicate device?",
    ]

    for q in queries:
        # Questions DO get the prefix -- this is how bge was trained
        qv = model.encode(
            "Represent this sentence for searching relevant passages: " + q,
            normalize_embeddings=True,
        )
        hits = client.query_points(
            collection_name=COLLECTION,
            query=qv.tolist(),
            limit=3,
        ).points

        print(f"\nQ: {q}")
        for h in hits:
            p = h.payload
            print(f"  [{h.score:.3f}] {p['doc_type']:<9} {p['doc_id'][:28]:<28} "
                  f"p.{p['page_start']}  {p['section'][:35]}")
            print(f"           {p['text'][:110]}...")

    client.close()
    print("\n\nIf the retrieved text is on-topic, embedding worked.")
    print("If it's random, check that VECTOR_SIZE matches the model (384).")


def main() -> None:
    chunks = load_chunks()
    build_vector_index(chunks)
    build_keyword_index(chunks)
    smoke_test(chunks)


if __name__ == "__main__":
    main()
