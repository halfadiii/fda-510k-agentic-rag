"""
Day 2, Part 2: Search.

THREE SEARCH MODES, so you can compare them on Day 5:

  dense   -- meaning-based. Finds "gamma irradiation" when you ask about
             sterilization. Bad at exact identifiers.
  bm25    -- keyword-based. Nails "ISO 10993-1" and "K212521".
             Blind to synonyms.
  hybrid  -- both, combined with Reciprocal Rank Fusion. Usually best.

HOW RRF WORKS (it's simpler than it sounds)
    Each search returns a ranked list. For every chunk, score it as
        1 / (60 + its rank in that list)
    and add up the scores across both lists. Rank 1 scores 1/61, rank 2 scores
    1/62, and so on. Anything appearing high in BOTH lists wins.

    The clever part: it uses only RANK, never the raw similarity scores. That
    matters because cosine similarity (0 to 1) and BM25 scores (unbounded) live
    on totally different scales and can't be added directly. Ranks are
    comparable by construction. The 60 is a standard damping constant -- it
    stops the top result from dominating everything.

Use:  python search.py "your question here"
Or:   from search import Retriever;  Retriever().search("question", k=5)
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QDRANT_PATH = Path("data/qdrant")
BM25_FILE = Path("data/bm25.pkl")
CHUNKS_FILE = Path("data/chunks.json")

COLLECTION = "fda_wound_dressings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60


@dataclass
class Result:
    chunk_id: str
    text: str
    doc_id: str
    doc_type: str
    doc_title: str
    section: str
    page_start: int
    page_end: int
    score: float
    rank_dense: int | None = None
    rank_bm25: int | None = None

    def citation(self) -> str:
        """Short, human-checkable source label."""
        return f"{self.doc_id} p.{self.page_start} ({self.section})"


def tokenize(text: str) -> list[str]:
    """Must match the tokenizer used in embed.py."""
    return re.findall(r"[a-z0-9][a-z0-9\.\-\(\)]*", text.lower())


class Retriever:
    def __init__(self) -> None:
        import json

        self.model = SentenceTransformer(MODEL_NAME)
        self.client = QdrantClient(path=str(QDRANT_PATH))

        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.by_id = {c["chunk_id"]: c for c in chunks}

        with BM25_FILE.open("rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        self.bm25_ids = saved["chunk_ids"]

    # -- individual searches ------------------------------------------------

    def _dense(self, query: str, limit: int,
               doc_type: str | None = None) -> list[str]:
        """Return chunk_ids, best first."""
        qv = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)

        flt = None
        if doc_type:
            flt = Filter(must=[FieldCondition(key="doc_type",
                                              match=MatchValue(value=doc_type))])

        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=qv.tolist(),
            limit=limit,
            query_filter=flt,
        ).points
        return [h.payload["chunk_id"] for h in hits]

    def _bm25(self, query: str, limit: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]
        return [self.bm25_ids[i] for i in ranked if scores[i] > 0]

    # -- fusion -------------------------------------------------------------

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               doc_type: str | None = None, pool: int = 30) -> list[Result]:
        """
        mode: "dense" | "bm25" | "hybrid"
        pool: how many candidates each method contributes before fusing.
              Bigger pool = better recall, slightly slower.
        """
        dense_ids = self._dense(query, pool, doc_type) if mode in ("dense", "hybrid") else []
        bm25_ids = self._bm25(query, pool) if mode in ("bm25", "hybrid") else []

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}

        for rank, cid in enumerate(dense_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            dense_rank[cid] = rank

        for rank, cid in enumerate(bm25_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            bm25_rank[cid] = rank

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]

        results: list[Result] = []
        for cid, score in top:
            c = self.by_id.get(cid)
            if not c:
                continue
            results.append(Result(
                chunk_id=cid,
                text=c["text"],
                doc_id=c["doc_id"],
                doc_type=c["doc_type"],
                doc_title=c["doc_title"],
                section=c["section"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                score=score,
                rank_dense=dense_rank.get(cid),
                rank_bm25=bm25_rank.get(cid),
            ))
        return results

    def close(self) -> None:
        self.client.close()


# ---------------------------------------------------------------------------

def demo(query: str) -> None:
    r = Retriever()

    for mode in ("dense", "bm25", "hybrid"):
        print("\n" + "=" * 70)
        print(f"{mode.upper()}  --  {query}")
        print("=" * 70)

        for i, res in enumerate(r.search(query, k=4, mode=mode), 1):
            where = []
            if res.rank_dense:
                where.append(f"dense#{res.rank_dense}")
            if res.rank_bm25:
                where.append(f"bm25#{res.rank_bm25}")
            tag = ", ".join(where) if where else "-"

            print(f"\n{i}. [{res.score:.4f}]  {res.citation()}   ({tag})")
            print(f"   {res.doc_type} | {res.doc_title[:55]}")
            print(f"   {res.text[:200]}...")

    r.close()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or \
        "What biocompatibility testing is required for a dressing that contacts an open wound?"
    demo(q)
