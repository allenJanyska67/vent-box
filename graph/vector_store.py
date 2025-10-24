from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import chromadb
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore
    PersistentClient = None  # type: ignore
    embedding_functions = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRIEVANCE_DIR = PROJECT_ROOT / "grievances"
VECTOR_DB_DIR = GRIEVANCE_DIR / "vector_store"
COLLECTION_NAME = "grievances"

 # cosine similarity ranges from -1 to 1 where 1 is the most similar and -1 is complete opposite
 # according the the chroma documentation the equation they use is 1 - cosine similarity 
 # which means the closer the cosine similarity is to 1, the more similar the two vectors are
 # So if the distance function returns 0 then the cosine similarity is 1 and the two vectors are the most similar
 # So the threshold needs to be low.
DEFAULT_SIMILARITY_THRESHOLD = 0.5

_collection = None


def _resolve_collection():
    global _collection
    if _collection is not None:
        return _collection

    if chromadb is None or PersistentClient is None or embedding_functions is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    try:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    try:
        embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
        )
        client = PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        _collection = None
    return _collection


def find_similar_topic(
    message: str,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_results: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Return the metadata for the closest grievance whose cosine similarity meets the threshold.
    """
    text = (message or "").strip()
    if not text:
        return None

    collection = _resolve_collection()
    if collection is None:
        return None

    try:
        results = collection.query(
            query_texts=[text],
            n_results=max_results,
            include=["distances", "metadatas"],
        )
    except Exception:
        return None

    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    if not distances or not metadatas:
        return None

    best_index = 0
    best_distance = distances[best_index]
    best_metadata = metadatas[best_index]

    # Chroma reports cosine distance, so convert to similarity.
    topic_key = (best_metadata or {}).get("topic_key")

    similarity = 1.0 - float(best_distance)

    if topic_key and best_distance <= threshold:
        return {
            "topic_key": str(topic_key),
            "similarity": similarity,
            "metadata": best_metadata,
        }
    return None


def upsert_grievance_embedding(
    *,
    document_id: str,
    message: str,
    topic_key: str,
    summary: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Store or update the embedding for a grievance entry.
    """
    text = (message or "").strip()
    if not text or not document_id:
        return False

    collection = _resolve_collection()
    if collection is None:
        return False

    metadata: Dict[str, Any] = {
        "topic_key": topic_key,
        "summary": summary,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    try:
        collection.upsert(
            ids=[document_id],
            documents=[text],
            metadatas=[metadata],
        )
        return True
    except Exception:
        return False
