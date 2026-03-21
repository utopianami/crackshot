"""Lightweight vector memory using ChromaDB + sentence-transformers.

No external API keys needed. Everything runs locally.
"""

import json
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


MEMORY_DIR = Path(__file__).parent.parent / "memory_store" / "chroma"


class AlphaSquadMemory:
    """Each agent has its own memory namespace via ChromaDB."""

    def __init__(self, persist_dir: str | None = None):
        persist_dir = persist_dir or str(MEMORY_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        self._client = chromadb.PersistentClient(path=persist_dir)

    # ChromaDB only allows [a-zA-Z0-9._-] in collection names
    _NAME_MAP = {
        "Researcher1_금융모델링": "researcher1-finance",
        "Researcher2_AI아키텍처": "researcher2-architecture",
        "Researcher3_PM": "researcher3-pm",
        "Expert1_피처전문가": "expert1-feature",
        "Expert2_문제정의전문가": "expert2-problem",
    }

    def _collection(self, agent_name: str):
        """Get or create a collection for an agent."""
        import re
        safe_name = self._NAME_MAP.get(agent_name, re.sub(r'[^a-zA-Z0-9._-]', '-', agent_name))
        if len(safe_name) < 3:
            safe_name = safe_name + "---"
        return self._client.get_or_create_collection(
            name=safe_name,
            embedding_function=self._ef,
        )

    def add(self, agent_name: str, content: str, metadata: dict | None = None) -> None:
        """Store a memory for an agent."""
        col = self._collection(agent_name)
        doc_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{col.count()}"
        meta = metadata or {}
        meta["timestamp"] = datetime.now().isoformat()
        # ChromaDB metadata values must be str, int, float, or bool
        meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}
        col.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id],
        )

    def search(self, agent_name: str, query: str, limit: int = 5) -> list[dict]:
        """Search memories for an agent."""
        col = self._collection(agent_name)
        if col.count() == 0:
            return []
        n = min(limit, col.count())
        results = col.query(query_texts=[query], n_results=n)
        out = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            out.append({"memory": doc, "metadata": meta})
        return out

    def get_all(self, agent_name: str) -> list[dict]:
        """Get all memories for an agent."""
        col = self._collection(agent_name)
        if col.count() == 0:
            return []
        results = col.get()
        out = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            out.append({"memory": doc, "metadata": meta})
        return out

    def get_context_string(self, agent_name: str, query: str, limit: int = 5) -> str:
        """Get formatted memory context for injection into agent prompt."""
        memories = self.search(agent_name, query, limit=limit)
        if not memories:
            return ""
        lines = ["[과거 기억]"]
        for m in memories:
            lines.append(f"- {m['memory'][:300]}")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Get memory stats per agent."""
        collections = self._client.list_collections()
        return {col.name: col.count() for col in collections}
