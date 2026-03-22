"""LightRAG backend using Claude CLI + local sentence-transformers.

No API keys needed. Uses:
- Claude CLI (Max subscription) for LLM
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
"""

import asyncio
import subprocess
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# Singleton embedding model (loaded once)
_embed_model: SentenceTransformer | None = None
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


async def claude_cli_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """LightRAG-compatible LLM function using claude CLI."""
    full_prompt = "IMPORTANT: Respond with text only. Do NOT use any tools. Do NOT read files or search the web. Just answer based on the provided context.\n\n"
    if system_prompt:
        full_prompt += system_prompt + "\n\n"
    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            full_prompt += f"[{role}]: {content}\n\n"
    full_prompt += prompt

    # Run claude CLI in thread to not block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _call_claude_cli, full_prompt)
    return result


def _call_claude_cli(prompt: str, timeout: int = 180) -> str:
    """Synchronous claude CLI call. Uses stdin for long prompts."""
    try:
        result = subprocess.run(
            [
                "claude",
                "-p", "-",
                "--output-format", "text",
                "--max-turns", "5",
                "--model", "sonnet",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = result.stdout.strip()
        if not text:
            # Debug: check stderr
            err = result.stderr.strip()
            if err:
                return f"(에러: {err[:200]})"
            return "(응답 없음)"
        return text
    except subprocess.TimeoutExpired:
        return "(타임아웃)"
    except Exception as e:
        return f"(에러: {e})"


async def local_embed(texts: list[str], **kwargs) -> np.ndarray:
    """LightRAG-compatible embedding function using sentence-transformers."""
    model = _get_embed_model()
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, lambda: model.encode(texts, normalize_embeddings=True)
    )
    return np.array(embeddings, dtype=np.float32)


# ─── Knowledge Store ─────────────────────────────────────────────

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_store"


class AlphaSquadKnowledge:
    """LightRAG-based knowledge store for alpha_squad agents.

    Each agent gets its own LightRAG instance with separate storage.
    """

    AGENTS = [
        "researcher1",
        "researcher2",
        "expert1",
        "expert2",
        "researcher3",
    ]

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir) if base_dir else KNOWLEDGE_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._instances: dict[str, LightRAG] = {}

    async def _get_rag(self, agent: str) -> LightRAG:
        """Get or create a LightRAG instance for an agent."""
        if agent not in self._instances:
            agent_dir = self._base_dir / agent
            agent_dir.mkdir(parents=True, exist_ok=True)

            rag = LightRAG(
                working_dir=str(agent_dir),
                llm_model_func=claude_cli_complete,
                llm_model_name="claude-sonnet",
                llm_model_max_async=1,  # CLI 동시 호출 제한
                embedding_func=EmbeddingFunc(
                    embedding_dim=EMBEDDING_DIM,
                    func=local_embed,
                    max_token_size=512,
                    model_name="all-MiniLM-L6-v2",
                ),
                embedding_batch_num=16,
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                max_parallel_insert=1,
            )
            await rag.initialize_storages()
            self._instances[agent] = rag

        return self._instances[agent]

    async def ingest(self, agent: str, text: str, source: str = "") -> None:
        """Ingest a document into an agent's knowledge base.

        Args:
            agent: Agent name (e.g., "researcher2")
            text: Document text content
            source: Source identifier (e.g., paper title, URL)
        """
        rag = await self._get_rag(agent)
        # Prepend source info if provided
        if source:
            text = f"[출처: {source}]\n\n{text}"
        await rag.ainsert(text)
        print(f"  [{agent}] ingested {len(text)} chars from: {source or 'unknown'}")

    async def ingest_file(self, agent: str, file_path: str | Path) -> None:
        """Ingest a text/markdown file into an agent's knowledge base."""
        path = Path(file_path)
        if not path.exists():
            print(f"  [{agent}] file not found: {path}")
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        await self.ingest(agent, text, source=path.name)

    async def query(
        self,
        agent: str,
        question: str,
        mode: str = "hybrid",
    ) -> str:
        """Query an agent's knowledge base.

        Args:
            agent: Agent name
            question: Query string
            mode: "local" (entity-focused), "global" (high-level), "hybrid" (both)

        Returns:
            Answer string from LightRAG
        """
        rag = await self._get_rag(agent)
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return result

    async def query_context(
        self,
        agent: str,
        question: str,
        mode: str = "hybrid",
        max_chars: int = 3000,
    ) -> str:
        """Query and return formatted context for injection into agent prompt.

        This is the main method used during meetings.
        """
        result = await self.query(agent, question, mode=mode)
        if not result or result.strip() == "":
            return ""
        # Truncate if needed
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return f"[Knowledge Base 검색 결과]\n{result}"

    async def stats(self) -> dict[str, dict]:
        """Get stats for all agent knowledge stores."""
        result = {}
        for agent in self.AGENTS:
            agent_dir = self._base_dir / agent
            if agent_dir.exists():
                # Count files in storage
                files = list(agent_dir.glob("**/*"))
                result[agent] = {
                    "files": len([f for f in files if f.is_file()]),
                    "exists": True,
                }
            else:
                result[agent] = {"files": 0, "exists": False}
        return result

    async def close(self) -> None:
        """Finalize all LightRAG instances."""
        for rag in self._instances.values():
            await rag.finalize_storages()
        self._instances.clear()
