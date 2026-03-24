"""Semantic Scholar API로 학회 논문 수집 + LightRAG 지식 그래프 구축.

Usage:
    # 1단계: 논문 수집만 (API 호출, 빠름)
    PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py collect

    # 2단계: 수집된 논문으로 그래프 구축 (ollama + Qwen3, 오래 걸림)
    PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py build

    # 전체 (수집 + 그래프)
    PYTHONPATH="." uv run python alpha_squad/knowledge/collect_papers.py all
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import quote

import numpy as np

# ─── Config ──────────────────────────────────────────────────────

CONFERENCES = {
    "ICML": "International Conference on Machine Learning",
    "NeurIPS": "Neural Information Processing Systems",
    "ICLR": "International Conference on Learning Representations",
    "KDD": "Knowledge Discovery and Data Mining",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "IJCAI": "International Joint Conference on Artificial Intelligence",
    "WSDM": "Web Search and Data Mining",
    "WWW": "World Wide Web",
    "SIGIR": "Annual International ACM SIGIR Conference",
    "CIKM": "Conference on Information and Knowledge Management",
    "ECML": "European Conference on Machine Learning",
    "AISTATS": "Artificial Intelligence and Statistics",
    "ICAIF": "ACM International Conference on AI in Finance",
}

YEARS = [2022, 2023, 2024, 2025]

PAPERS_DIR = Path(__file__).parent.parent / "papers"
PAPERS_INDEX = PAPERS_DIR / "paper_index.json"
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_store"

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,venue,year,externalIds,openAccessPdf,citationCount"
S2_RATE_LIMIT_WAIT = 3.5  # seconds between requests (100 req / 5 min)


# ─── Step 1: Collect papers ─────────────────────────────────────

def _s2_request(url: str) -> dict:
    """Make a request to Semantic Scholar API with rate limiting."""
    req = Request(url, headers={"User-Agent": "AlphaSquad/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        if e.code == 429:
            print(f"  ⚠️ Rate limited. Waiting 60s...")
            time.sleep(60)
            return _s2_request(url)
        raise


def collect_conference_year(venue_name: str, year: int) -> list[dict]:
    """Collect all papers from a conference for a given year."""
    papers = []
    token = None
    page = 0

    while True:
        url = f"{S2_API}/paper/search/bulk?venue={quote(venue_name)}&year={year}&fields={S2_FIELDS}"
        if token:
            url += f"&token={token}"

        page += 1
        data = _s2_request(url)

        batch = data.get("data", [])
        papers.extend(batch)

        total = data.get("total", 0)
        token = data.get("token")

        print(f"    page {page}: +{len(batch)} papers (total so far: {len(papers)}/{total})")

        if not token or not batch:
            break

        time.sleep(S2_RATE_LIMIT_WAIT)

    return papers


def collect_all() -> dict:
    """Collect papers from all conferences and years."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    all_papers = {}
    total_count = 0

    for conf_short, conf_full in CONFERENCES.items():
        for year in YEARS:
            key = f"{conf_short}_{year}"
            print(f"\n📚 {conf_short} {year} ({conf_full})")

            papers = collect_conference_year(conf_full, year)

            # Filter out papers without abstract
            with_abstract = [p for p in papers if p.get("abstract")]
            without = len(papers) - len(with_abstract)
            if without:
                print(f"    ⚠️ {without} papers without abstract (skipped)")

            all_papers[key] = with_abstract
            total_count += len(with_abstract)
            print(f"    ✅ {len(with_abstract)} papers collected")

            # Save per-conference file
            conf_file = PAPERS_DIR / f"{key}.json"
            conf_file.write_text(json.dumps(with_abstract, ensure_ascii=False, indent=1))

    # Save combined index
    index = {
        "collected_at": datetime.now().isoformat(),
        "conferences": list(CONFERENCES.keys()),
        "years": YEARS,
        "counts": {k: len(v) for k, v in all_papers.items()},
        "total": total_count,
    }
    PAPERS_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"  수집 완료: 총 {total_count}편")
    for k, v in all_papers.items():
        print(f"    {k}: {len(v)}편")
    print(f"  저장: {PAPERS_DIR}/")
    print(f"{'='*60}")

    return all_papers


# ─── Step 2: Build knowledge graph ──────────────────────────────

async def build_graph():
    """Build LightRAG knowledge graph from collected abstracts using ollama."""
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    from alpha_squad.knowledge.lightrag_backend import local_embed, EMBEDDING_DIM

    if not PAPERS_INDEX.exists():
        print("❌ 논문이 수집되지 않았습니다. 먼저 'collect'를 실행하세요.")
        return

    index = json.loads(PAPERS_INDEX.read_text())
    print(f"논문 인덱스: {index['total']}편, {index['conferences']}")

    # Init LightRAG with ollama
    agent_dir = KNOWLEDGE_DIR / "researcher2"
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Wrapper to disable Qwen3 thinking mode
    async def qwen3_no_think(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ):
        # Prepend /no_think to disable thinking mode
        prompt = f"/no_think\n{prompt}"
        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    rag = LightRAG(
        working_dir=str(agent_dir),
        llm_model_func=qwen3_no_think,
        llm_model_name="qwen3:30b-a3b",
        llm_model_max_async=2,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            func=local_embed,
            max_token_size=512,
            model_name="all-MiniLM-L6-v2",
        ),
        embedding_batch_num=32,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        max_parallel_insert=2,
    )
    await rag.initialize_storages()

    # Process each conference file
    total_ingested = 0
    total_failed = 0

    for conf_year, count in index["counts"].items():
        conf_file = PAPERS_DIR / f"{conf_year}.json"
        if not conf_file.exists():
            print(f"⚠️ {conf_file} not found, skipping")
            continue

        papers = json.loads(conf_file.read_text())
        print(f"\n📖 {conf_year}: {len(papers)}편 ingest 시작")

        # Batch abstracts: 10 papers per document for efficiency
        batch_size = 10
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            # Combine abstracts into one document
            doc_parts = []
            for p in batch:
                title = p.get("title", "Unknown")
                abstract = p.get("abstract", "")
                year = p.get("year", "")
                venue = p.get("venue", "")
                arxiv_id = p.get("externalIds", {}).get("ArXiv", "")
                citations = p.get("citationCount", 0)

                doc_parts.append(
                    f"## {title}\n"
                    f"- Venue: {venue} {year}\n"
                    f"- ArXiv: {arxiv_id}\n"
                    f"- Citations: {citations}\n"
                    f"\n{abstract}\n"
                )

            doc = "\n---\n\n".join(doc_parts)

            try:
                await rag.ainsert(doc)
                total_ingested += len(batch)
            except Exception as e:
                total_failed += len(batch)
                print(f"  ❌ batch {i//batch_size + 1} failed: {e}")

            if (i // batch_size + 1) % 10 == 0:
                print(f"  진행: {min(i + batch_size, len(papers))}/{len(papers)} "
                      f"(ingested: {total_ingested}, failed: {total_failed})")

        print(f"  ✅ {conf_year} 완료")

    await rag.finalize_storages()

    print(f"\n{'='*60}")
    print(f"  그래프 구축 완료")
    print(f"  총 {total_ingested}편 ingest, {total_failed}편 실패")
    print(f"  저장: {agent_dir}/")
    print(f"{'='*60}")


# ─── Step 3: Embed abstracts ────────────────────────────────────

EMBEDDINGS_FILE = PAPERS_DIR / "abstract_embeddings.npz"


def embed_all():
    """Embed all collected abstracts using sentence-transformers (local GPU)."""
    from sentence_transformers import SentenceTransformer

    if not PAPERS_INDEX.exists():
        print("❌ 논문이 수집되지 않았습니다. 먼저 'collect'를 실행하세요.")
        return

    index = json.loads(PAPERS_INDEX.read_text())
    print(f"논문 인덱스: {index['total']}편")

    # Load all abstracts
    all_texts = []
    all_meta = []  # (conf_year, paper_idx, title, paperId)

    for conf_year, count in index["counts"].items():
        conf_file = PAPERS_DIR / f"{conf_year}.json"
        if not conf_file.exists():
            continue
        papers = json.loads(conf_file.read_text())
        for i, p in enumerate(papers):
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            if not abstract:
                continue
            # Embed: title + abstract
            all_texts.append(f"{title}\n{abstract}")
            all_meta.append({
                "conf": conf_year,
                "idx": i,
                "title": title,
                "abstract": abstract,
                "paperId": p.get("paperId", ""),
                "year": p.get("year", 0),
                "venue": p.get("venue", ""),
                "citations": p.get("citationCount", 0),
                "arxiv": p.get("externalIds", {}).get("ArXiv", ""),
            })

    print(f"임베딩할 논문: {len(all_texts)}편")

    # Embed in batches
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("임베딩 생성 중...")
    embeddings = model.encode(
        all_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Save
    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings,
    )
    # Save metadata separately (JSON)
    meta_file = PAPERS_DIR / "abstract_meta.json"
    meta_file.write_text(json.dumps(all_meta, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  임베딩 완료: {len(all_texts)}편, shape={embeddings.shape}")
    print(f"  저장: {EMBEDDINGS_FILE}")
    print(f"  메타: {meta_file}")
    print(f"{'='*60}")


def search_papers(query: str, top_k: int = 20) -> list[dict]:
    """Search papers by query using embedding similarity."""
    from sentence_transformers import SentenceTransformer

    if not EMBEDDINGS_FILE.exists():
        print("❌ 임베딩이 없습니다. 먼저 'embed'를 실행하세요.")
        return []

    # Load
    data = np.load(EMBEDDINGS_FILE)
    embeddings = data["embeddings"]
    meta = json.loads((PAPERS_DIR / "abstract_meta.json").read_text())

    # Embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True)

    # Cosine similarity
    sims = embeddings @ q_emb.T
    sims = sims.squeeze()

    # Top-K
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        m = meta[idx]
        m["score"] = float(sims[idx])
        results.append(m)

    return results


def cmd_search(query: str, top_k: int = 20):
    """CLI search command."""
    results = search_papers(query, top_k=top_k)
    if not results:
        return

    print(f"\n=== '{query}' 검색 결과 (top {top_k}) ===\n")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. [{r['score']:.3f}] {r['title']}")
        print(f"    {r['venue']} {r['year']} | citations: {r['citations']} | arXiv: {r['arxiv']}")
        print()


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="논문 수집 + 임베딩 + 지식 그래프 구축")
    parser.add_argument("command", choices=["collect", "embed", "search", "build", "all"],
                        help="collect: 논문 수집, embed: 임베딩 생성, search: 논문 검색, build: 그래프 구축, all: collect+embed")

    parser.add_argument("query", nargs="?", default=None, help="search 시 검색어")
    parser.add_argument("--top-k", type=int, default=20, help="search 시 결과 수")

    args = parser.parse_args()

    if args.command == "collect":
        collect_all()
    elif args.command == "embed":
        embed_all()
    elif args.command == "search":
        if not args.query:
            print("검색어를 입력하세요: collect_papers.py search \"query\"")
            return
        cmd_search(args.query, top_k=args.top_k)
    elif args.command == "build":
        asyncio.run(build_graph())
    elif args.command == "all":
        collect_all()
        embed_all()


if __name__ == "__main__":
    main()
