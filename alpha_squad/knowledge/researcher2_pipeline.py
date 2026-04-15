"""Researcher2 리서치 파이프라인: 2소스 병렬 검색 + Opus 순위 + 논문 요약.

Usage:
    # 리서치 실행
    PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py research "non-stationary tabular data 모델"

    # 이전 리서치 목록
    PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py list

    # 이전 컨텍스트 보기
    PYTHONPATH="." uv run python alpha_squad/knowledge/researcher2_pipeline.py show 20260324_non_stationary
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

RESEARCH_STORE = Path(__file__).parent.parent / "research_store"
PAPERS_DIR = RESEARCH_STORE / "papers"
QUERIES_DIR = RESEARCH_STORE / "queries"
CONTEXTS_DIR = RESEARCH_STORE / "contexts"
INDEX_FILE = CONTEXTS_DIR / "index.json"

# Embedding index (from collect_papers.py)
EMBED_DIR = Path(__file__).parent.parent / "papers"
EMBEDDINGS_FILE = EMBED_DIR / "abstract_embeddings.npz"
META_FILE = EMBED_DIR / "abstract_meta.json"


def _ensure_dirs():
    for d in [PAPERS_DIR, QUERIES_DIR, CONTEXTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:12]


def _load_index() -> list[dict]:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return []


def _save_index(index: list[dict]):
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2))


def _load_previous_papers(query: str) -> list[str]:
    """이전에 이 주제로 찾았던 논문 제목 목록."""
    index = _load_index()
    previous = []
    for entry in index:
        # 키워드 겹침 확인 (간단한 방식)
        entry_kw = set(entry.get("keywords", []))
        query_words = set(query.lower().split())
        if len(entry_kw & query_words) >= 2 or entry.get("topic", "").lower() in query.lower():
            qfile = QUERIES_DIR / f"{entry['id']}.json"
            if qfile.exists():
                qdata = json.loads(qfile.read_text())
                for pid in qdata.get("ranked", []):
                    pfile = PAPERS_DIR / f"{pid}.json"
                    if pfile.exists():
                        pdata = json.loads(pfile.read_text())
                        previous.append(pdata.get("title", ""))
    return list(set(previous))


# ─── Source 1: HyDE + Local Embedding ───────────────────────────

def _generate_hyde(query: str) -> str:
    """HyDE: 가상 논문 abstract 생성."""
    prompt = f"""You are a machine learning researcher. Write a hypothetical paper abstract that would answer this research question:

"{query}"

Write a realistic academic abstract (200-300 words) in English. Include specific method names, datasets, and results. Do not use tools."""

    try:
        result = subprocess.run(
            ["claude", "-p", "-", "--output-format", "text", "--max-turns", "1", "--model", "sonnet"],
            input=prompt, capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() or query
    except Exception:
        return query


def _search_local_embeddings(hyde_abstract: str, top_k: int = 30) -> list[dict]:
    """로컬 임베딩 인덱스에서 검색."""
    if not EMBEDDINGS_FILE.exists():
        print("  ⚠️ 임베딩 파일 없음. collect_papers.py embed 실행 필요.")
        return []

    from sentence_transformers import SentenceTransformer

    data = np.load(EMBEDDINGS_FILE)
    embeddings = data["embeddings"]
    meta = json.loads(META_FILE.read_text())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([hyde_abstract], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).squeeze()

    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        m = meta[idx].copy()
        m["score"] = float(sims[idx])
        m["source"] = "local_emb"
        results.append(m)

    return results


# ─── Source 2: Opus 직접 추천 ───────────────────────────────────

def _search_opus(query: str, previous_papers: list[str]) -> list[dict]:
    """Opus에게 직접 논문 추천 요청."""
    prev_text = ""
    if previous_papers:
        prev_list = "\n".join(f"- {t}" for t in previous_papers[:20])
        prev_text = f"\n\n이전에 이미 찾은 논문들 (이것 외에 추천해주세요):\n{prev_list}"

    prompt = f"""You are an expert ML researcher. Recommend papers for this research question:

"{query}"
{prev_text}

For each paper, provide EXACTLY this JSON format (array of objects):
[
  {{
    "title": "Paper Title",
    "authors": "Author1, Author2",
    "year": 2024,
    "arxiv": "2401.xxxxx",
    "venue": "ICML 2024",
    "key_idea": "2-sentence description of the key idea",
    "relevance": "1-sentence why this is relevant to the question"
  }}
]

Recommend 10-15 papers. Return ONLY the JSON array. Do not use tools."""

    try:
        result = subprocess.run(
            ["claude", "-p", "-", "--output-format", "text", "--max-turns", "1", "--model", "opus"],
            input=prompt, capture_output=True, text=True, timeout=180,
        )
        text = result.stdout.strip()
        # Parse JSON
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            papers = json.loads(text[start:end])
            results = []
            for p in papers:
                results.append({
                    "paperId": f"opus_{_query_hash(p.get('title', ''))}",
                    "title": p.get("title", ""),
                    "abstract": "",  # Opus 추천은 abstract 없음 — 나중에 채움
                    "key_idea": p.get("key_idea", ""),
                    "relevance": p.get("relevance", ""),
                    "venue": p.get("venue", ""),
                    "year": p.get("year"),
                    "arxiv": p.get("arxiv", ""),
                    "citations": 0,
                    "source": "opus",
                })
            return results
    except Exception as e:
        print(f"  ⚠️ Opus 추천 실패: {e}")
    return []


# ─── Merge + Rank ───────────────────────────────────────────────

def _merge_results(local: list[dict], opus: list[dict]) -> list[dict]:
    """2소스 결과 합치기 + 중복 제거."""
    seen_titles = {}  # normalized title → paper
    all_papers = []

    for source_name, papers in [("local_emb", local), ("opus", opus)]:
        for p in papers:
            title_norm = p.get("title", "").lower().strip()
            if not title_norm:
                continue
            if title_norm in seen_titles:
                # 중복 — source 추가
                existing = seen_titles[title_norm]
                sources = existing.get("sources", [existing.get("source", "")])
                if source_name not in sources:
                    sources.append(source_name)
                existing["sources"] = sources
                # abstract가 비어있으면 채우기
                if not existing.get("abstract") and p.get("abstract"):
                    existing["abstract"] = p["abstract"]
            else:
                p["sources"] = [p.pop("source", source_name)]
                seen_titles[title_norm] = p
                all_papers.append(p)

    return all_papers


def _rank_with_opus(query: str, papers: list[dict]) -> list[dict]:
    """Opus가 문제 적합도 순위 매기기."""
    papers_text = ""
    for i, p in enumerate(papers[:40]):  # 최대 40편
        title = p.get("title", "")
        abstract = (p.get("abstract", "") or "")[:200]
        venue = p.get("venue", "")
        year = p.get("year", "")
        sources = ", ".join(p.get("sources", []))
        papers_text += f"\n[{i}] {title}\n    {venue} {year} | sources: {sources}\n    {abstract}\n"

    prompt = f"""You are ranking papers by relevance to this research question:

"{query}"

Papers:
{papers_text}

Return a JSON array of indices sorted by relevance (most relevant first).
Include ONLY the top 15 most relevant papers.
Format: [index1, index2, ...]

Return ONLY the JSON array. Do not use tools."""

    try:
        result = subprocess.run(
            ["claude", "-p", "-", "--output-format", "text", "--max-turns", "1", "--model", "opus"],
            input=prompt, capture_output=True, text=True, timeout=120,
        )
        text = result.stdout.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            indices = json.loads(text[start:end])
            ranked = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(papers):
                    ranked.append(papers[idx])
            return ranked
    except Exception as e:
        print(f"  ⚠️ Opus 순위 매기기 실패: {e}")

    # Fallback: 여러 소스에서 나온 논문 우선
    papers.sort(key=lambda p: -len(p.get("sources", [])))
    return papers[:15]


# ─── Paper fetch + summary ──────────────────────────────────────

def _save_paper(paper: dict):
    """논문을 papers/에 저장 (이미 있으면 메타데이터만 업데이트)."""
    pid = paper.get("paperId", "")
    if not pid:
        pid = _query_hash(paper.get("title", "unknown"))
        paper["paperId"] = pid

    pfile = PAPERS_DIR / f"{pid}.json"
    if pfile.exists():
        existing = json.loads(pfile.read_text())
        # 기존 sources에 새 source 추가
        existing_sources = set(existing.get("sources", []))
        new_sources = set(paper.get("sources", []))
        existing["sources"] = list(existing_sources | new_sources)
        # abstract가 비어있으면 채우기
        if not existing.get("abstract") and paper.get("abstract"):
            existing["abstract"] = paper["abstract"]
        if not existing.get("summary") and paper.get("summary"):
            existing["summary"] = paper["summary"]
        pfile.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    else:
        paper["first_found"] = datetime.now().strftime("%Y-%m-%d")
        pfile.write_text(json.dumps(paper, ensure_ascii=False, indent=2))


def _summarize_paper(paper: dict) -> str:
    """논문 abstract를 Claude로 요약 (이미 요약이 있으면 재사용, abstract 없으면 검색)."""
    pid = paper.get("paperId", "")
    pfile = PAPERS_DIR / f"{pid}.json"

    # 이미 요약이 있는지 확인
    if pfile.exists():
        existing = json.loads(pfile.read_text())
        if existing.get("summary"):
            return existing["summary"]

    abstract = paper.get("abstract", "")
    title = paper.get("title", "")
    key_idea = paper.get("key_idea", "")

    # abstract 없으면 Claude에게 검색 + 요약 요청
    if not abstract and title:
        prompt = f"""Find and summarize this ML paper in 3-4 sentences focusing on: method, key innovation, results.

Title: {title}
{f"Hint: {key_idea}" if key_idea else ""}

Search the web if needed. Write in English. Be specific about the method name and results."""

        try:
            result = subprocess.run(
                ["claude", "-p", "-", "--output-format", "text", "--max-turns", "3",
                 "--model", "sonnet", "--tools", "WebSearch", "--allowedTools", "WebSearch"],
                input=prompt, capture_output=True, text=True, timeout=120,
            )
            summary = result.stdout.strip()
            if summary and not summary.startswith("(") and not summary.startswith("Error"):
                if pfile.exists():
                    data = json.loads(pfile.read_text())
                    data["summary"] = summary
                    pfile.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                return summary
        except Exception:
            pass
        # Fallback: key_idea가 있으면 사용
        return key_idea or ""

    if not abstract:
        return key_idea or ""

    # abstract 있으면 요약
    prompt = f"""Summarize this ML paper in 3-4 sentences focusing on: method, key innovation, results.

Title: {title}
Abstract: {abstract}

Write in English. Be specific about the method name and results. Do not use tools."""

    try:
        result = subprocess.run(
            ["claude", "-p", "-", "--output-format", "text", "--max-turns", "1", "--model", "haiku"],
            input=prompt, capture_output=True, text=True, timeout=60,
        )
        summary = result.stdout.strip()
        if summary and not summary.startswith("("):
            if pfile.exists():
                data = json.loads(pfile.read_text())
                data["summary"] = summary
                pfile.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return summary
    except Exception:
        pass
    return abstract[:300]


# ─── Context generation ────────────────────────────────────────

def _generate_context(query: str, ranked_papers: list[dict]) -> str:
    """회의용 최종 컨텍스트 생성 — Opus가 논문들을 종합 정리."""

    # 논문 정보를 텍스트로 모으기
    papers_text = ""
    for i, p in enumerate(ranked_papers, 1):
        title = p.get("title", "")
        venue = p.get("venue", "")
        year = p.get("year", "")
        citations = p.get("citations", 0)
        summary = p.get("summary", "")
        abstract = p.get("abstract", "")
        key_idea = p.get("key_idea", "")
        relevance = p.get("relevance", "")
        arxiv = p.get("arxiv", "")

        content = summary or abstract[:400] or key_idea or ""
        papers_text += f"\n[{i}] {title} ({venue} {year}, {citations} citations)"
        if arxiv:
            papers_text += f" arXiv:{arxiv}"
        papers_text += f"\n{content}\n"
        if key_idea and not summary:
            papers_text += f"Key idea: {key_idea}\n"
        if relevance:
            papers_text += f"Relevance: {relevance}\n"

    prompt = f"""You are writing a research briefing for an AI architecture expert who will participate in a meeting about:

"{query}"

Below are {len(ranked_papers)} relevant papers found from multiple sources. Synthesize them into a structured briefing.

Papers:
{papers_text}

Write the briefing in this structure:
1. **Problem Analysis**: What are the core challenges of this problem? (2-3 sentences)
2. **Approach Categories**: Group the papers into 3-5 categories by approach type. For each category:
   - Category name and key principle
   - Representative papers (title, year) with 1-sentence description
   - Strengths and limitations
3. **Key Insights**: What cross-cutting insights emerge from these papers? (3-5 bullet points)
4. **Recommendation**: Based on the papers, what approach(es) seem most promising and why?
5. **Paper References**: List all papers with title, venue, year, arXiv (for citation)

Write in English. Be specific about method names and results. Do not use tools."""

    try:
        result = subprocess.run(
            ["claude", "-p", "-", "--output-format", "text", "--max-turns", "1", "--model", "opus"],
            input=prompt, capture_output=True, text=True, timeout=300,
        )
        context = result.stdout.strip()
        if context and not context.startswith("(") and not context.startswith("Error"):
            header = f"# Research Context: {query}\n\nDate: {datetime.now().strftime('%Y-%m-%d')}\nPapers analyzed: {len(ranked_papers)}\n\n"
            return header + context
    except Exception as e:
        print(f"  ⚠️ Context synthesis failed: {e}")

    # Fallback: 기존 나열 방식
    lines = [f"# Research Context: {query}\n"]
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Papers found: {len(ranked_papers)}\n")
    for i, p in enumerate(ranked_papers, 1):
        title = p.get("title", "")
        summary = p.get("summary", p.get("abstract", "")[:300])
        lines.append(f"## {i}. {title}")
        lines.append(f"- {summary}")
        lines.append("")
    return "\n".join(lines)


# ─── Main pipeline ─────────────────────────────────────────────

def run_research(query: str) -> dict:
    """전체 리서치 파이프라인 실행."""
    _ensure_dirs()

    qhash = _query_hash(query)
    timestamp = datetime.now().strftime("%Y%m%d")
    research_id = f"{timestamp}_{qhash}"

    print(f"\n{'='*60}")
    print(f"  Researcher2 리서치 파이프라인")
    print(f"  쿼리: {query}")
    print(f"  ID: {research_id}")
    print(f"{'='*60}\n")

    # 0. 이전 결과 확인
    previous_papers = _load_previous_papers(query)
    if previous_papers:
        print(f"[이전 리서치] {len(previous_papers)}편 이전에 찾은 논문 발견")

    # 1. HyDE
    print("[Step 1/6] HyDE — 가상 abstract 생성...")
    hyde_abstract = _generate_hyde(query)
    print(f"  생성 완료 ({len(hyde_abstract)} chars)")

    # 2. 2소스 병렬 검색
    print("[Step 2/6] 2소스 병렬 검색...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_local = executor.submit(_search_local_embeddings, hyde_abstract, 30)
        f_opus = executor.submit(_search_opus, query, previous_papers)

        local_results = f_local.result()
        opus_results = f_opus.result()

    print(f"  ① 로컬 임베딩: {len(local_results)}편")
    print(f"  ② Opus 추천: {len(opus_results)}편")

    # 3. Merge
    print("[Step 3/6] 결과 합치기 + 중복 제거...")
    merged = _merge_results(local_results, opus_results)
    multi_source = [p for p in merged if len(p.get("sources", [])) > 1]
    print(f"  합계: {len(merged)}편 (다중 소스: {len(multi_source)}편)")

    # 4. Opus 순위
    print("[Step 4/6] Opus 순위 매기기...")
    ranked = _rank_with_opus(query, merged)
    print(f"  상위 {len(ranked)}편 선정")

    # 5. 논문 저장 + 요약
    print("[Step 5/6] 논문 저장 + 요약...")
    for i, p in enumerate(ranked):
        _save_paper(p)
        summary = _summarize_paper(p)
        p["summary"] = summary
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(ranked)} 요약 완료")

    # 6. 컨텍스트 생성
    print("[Step 6/6] 컨텍스트 생성...")
    context = _generate_context(query, ranked)

    # 저장
    context_file = CONTEXTS_DIR / f"{research_id}.md"
    context_file.write_text(context, encoding="utf-8")

    # 쿼리 저장
    query_data = {
        "query": query,
        "hyde_abstract": hyde_abstract,
        "results": {
            "local_emb": [p.get("title", "") for p in local_results[:10]],
            "opus": [p.get("title", "") for p in opus_results[:10]],
        },
        "ranked": [p.get("paperId", "") for p in ranked],
        "date": datetime.now().isoformat(),
    }
    (QUERIES_DIR / f"{research_id}.json").write_text(
        json.dumps(query_data, ensure_ascii=False, indent=2)
    )

    # index 업데이트
    index = _load_index()
    # 키워드 추출 (간단)
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    index.append({
        "id": research_id,
        "topic": query,
        "keywords": keywords,
        "papers_found": len(ranked),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "file": f"{research_id}.md",
    })
    _save_index(index)

    print(f"\n{'='*60}")
    print(f"  리서치 완료!")
    print(f"  {len(ranked)}편 논문 정리")
    print(f"  컨텍스트: {context_file}")
    print(f"{'='*60}\n")

    # 상위 5편 미리보기
    for i, p in enumerate(ranked[:5], 1):
        print(f"  {i}. {p.get('title', '')}")
        print(f"     {p.get('venue', '')} {p.get('year', '')} | sources: {', '.join(p.get('sources', []))}")
        print()

    return {
        "research_id": research_id,
        "query": query,
        "ranked_papers": ranked,
        "context_file": str(context_file),
        "context": context,
    }


def get_research_context(query: str) -> str:
    """회의 시 호출: 항상 새로 리서치 실행.

    이전 리서치 결과는 run_research 내부에서 중복 논문 제외용으로만 활용.
    캐시 재활용은 회의 품질을 떨어뜨리므로 하지 않는다.
    """
    print("[Researcher2] 새 리서치 실행...")
    result = run_research(query)
    return result["context"]


# ─── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Researcher2 리서치 파이프라인")
    parser.add_argument("command", choices=["research", "list", "show"],
                        help="research: 리서치 실행, list: 이전 리서치 목록, show: 컨텍스트 보기")
    parser.add_argument("query", nargs="?", default=None)

    args = parser.parse_args()

    if args.command == "research":
        if not args.query:
            print("쿼리를 입력하세요.")
            return
        run_research(args.query)

    elif args.command == "list":
        _ensure_dirs()
        index = _load_index()
        if not index:
            print("이전 리서치 없음.")
            return
        print(f"\n=== 이전 리서치 ({len(index)}건) ===\n")
        for entry in index:
            print(f"  [{entry['date']}] {entry['topic']} ({entry['papers_found']}편)")
            print(f"    ID: {entry['id']}")
            print()

    elif args.command == "show":
        if not args.query:
            print("리서치 ID를 입력하세요.")
            return
        ctx_file = CONTEXTS_DIR / f"{args.query}.md"
        if ctx_file.exists():
            print(ctx_file.read_text())
        else:
            print(f"파일 없음: {ctx_file}")


if __name__ == "__main__":
    main()
