"""study-agent: 웹 검색 기반 지식 수집 + snowball 확장.

Usage:
    PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert1 "alpha factor design" --depth 2
    PYTHONPATH="." uv run python alpha_squad/knowledge/study_agent.py expert2 "ranking loss functions financial ML" --depth 3
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alpha_squad.knowledge.lightrag_backend import AlphaSquadKnowledge


def _claude_search(query: str, context: str = "", timeout: int = 300) -> str:
    """Claude CLI with web search enabled to research a topic."""
    prompt = f"""다음 주제에 대해 깊이 있게 조사하세요. 반드시 웹 검색을 활용해서 최신 정보를 포함하세요.

주제: {query}

{f"배경 컨텍스트: {context}" if context else ""}

다음을 포함해서 상세하게 작성:
1. 핵심 개념과 원리 (수식 포함 가능)
2. 관련 논문/연구 (제목, 저자, 연도)
3. 장점과 한계
4. 금융/트레이딩 데이터에서의 적용 사례
5. 실전에서의 주의점

한국어로 작성하되, 논문명과 기술 용어는 영어 유지."""

    try:
        result = subprocess.run(
            [
                "claude",
                "-p", "-",
                "--output-format", "text",
                "--max-turns", "5",
                "--model", "sonnet",
                "--tools", "WebSearch",
                "--allowedTools", "WebSearch",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = result.stdout.strip()
        if not text:
            err = result.stderr.strip()
            return f"(검색 실패: {err[:200]})" if err else "(응답 없음)"
        return text
    except subprocess.TimeoutExpired:
        return "(타임아웃)"
    except Exception as e:
        return f"(에러: {e})"


def _extract_new_keywords(research_result: str, explored: set[str], timeout: int = 60) -> list[str]:
    """Claude에게 리서치 결과에서 새로운 키워드를 추출하게 함."""
    explored_str = ", ".join(explored)
    prompt = f"""다음 리서치 결과를 읽고, 더 깊이 조사해야 할 새로운 키워드/주제 5개를 추출하세요.

이미 탐색한 키워드: {explored_str}

위 키워드와 겹치지 않는, 구체적이고 검색 가능한 키워드만 뽑으세요.
금융 ML / 트레이딩 맥락에서 의미 있는 것만.

리서치 결과:
{research_result[:3000]}

JSON 배열로만 응답하세요. 다른 텍스트 없이.
예: ["keyword1", "keyword2", "keyword3"]"""

    try:
        result = subprocess.run(
            [
                "claude",
                "-p", "-",
                "--output-format", "text",
                "--max-turns", "1",
                "--model", "haiku",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = result.stdout.strip()
        # Parse JSON array
        # Find the JSON array in the response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            keywords = json.loads(text[start:end])
            return [k for k in keywords if isinstance(k, str) and k.strip()]
        return []
    except Exception:
        return []


async def study_agent(
    agent: str,
    seed_query: str,
    depth: int = 2,
    context: str = "",
) -> dict:
    """Snowball 방식으로 지식을 수집하고 LightRAG에 저장.

    Args:
        agent: 에이전트 이름 (expert1, expert2, researcher1, researcher2, researcher3)
        seed_query: 시드 검색어
        depth: snowball 깊이 (1=시드만, 2=1단계 확장, 3=2단계 확장)
        context: 추가 배경 컨텍스트

    Returns:
        dict with stats about what was collected
    """
    knowledge = AlphaSquadKnowledge()
    explored: set[str] = set()
    queue: list[str] = [seed_query]
    total_ingested = 0
    all_results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Study Agent: {agent}")
    print(f"  Seed: {seed_query}")
    print(f"  Depth: {depth}")
    print(f"{'='*60}\n")

    for current_depth in range(depth):
        if not queue:
            print(f"\n[Depth {current_depth+1}] 더 이상 새로운 키워드 없음. 종료.")
            break

        next_queue: list[str] = []
        print(f"\n[Depth {current_depth+1}/{depth}] 탐색할 키워드 {len(queue)}개:")
        for q in queue:
            print(f"  - {q}")

        for query in queue:
            if query in explored:
                continue
            explored.add(query)

            # 1. 웹 검색 + 리서치
            print(f"\n  🔍 검색 중: {query}")
            result = _claude_search(query, context=context)

            if result.startswith("(") and result.endswith(")"):
                print(f"  ⚠️ 실패: {result}")
                continue

            print(f"  ✅ 수집: {len(result)} chars")

            # 2. LightRAG에 ingest
            await knowledge.ingest(agent, result, source=f"study:{query}")
            total_ingested += 1
            all_results.append({
                "query": query,
                "depth": current_depth + 1,
                "chars": len(result),
            })

            # 3. 새로운 키워드 추출 (마지막 depth가 아닐 때만)
            if current_depth < depth - 1:
                new_keywords = _extract_new_keywords(result, explored)
                new_keywords = [k for k in new_keywords if k not in explored]
                if new_keywords:
                    print(f"  🌱 새 키워드 발견: {new_keywords}")
                    next_queue.extend(new_keywords)

        queue = next_queue

    await knowledge.close()

    # Save study log
    log_dir = Path(__file__).parent.parent / "study_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{agent}.json"
    log_data = {
        "agent": agent,
        "seed_query": seed_query,
        "depth": depth,
        "total_ingested": total_ingested,
        "explored_keywords": list(explored),
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"  Study 완료!")
    print(f"  총 {total_ingested}건 수집, {len(explored)}개 키워드 탐색")
    print(f"  로그: {log_path}")
    print(f"{'='*60}\n")

    return log_data


# ─── 에이전트별 시드 키워드 ──────────────────────────────────────

SEED_TOPICS = {
    "expert1": [
        "alpha factor design quantitative finance feature engineering",
        "microstructure features VPIN Kyle lambda order flow imbalance",
        "feature selection IC information coefficient decay half-life financial ML",
        "look-ahead bias survivorship bias financial feature engineering",
        "cross-sectional normalization z-score rank transform financial data",
    ],
    "expert2": [
        "ListNet LambdaRank ranking loss learning to rank financial prediction",
        "focal loss label noise handling low SNR financial machine learning",
        "backtest overfitting deflated Sharpe ratio CSCV probability",
        "walk-forward validation purged k-fold time series cross-validation finance",
        "multi-task multi-horizon prediction financial deep learning",
    ],
    "researcher1": [
        "tree-based vs deep learning tabular data benchmark 2024 2025",
        "cross-sectional vs per-stock learning financial prediction tradeoff",
        "regularization deep learning financial data dropout weight decay label smoothing",
        "ensemble methods financial ML feature diversity model diversity stacking",
        "non-stationarity regime change adaptive learning financial time series",
    ],
    "researcher3": [
        "hyperparameter optimization Bayesian Hyperband multi-fidelity efficient",
        "reproducibility statistical significance financial ML experiments",
        "model decay monitoring concept drift detection reference ML",
        "experiment management MLflow Weights-and-Biases ablation study design",
    ],
}


async def study_all_seeds(agent: str, depth: int = 2) -> None:
    """에이전트의 모든 시드 키워드로 study 실행."""
    seeds = SEED_TOPICS.get(agent, [])
    if not seeds:
        print(f"에이전트 '{agent}'에 대한 시드 키워드가 없습니다.")
        print(f"사용 가능: {list(SEED_TOPICS.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  {agent}: {len(seeds)}개 시드 키워드로 전체 study 시작")
    print(f"{'='*60}")

    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] 시드: {seed}")
        await study_agent(agent, seed, depth=depth)


def main():
    parser = argparse.ArgumentParser(description="Study Agent: 웹 검색 기반 지식 수집")
    parser.add_argument("agent", choices=AlphaSquadKnowledge.AGENTS, help="에이전트 이름")
    parser.add_argument("query", nargs="?", default=None, help="검색 주제 (없으면 시드 전체 실행)")
    parser.add_argument("--depth", type=int, default=2, help="Snowball 깊이 (default: 2)")

    args = parser.parse_args()

    if args.query:
        asyncio.run(study_agent(args.agent, args.query, depth=args.depth))
    else:
        asyncio.run(study_all_seeds(args.agent, depth=args.depth))


if __name__ == "__main__":
    main()
