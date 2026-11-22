"""Pre-meeting research phase: static (cached) + topic-specific."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from alpha_squad.config import latest_research_note

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "alpha_squad" / "research_cache"


def _claude_query(prompt: str, timeout: int = 180) -> str:
    """Call claude CLI with a prompt and return the response."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text", "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() or "(응답 없음)"
    except subprocess.TimeoutExpired:
        return "(타임아웃)"
    except Exception as e:
        return f"(에러: {e})"


def _read_file(path: str, max_chars: int = 5000) -> str:
    """Read a file, truncated to max_chars."""
    full = PROJECT_ROOT / path
    if not full.exists():
        return f"(파일 없음: {path})"
    text = full.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated, total {len(text)} chars)"
    return text


def _run_python(code: str, timeout: int = 60) -> str:
    """Run a Python snippet and return stdout."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n[STDERR]: {result.stderr.strip()}"
        return output or "(출력 없음)"
    except subprocess.TimeoutExpired:
        return "(타임아웃)"
    except Exception as e:
        return f"(에러: {e})"


# ─── Static research (topic-independent, cached) ────────────────


def _static_cache_path() -> Path:
    """Today's static cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"static_{datetime.now().strftime('%Y%m%d')}.json"


def _load_static_cache() -> dict | None:
    path = _static_cache_path()
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_static_cache(data: dict):
    path = _static_cache_path()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _static_feature_analysis() -> str:
    """Expert1: feature correlation, NaN rates, variance."""
    code = """
import numpy as np
from pathlib import Path

import os
npy_dir = Path(os.environ.get("KOSPI_NPY_DIR", "data/kospi/260401_200_base"))

X_path = npy_dir / "X_train.npy"
if X_path.exists():
    X = np.load(X_path, mmap_mode='r')
    print(f"Feature shape: {X.shape}")
    print(f"Feature count: {X.shape[-1]}")

    if len(X) > 50000:
        idx = np.random.RandomState(42).choice(len(X), 50000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = np.array(X)

    nan_rate = np.isnan(X_sample).mean(axis=0)
    print(f"\\nNaN rate per feature (top 10 worst):")
    worst = np.argsort(nan_rate)[-10:][::-1]
    for i in worst:
        print(f"  feat[{i}]: {nan_rate[i]:.3f}")

    X_clean = np.nan_to_num(X_sample, nan=0.0)
    corr = np.corrcoef(X_clean.T)
    n = corr.shape[0]
    high_corr_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i,j]) > 0.9:
                high_corr_pairs.append((i, j, corr[i,j]))
    high_corr_pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\\nHighly correlated pairs (|r| > 0.9): {len(high_corr_pairs)}")
    for i, j, r in high_corr_pairs[:20]:
        print(f"  feat[{i}] - feat[{j}]: r={r:.3f}")

    var = np.nanvar(X_sample, axis=0)
    low_var = np.argsort(var)[:10]
    print(f"\\nLowest variance features:")
    for i in low_var:
        print(f"  feat[{i}]: var={var[i]:.6f}")
else:
    print(f"X_train.npy not found in {npy_dir}")
"""
    return _run_python(code, timeout=120)


def _static_label_analysis() -> str:
    """Expert2: label distribution analysis."""
    code = """
import numpy as np
from pathlib import Path

import os
npy_dir = Path(os.environ.get("KOSPI_NPY_DIR", "data/kospi/260401_200_base"))

y_path = npy_dir / "y_train.npy"
if y_path.exists():
    y = np.load(y_path, mmap_mode='r')
    print(f"Label shape: {y.shape}")
    print(f"Label columns: {y.shape[-1]}")

    for col_idx in [0, 1, 21]:
        if col_idx < y.shape[-1]:
            col = y[:, col_idx] if y.ndim == 2 else y[..., col_idx].flatten()
            valid = col[~np.isnan(col)]
            print(f"\\ny[:, {col_idx}] stats:")
            print(f"  count: {len(valid)}, NaN: {np.isnan(col).sum()}")
            print(f"  mean: {valid.mean():.6f}, std: {valid.std():.6f}")
            print(f"  min: {valid.min():.6f}, max: {valid.max():.6f}")
            print(f"  >0 ratio: {(valid > 0).mean():.3f}")
            for q in [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]:
                print(f"  q{q:.0%}: {np.quantile(valid, q):.6f}")
else:
    print(f"y_train.npy not found in {npy_dir}")
"""
    return _run_python(code, timeout=60)


def _static_loss_implementations() -> str:
    """Expert2: read loss function code."""
    loss_dir = PROJECT_ROOT / "src" / "training" / "losses"
    parts = []
    if loss_dir.exists():
        for f in sorted(loss_dir.glob("*.py")):
            if f.name != "__init__.py":
                parts.append(f"### {f.name}\n{f.read_text()[:1500]}")
    return "\n\n".join(parts[:3]) if parts else "(loss 파일 없음)"


def _static_experiment_results() -> str:
    """Researcher1: current experiment results."""
    return _read_file("autoresearch/records/kospi-results.csv", max_chars=4000)


def _static_latest_research() -> str:
    """Researcher1: latest autoresearch file."""
    note = latest_research_note(PROJECT_ROOT)
    return note.read_text(encoding="utf-8")[:3000] if note else ""


def _static_codebase_structure() -> str:
    """Researcher3 PM: models, configs, data directories."""
    models = []
    arch_dir = PROJECT_ROOT / "src" / "models" / "architectures"
    if arch_dir.exists():
        models = [f.stem for f in arch_dir.glob("*.py") if f.name != "__init__.py"]

    configs = []
    cfg_dir = PROJECT_ROOT / "configs"
    if cfg_dir.exists():
        configs = [f.stem for f in cfg_dir.glob("*.yaml")]

    data_code = """
from pathlib import Path
for base in ["data/kospi/", "data/binance/"]:
    p = Path(base)
    if p.exists():
        print(f"\\n{base}:")
        for d in sorted(p.iterdir()):
            if d.is_dir():
                files = list(d.glob("*.npy"))
                print(f"  {d.name}/ ({len(files)} npy)")
"""
    data_result = _run_python(data_code, timeout=30)

    return f"""### 구현된 모델
{', '.join(sorted(models))}

### Config 파일
{', '.join(sorted(configs))}

### 데이터 디렉토리
{data_result}"""


def _static_model_config() -> str:
    """Researcher1: current best model config."""
    return _read_file("configs/cs_mlp_200_base.yaml", max_chars=2000)


def _static_feature_docs() -> str:
    """Expert1: feature documentation."""
    data_md = _read_file("research/kospi.md", max_chars=2000)
    feat_md = _read_file("research/binance.md", max_chars=2000)
    return f"{data_md}\n\n{feat_md}"


def _static_constraints() -> str:
    """PM: project constraints."""
    return _read_file("autoresearch/README.md", max_chars=2000)


def run_static_research(force: bool = False) -> dict:
    """Run static (topic-independent) research. Cached per day.

    Returns dict with keys for each data source.
    """
    if not force:
        cached = _load_static_cache()
        if cached:
            print("[Static Research] Using today's cache")
            return cached

    print("[Static Research] Gathering project data (cached for today)...")

    tasks = {
        "feature_analysis": _static_feature_analysis,
        "label_analysis": _static_label_analysis,
        "loss_code": _static_loss_implementations,
        "experiment_results": _static_experiment_results,
        "latest_research": _static_latest_research,
        "codebase_structure": _static_codebase_structure,
        "model_config": _static_model_config,
        "feature_docs": _static_feature_docs,
        "constraints": _static_constraints,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(func): name for name, func in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"  [{name}] done ({len(results[name])} chars)")
            except Exception as e:
                results[name] = f"(실패: {e})"
                print(f"  [{name}] FAILED: {e}")

    _save_static_cache(results)
    print("[Static Research] Complete and cached.")
    return results


# ─── Topic-specific research ────────────────────────────────────


def _topic_researcher2(topic: str) -> str:
    """Researcher2: topic-specific architecture/paper search."""
    prompt = f"""당신은 ML 리서처입니다. 다음 주제에 대해 최신 딥러닝 아키텍처와 관련 논문을 조사하세요.

주제: {topic}

다음을 포함해서 간결하게 정리:
1. 이 주제와 관련된 최신 아키텍처 (2024-2025)
2. 금융 시계열/tabular 데이터에 적용 가능한 방법론
3. 각 방법론의 핵심 아이디어와 장단점
4. 구체적 논문 이름과 핵심 contribution

한국어로 작성하되, 논문 이름은 영어 원문 유지."""
    return _claude_query(prompt, timeout=120)


def _topic_expert1(topic: str) -> str:
    """Expert1: topic-specific feature perspective."""
    prompt = f"""당신은 금융 피처 엔지니어링 전문가입니다. 다음 주제에 대해 피처 관점에서 분석하세요.

주제: {topic}

다음을 포함:
1. 이 주제에서 중요한 피처 카테고리
2. 피처 leakage 위험이 있는 부분
3. regime 변화에 robust한 피처 설계 방안
4. 구체적 피처 아이디어 (수식 포함)

한국어로 작성."""
    return _claude_query(prompt, timeout=120)


def _topic_expert2(topic: str) -> str:
    """Expert2: topic-specific problem definition perspective."""
    prompt = f"""당신은 ML 문제 정의 전문가입니다. 금융 데이터 맥락에서 다음 주제를 비판적으로 분석하세요.

주제: {topic}

다음을 포함:
1. 이 주제에서 문제 정의의 근본적 결함 가능성
2. loss function과 실제 투자 목적의 alignment
3. label noise 관점에서의 위험
4. 대안적 문제 정의 (formulation) 제안

한국어로 작성."""
    return _claude_query(prompt, timeout=120)


def _topic_researcher1(topic: str) -> str:
    """Researcher1: topic-specific financial feasibility."""
    prompt = f"""당신은 금융 시계열 ML 엔지니어입니다. 다음 주제를 금융 데이터 실전 경험 관점에서 분석하세요.

주제: {topic}

다음을 포함:
1. 금융 데이터에서 이 접근이 실제 작동한 사례/실패 사례
2. non-stationarity, regime change 관점에서의 위험
3. 실현 가능성과 구현 난이도 평가
4. 기대 효과 대비 리스크 분석

한국어로 작성."""
    return _claude_query(prompt, timeout=120)


def run_topic_research(topic: str) -> dict[str, str]:
    """Run topic-specific research for all agents in parallel.

    Returns dict mapping agent name to research content.
    """
    print(f"[Topic Research] Researching: {topic}")

    tasks = {
        "Researcher2_AI아키텍처": _topic_researcher2,
        "Expert1_피처전문가": _topic_expert1,
        "Expert2_문제정의전문가": _topic_expert2,
        "Researcher1_금융모델링": _topic_researcher1,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(func, topic): name for name, func in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"  [{name}] done ({len(results[name])} chars)")
            except Exception as e:
                results[name] = f"(리서치 실패: {e})"
                print(f"  [{name}] FAILED: {e}")

    # PM doesn't need topic research — they synthesize others' work
    results["Researcher3_PM"] = ""
    print("[Topic Research] Complete.")
    return results


# ─── Combined research phase ────────────────────────────────────


def run_research_phase(topic: str, parallel: bool = True) -> dict[str, str]:
    """Run full research phase: static (cached) + topic-specific.

    Returns dict mapping agent name to combined research context.
    """
    # 1. Static research (cached per day)
    static = run_static_research()

    # 2. Topic-specific research (always fresh)
    topic_results = run_topic_research(topic)

    # 3. Combine per agent
    combined = {}

    combined["Researcher2_AI아키텍처"] = f"""[토픽 리서치]
{topic_results.get("Researcher2_AI아키텍처", "")}"""

    combined["Expert1_피처전문가"] = f"""[피처 데이터 분석]
{static.get("feature_analysis", "")}

[피처 문서]
{static.get("feature_docs", "")[:1500]}

[토픽 리서치]
{topic_results.get("Expert1_피처전문가", "")}"""

    combined["Expert2_문제정의전문가"] = f"""[Loss 구현]
{static.get("loss_code", "")[:1500]}

[Label 분포]
{static.get("label_analysis", "")}

[실험 결과]
{static.get("experiment_results", "")[:1500]}

[토픽 리서치]
{topic_results.get("Expert2_문제정의전문가", "")}"""

    combined["Researcher1_금융모델링"] = f"""[실험 결과]
{static.get("experiment_results", "")}

[최신 실험 기록]
{static.get("latest_research", "")[:2000]}

[현재 SOTA Config]
{static.get("model_config", "")}

[토픽 리서치]
{topic_results.get("Researcher1_금융모델링", "")}"""

    combined["Researcher3_PM"] = f"""[코드베이스 구조]
{static.get("codebase_structure", "")}

[프로젝트 제약]
{static.get("constraints", "")[:1500]}

[실험 결과]
{static.get("experiment_results", "")[:2000]}"""

    # Save research results
    output_dir = PROJECT_ROOT / "alpha_squad" / "research_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for agent_name, content in combined.items():
        safe_name = agent_name.split("_")[0].lower()
        filename = f"{timestamp}_{safe_name}.md"
        (output_dir / filename).write_text(
            f"# Research: {agent_name}\n## Topic: {topic}\n\n{content}",
            encoding="utf-8",
        )

    return combined
