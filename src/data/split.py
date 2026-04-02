"""Offsets 기반 train/val split 생성."""

import numpy as np


def build_splits(
    offsets: dict,
    y: np.ndarray,
    *,
    vol_idx: int = -1,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """offsets dict + y array에서 유효한 train/val 인덱스를 반환.

    필터: vol > 0 AND target != 0.

    Args:
        offsets: {stock_id: {"start", "train_end", "val_end", "end"}}
        y: [N, L] 라벨 배열
        vol_idx: volume 컬럼 인덱스
        target_idx: target 컬럼 인덱스
    """
    tr_indices = []
    va_indices = []

    for info in offsets.values():
        start = info["start"]
        train_end = info["train_end"]
        val_end = info["val_end"]

        for i in range(start, train_end):
            vol = y[i, vol_idx]
            tgt = y[i, target_idx]
            if vol != 0 and tgt != 0 and not (vol != vol) and not (tgt != tgt):
                tr_indices.append(i)
        for i in range(train_end, val_end):
            vol = y[i, vol_idx]
            tgt = y[i, target_idx]
            if vol != 0 and tgt != 0 and not (vol != vol) and not (tgt != tgt):
                va_indices.append(i)

    return np.array(tr_indices, dtype=np.int64), np.array(va_indices, dtype=np.int64)
