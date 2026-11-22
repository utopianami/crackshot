import numpy as np
import pandas as pd
import pytest

from examples.backtest_sample import run_sample
from scripts.binance.build_npy_phase2 import _anchor_4h_utc_stride


@pytest.mark.parametrize("timezone", [None, "Asia/Seoul"])
def test_sample_stays_on_utc_boundaries_after_missing_bars(timezone):
    times = pd.date_range("2025-01-01", periods=145, freq="5min", tz="UTC")
    expected = times[[0, 48, 144]]
    # An interior bar and the 08:00 boundary itself are missing.
    times = times.delete([17, 96])
    if timezone:
        times = times.tz_convert(timezone)
    else:
        times = times.tz_localize(None)
    result = _anchor_4h_utc_stride(pd.DataFrame({"open_time": times}))
    assert list(pd.to_datetime(result.open_time, utc=True)) == list(expected)


@pytest.fixture(scope="module")
def backtest_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("portfolio")
    run_sample(directory)
    stats = pd.read_csv(directory / "model/backtest_test.csv", parse_dates=["rel_t"])
    weights = pd.read_parquet(directory / "weights.parquet")
    return stats, weights


def test_csv_reports_only_evaluated_timestamps(backtest_files):
    stats, weights = backtest_files
    expected = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 08:00", "2025-01-01 12:00"], utc=True)
    assert list(stats.rel_t) == list(expected)
    assert list(weights.timestamp.drop_duplicates()) == list(expected)


def test_turnover_matches_trades_by_symbol_and_fees(backtest_files):
    stats, weights = backtest_files
    stats = stats.set_index("rel_t")
    positions = weights.pivot(index="timestamp", columns="symbol", values="weight").fillna(0)
    expected = positions.sub(positions.shift(fill_value=0)).abs().sum(axis=1)
    # The first two evaluated portfolios hold entirely different assets.
    assert expected.iloc[1] == pytest.approx(positions.iloc[:2].abs().sum().sum())
    assert expected.iloc[1] > 0
    np.testing.assert_allclose(stats.turnover_l1, expected, rtol=1e-6)
    np.testing.assert_allclose(stats.net_ret, stats.port_ret - 0.0005 * expected, atol=1e-9)
