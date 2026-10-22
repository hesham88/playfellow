"""Tests for timing summary statistics."""
from timing import summarize_timing


def test_empty_timings():
    """Empty lists return zeros."""
    result = summarize_timing([], [])
    assert result == {"median": 0.0, "p90": 0.0, "backup_share": 0.0, "local_share": 0.0}


def test_single_timing():
    """Single value is both median and p90."""
    result = summarize_timing([5.5], ["primary"])
    assert result["median"] == 5.5
    assert result["p90"] == 5.5
    assert result["backup_share"] == 0.0
    assert result["local_share"] == 0.0


def test_even_count_median():
    """Even count: median is average of middle two."""
    result = summarize_timing([1.0, 2.0, 3.0, 4.0], ["primary"] * 4)
    assert result["median"] == 2.5


def test_odd_count_median():
    """Odd count: median is middle value."""
    result = summarize_timing([1.0, 2.0, 5.0], ["primary"] * 3)
    assert result["median"] == 2.0


def test_p90_calculation():
    """P90 is the 90th percentile value."""
    timings = [float(i) for i in range(1, 11)]  # 1.0 to 10.0
    result = summarize_timing(timings, ["primary"] * 10)
    # 90% of 10 = index 9, which is value 10.0
    assert result["p90"] == 10.0


def test_backup_share():
    """Backup share is fraction of backup brain uses."""
    result = summarize_timing([1.0] * 5, ["primary", "backup", "primary", "backup", "local"])
    assert result["backup_share"] == 0.4  # 2 out of 5


def test_local_share():
    """Local share is fraction of local brain uses."""
    result = summarize_timing([1.0] * 5, ["primary", "backup", "primary", "backup", "local"])
    assert result["local_share"] == 0.2  # 1 out of 5


def test_all_primary():
    """All primary brain uses."""
    result = summarize_timing([2.0, 3.0, 4.0], ["primary", "primary", "primary"])
    assert result["backup_share"] == 0.0
    assert result["local_share"] == 0.0


def test_all_backup():
    """All backup brain uses."""
    result = summarize_timing([2.0, 3.0, 4.0], ["backup", "backup", "backup"])
    assert result["backup_share"] == 1.0
    assert result["local_share"] == 0.0


def test_all_local():
    """All local brain uses."""
    result = summarize_timing([2.0, 3.0, 4.0], ["local", "local", "local"])
    assert result["backup_share"] == 0.0
    assert result["local_share"] == 1.0


def test_mixed_scenario():
    """Realistic mix of timings and brain types."""
    timings = [2.1, 3.5, 1.8, 4.2, 2.9, 5.1, 3.3, 2.6, 4.8, 3.1]
    brains = ["primary", "primary", "backup", "primary", "local", "primary", "backup", "primary", "local", "primary"]
    result = summarize_timing(timings, brains)

    # Median of sorted [1.8, 2.1, 2.6, 2.9, 3.1, 3.3, 3.5, 4.2, 4.8, 5.1] is (3.1 + 3.3) / 2
    assert result["median"] == 3.2
    # P90 is at index 9, which is 5.1 (the last/highest value)
    assert result["p90"] == 5.1
    # 2 backup out of 10
    assert result["backup_share"] == 0.2
    # 2 local out of 10
    assert result["local_share"] == 0.2
