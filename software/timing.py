"""Timing statistics for PlayFellow runs. / إحصائيات التوقيت لعمليات تشغيل بلاي فيلو."""
from __future__ import annotations


def summarize_timing(timings: list[float], brain_uses: list[str]) -> dict[str, float | int]:
    """Calculate timing summary: median, p90, and share of backup/local brain use.

    Args:
        timings: List of step times in seconds
        brain_uses: List of brain types used ("backup", "local", or "primary")

    Returns:
        Dictionary with:
        - median: median step time in seconds
        - p90: 90th percentile step time in seconds
        - backup_share: fraction of steps using backup brain (0-1)
        - local_share: fraction of steps using local brain (0-1)
    """
    if not timings:
        return {"median": 0.0, "p90": 0.0, "backup_share": 0.0, "local_share": 0.0}

    # Sort timings for percentile calculations
    sorted_timings = sorted(timings)
    n = len(sorted_timings)

    # Median: middle value
    if n % 2 == 0:
        median = (sorted_timings[n // 2 - 1] + sorted_timings[n // 2]) / 2
    else:
        median = sorted_timings[n // 2]

    # P90: 90th percentile
    p90_index = min(int(n * 0.9), n - 1)
    p90 = sorted_timings[p90_index]

    # Count backup and local uses
    total_steps = len(brain_uses)
    if total_steps == 0:
        backup_share = 0.0
        local_share = 0.0
    else:
        backup_count = brain_uses.count("backup")
        local_count = brain_uses.count("local")
        backup_share = backup_count / total_steps
        local_share = local_count / total_steps

    return {
        "median": round(median, 2),
        "p90": round(p90, 2),
        "backup_share": round(backup_share, 3),
        "local_share": round(local_share, 3),
    }
