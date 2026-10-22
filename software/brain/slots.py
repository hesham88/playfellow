"""Four sockets in the built-in brain where a LEARNED part can replace a hand-written rule. / أربع فتحات لأجزاء متعلَّمة.

Think of a toy robot with four sockets. Today each one holds a simple rule that always works. When a learned part has
been trained, it is plugged into its socket, but it is only USED when ALL of these are true:
  1. someone switched the socket on (every socket starts OFF),
  2. the part has a passed gate report: on layouts it never trained on it did better than the rule, by a clear margin,
     on at least MIN_LAYOUTS different layouts, and never worse than the rule on any of them,
  3. it is sure enough about this answer (its own confidence), and
  4. it did not fail. If it fails, is unsure, or answers with nonsense, the rule's answer is used: always, the same
     way every time. The rule is the safety floor and is never deleted.
Every choice is counted (`usage()`), so a log can say which one ran.

  S1_see     - where am I in the picture (rule: press keys and watch what moves)       [socket + gate only: not wired yet]
  S2_hud     - did the score bar change (rule: the top-bar change counter)                 -> BuiltinNavBrain._bar
  S3_policy  - which key next (rule: the planner's route)                                 -> BuiltinNavBrain.ask
  S4_value   - which move pays (rule: the practice values)                                -> Planner._valued
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SLOTS = ("S1_see", "S2_hud", "S3_policy", "S4_value")
WIRED = {"S1_see": False, "S2_hud": True, "S3_policy": True, "S4_value": True}  # has the brain a place that asks this socket?
MIN_LAYOUTS = 3  # a gate needs at least this many different held-out layouts
MARGIN = 0.05  # and the learned part's average must beat the rule's by at least this much
MIN_CONFIDENCE = 0.85  # below this the learned part's answer is not used


class GateError(ValueError):
    """The gate report does not justify plugging the part in."""


def gate_report(slot: str, metric: str, rule_scores: dict[str, float], learned_scores: dict[str, float],
                higher_is_better: bool = True) -> dict:
    """Compare the rule and the learned part layout by layout (scores for layouts the part never trained on).
    Returns the report that `SlotRegistry.promote` checks; `passed` is only a summary, promote() re-checks everything."""
    if slot not in SLOTS:
        raise GateError(f"unknown socket {slot!r}")
    layouts = sorted(set(rule_scores) & set(learned_scores))
    sign = 1.0 if higher_is_better else -1.0
    gains = [sign * (learned_scores[k] - rule_scores[k]) for k in layouts]
    mean_gain = sum(gains) / len(gains) if gains else 0.0
    return {"slot": slot, "metric": metric, "layouts": layouts, "rule": {k: rule_scores[k] for k in layouts},
            "learned": {k: learned_scores[k] for k in layouts}, "higher_is_better": higher_is_better,
            "mean_gain": mean_gain, "worst_gain": min(gains, default=0.0), "margin": MARGIN,
            "passed": len(layouts) >= MIN_LAYOUTS and mean_gain >= MARGIN and min(gains) >= 0.0}


def check_report(slot: str, report: dict) -> None:
    """Raise GateError unless the report really shows the learned part beating the rule on held-out layouts."""
    if report.get("slot") != slot:
        raise GateError(f"this report is for {report.get('slot')!r}, not {slot!r}")
    try:
        layouts = sorted(set(report["rule"]) & set(report["learned"]))
        sign = 1.0 if report.get("higher_is_better", True) else -1.0
        gains = [sign * (float(report["learned"][k]) - float(report["rule"][k])) for k in layouts]
    except (KeyError, TypeError, ValueError) as err:
        raise GateError("the gate report is not readable") from err
    if len(layouts) < MIN_LAYOUTS:
        raise GateError(f"needs {MIN_LAYOUTS} held-out layouts, the report has {len(layouts)}")
    if min(gains) < 0.0:
        raise GateError("the learned part is worse than the rule on at least one held-out layout")
    if sum(gains) / len(gains) < MARGIN:
        raise GateError(f"the learned part must beat the rule by at least {MARGIN} on average")


@dataclass
class Slot:
    name: str
    learned: Callable[..., Any] | None = None  # learned(*args) -> (answer, confidence)
    report: dict | None = None  # the passed gate report (None: not promoted)
    enabled: bool = False  # off by default
    used: dict[str, int] = field(default_factory=lambda: {"rule": 0, "learned": 0, "unsure": 0, "error": 0, "invalid": 0})

    @property
    def promoted(self) -> bool:
        return self.report is not None

    @property
    def active(self) -> bool:
        return self.enabled and self.promoted and self.learned is not None


class SlotRegistry:
    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        self.min_confidence = min_confidence
        self.slots = {name: Slot(name) for name in SLOTS}

    def _slot(self, name: str) -> Slot:
        if name not in self.slots:
            raise KeyError(f"unknown socket {name!r}")
        return self.slots[name]

    # ---- plugging a part in
    def attach(self, name: str, learned: Callable[..., Any]) -> None:
        self._slot(name).learned = learned

    def promote(self, name: str, report: dict) -> None:
        """Record a passed gate report. The report is re-checked here: a report that says 'passed' by itself is not trusted."""
        check_report(name, report)
        self._slot(name).report = report

    def enable(self, name: str, on: bool = True) -> None:
        slot = self._slot(name)
        if on and not slot.promoted:
            raise GateError(f"{name} has no passed gate report, so it cannot be switched on")
        slot.enabled = on

    def disable_all(self) -> None:
        for slot in self.slots.values():
            slot.enabled = False

    # ---- using it
    def decide(self, name: str, rule_answer: Any, args: tuple = (), valid: Callable[[Any], bool] | None = None) -> Any:
        """The answer to use: the learned part's if the socket is active, it is sure, and the answer makes sense;
        otherwise `rule_answer` (computed by the caller first, so the rule's own memory keeps learning either way)."""
        slot = self._slot(name)
        if not slot.active:
            slot.used["rule"] += 1
            return rule_answer
        try:
            answer, confidence = slot.learned(*args)
        except Exception:  # a broken learned part must never stop the game
            slot.used["error"] += 1
            return rule_answer
        if not isinstance(confidence, (int, float)) or confidence != confidence or confidence < self.min_confidence:
            slot.used["unsure"] += 1
            return rule_answer
        if answer is None or (valid is not None and not valid(answer)):
            slot.used["invalid"] += 1
            return rule_answer
        slot.used["learned"] += 1
        return answer

    def usage(self) -> dict[str, dict[str, int]]:
        return {name: dict(slot.used) for name, slot in self.slots.items()}

    # ---- remembering which parts passed (the learned functions themselves are re-attached by whoever loads the model)
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "slots": {n: {"report": s.report, "enabled": s.enabled} for n, s in self.slots.items()}},
                                   indent=2), encoding="utf-8")
        return path

    def load(self, path: Path | str) -> None:
        """Read back saved gate reports (each is re-checked). A bad or missing file leaves everything off."""
        try:
            saved = json.loads(Path(path).read_text(encoding="utf-8"))["slots"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        for name, entry in saved.items():
            if name not in self.slots or not isinstance(entry, dict) or not entry.get("report"):
                continue
            try:
                self.promote(name, entry["report"])
            except GateError:
                continue
            self.slots[name].enabled = bool(entry.get("enabled"))
