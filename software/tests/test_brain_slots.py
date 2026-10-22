"""Plan 19 T46: the four sockets for learned parts. Off by default; used only after a passed gate; the rule is the floor."""
from __future__ import annotations

import json

import pytest

from brain import slots as slots_module
from brain.builtin_nav import BuiltinNavBrain
from brain.slots import GateError, SlotRegistry, gate_report
from test_builtin_nav import CARD, play

LAYOUTS = ("a", "b", "c")
GOOD = gate_report("S3_policy", "keys_to_win", {"a": 0.5, "b": 0.5, "c": 0.5}, {"a": 0.7, "b": 0.6, "c": 0.7})


def sure(answer, confidence=0.99):
    return lambda *args: (answer, confidence)


def boom(*args):
    raise RuntimeError("boom")


def ready(name="S3_policy", learned=None, report=None) -> SlotRegistry:
    registry = SlotRegistry()
    registry.attach(name, learned or sure("up"))
    registry.promote(name, report or GOOD)
    registry.enable(name)
    return registry


# ---- the gate ----------------------------------------------------------------------------------------------------
def test_a_good_report_passes_and_says_so():
    assert GOOD["passed"] and GOOD["layouts"] == list(LAYOUTS) and GOOD["mean_gain"] == pytest.approx(1 / 6)


def test_lower_is_better_metrics_flip_the_sign():
    report = gate_report("S3_policy", "steps_share", {k: 0.9 for k in LAYOUTS}, {k: 0.7 for k in LAYOUTS}, higher_is_better=False)
    assert report["passed"] and report["mean_gain"] == pytest.approx(0.2)
    worse = gate_report("S3_policy", "steps_share", {k: 0.7 for k in LAYOUTS}, {k: 0.9 for k in LAYOUTS}, higher_is_better=False)
    assert not worse["passed"]


@pytest.mark.parametrize("rule,learned,why", [
    ({"a": .5, "b": .5}, {"a": .9, "b": .9}, "needs 3"),  # too few layouts
    ({"a": .5, "b": .5, "c": .5}, {"a": .9, "b": .9, "c": .4}, "worse"),  # worse on one layout
    ({"a": .5, "b": .5, "c": .5}, {"a": .51, "b": .52, "c": .5}, "at least"),  # a gain too small
])
def test_a_report_that_does_not_justify_the_part_is_refused(rule, learned, why):
    report = gate_report("S3_policy", "x", rule, learned)
    assert not report["passed"]
    with pytest.raises(GateError, match=why):
        SlotRegistry().promote("S3_policy", report)


def test_a_report_that_lies_about_passing_is_not_trusted():
    report = dict(gate_report("S3_policy", "x", {"a": .5, "b": .5, "c": .5}, {"a": .5, "b": .5, "c": .5}), passed=True)
    with pytest.raises(GateError):
        SlotRegistry().promote("S3_policy", report)


def test_a_report_for_another_socket_or_unreadable_is_refused():
    with pytest.raises(GateError, match="not 'S4_value'"):
        SlotRegistry().promote("S4_value", GOOD)
    with pytest.raises(GateError, match="readable"):
        SlotRegistry().promote("S3_policy", {"slot": "S3_policy", "rule": {"a": "x"}, "learned": {"a": 1}})
    with pytest.raises(GateError, match="unknown socket"):
        gate_report("S9", "x", {}, {})


# ---- the sockets -------------------------------------------------------------------------------------------------
def test_every_socket_starts_off_and_cannot_be_switched_on_without_a_gate():
    registry = SlotRegistry()
    assert not any(slot.enabled or slot.promoted or slot.active for slot in registry.slots.values())
    registry.attach("S3_policy", sure("up"))
    with pytest.raises(GateError, match="no passed gate"):
        registry.enable("S3_policy")
    assert registry.decide("S3_policy", "left") == "left"


def test_a_promoted_enabled_sure_part_is_used_and_counted():
    registry = ready()
    assert registry.decide("S3_policy", "left") == "up"
    assert registry.usage()["S3_policy"]["learned"] == 1


def test_switched_off_means_the_rule_even_after_a_pass():
    registry = ready()
    registry.enable("S3_policy", False)
    assert registry.decide("S3_policy", "left") == "left" and registry.usage()["S3_policy"]["rule"] == 1


def test_no_part_attached_means_the_rule():
    registry = SlotRegistry()
    registry.promote("S3_policy", GOOD)
    registry.enable("S3_policy")
    assert registry.decide("S3_policy", "left") == "left"


@pytest.mark.parametrize("learned,counter", [
    (sure("up", 0.5), "unsure"), (sure("up", float("nan")), "unsure"), (sure("up", "high"), "unsure"),
    (sure(None), "invalid"), (sure("sideways"), "invalid"),
    (boom, "error"), (lambda *a: "not a pair", "error")])
def test_a_part_that_is_unsure_wrong_or_broken_falls_back_to_the_rule(learned, counter):
    registry = ready(learned=learned)
    assert registry.decide("S3_policy", "left", valid=lambda k: k in ("left", "right", "up")) == "left"
    assert registry.usage()["S3_policy"][counter] == 1


def test_the_fallback_is_the_same_every_time():
    registry = ready(learned=boom)
    assert {registry.decide("S3_policy", "left") for _ in range(20)} == {"left"}


def test_saved_reports_come_back_checked_and_start_as_saved(tmp_path):
    path = ready().save(tmp_path / "slots.json")
    again = SlotRegistry()
    again.load(path)
    slot = again.slots["S3_policy"]
    assert slot.promoted and slot.enabled and not slot.active  # no part attached yet
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["slots"]["S3_policy"]["report"]["learned"] = {"a": 0.0, "b": 0.0, "c": 0.0}  # someone edits the numbers
    path.write_text(json.dumps(saved), encoding="utf-8")
    tampered = SlotRegistry()
    tampered.load(path)
    assert not tampered.slots["S3_policy"].promoted


def test_a_missing_or_broken_file_leaves_everything_off(tmp_path):
    registry = SlotRegistry()
    registry.load(tmp_path / "nothing.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    registry.load(tmp_path / "bad.json")
    assert not any(slot.promoted for slot in registry.slots.values())


def test_the_honest_wiring_table():
    assert slots_module.WIRED == {"S1_see": False, "S2_hud": True, "S3_policy": True, "S4_value": True}


# ---- inside the built-in brain -----------------------------------------------------------------------------------
def turned_on(brain, name, learned, **report_changes):
    brain.slots.attach(name, learned)
    brain.slots.promote(name, dict(GOOD, slot=name, **report_changes))
    brain.slots.enable(name)


def test_with_every_socket_off_the_brain_plays_exactly_as_before():
    brain = BuiltinNavBrain(CARD)
    world, _ = play(brain)
    assert world.game.won
    usage = brain.slots.usage()
    assert usage["S3_policy"]["learned"] == 0 and usage["S3_policy"]["rule"] > 0 and usage["S2_hud"]["rule"] > 0


def test_an_unpromoted_part_is_ignored_even_if_attached():
    brain = BuiltinNavBrain(CARD)
    brain.slots.attach("S3_policy", sure("up"))
    world, _ = play(brain)
    assert world.game.won and brain.slots.usage()["S3_policy"]["learned"] == 0


def test_a_promoted_policy_part_chooses_the_keys():
    brain = BuiltinNavBrain(CARD)
    turned_on(brain, "S3_policy", sure("UP"))
    _, replies = play(brain, max_steps=12)
    pressed = {step.action for _, reply in replies for step in reply.steps}
    assert pressed == {"UP"} and brain.slots.usage()["S3_policy"]["learned"] > 0


def test_a_part_that_names_a_key_not_on_the_card_is_ignored():
    brain = BuiltinNavBrain(CARD)
    turned_on(brain, "S3_policy", sure("teleport"))
    world, _ = play(brain)
    usage = brain.slots.usage()["S3_policy"]
    assert world.game.won and usage["invalid"] > 0 and usage["learned"] == 0


def test_a_broken_policy_part_still_lets_the_buddy_win_the_arena():
    brain = BuiltinNavBrain(CARD)
    turned_on(brain, "S3_policy", boom)
    world, _ = play(brain)
    assert world.game.won and brain.slots.usage()["S3_policy"]["error"] > 0


def test_the_hud_socket_is_asked_for_every_picture_and_its_nonsense_is_dropped():
    brain = BuiltinNavBrain(CARD)
    turned_on(brain, "S2_hud", sure("not a dict"))
    world, _ = play(brain)
    assert world.game.won and brain.slots.usage()["S2_hud"]["invalid"] > 0


def test_the_value_socket_replaces_the_practice_values_only_when_active():
    brain = BuiltinNavBrain(CARD)
    play(brain, max_steps=40)
    planner = brain.navigator.planner
    planner._valued()
    assert brain.slots.usage()["S4_value"]["rule"] >= 1
    turned_on(brain, "S4_value", sure({"UP": 9.0, "DOWN": 0.0}))
    advice = planner._valued()
    assert advice is not None and advice.actions == ["UP"] and advice.kind == "value"
    assert brain.slots.usage()["S4_value"]["learned"] == 1
