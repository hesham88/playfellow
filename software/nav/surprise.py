"""Curiosity and Intrinsic Motivation: forward-dynamics prediction error and empowerment.
/ الفضول والتحفيز الداخلي: خطأ التنبؤ بالديناميكيات والتمكين."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence


def _hashable(s: Any) -> Any:
    """Recursively converts lists or dicts into hashable tuples for state dictionary keys."""
    if isinstance(s, list):
        return tuple(_hashable(x) for x in s)
    elif isinstance(s, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in s.items()))
    elif isinstance(s, set):
        return tuple(sorted(_hashable(x) for x in s))
    try:
        hash(s)
        return s
    except TypeError:
        if hasattr(s, "size") and hasattr(s, "mode"):
            return (s.size, s.mode)
        if hasattr(s, "shape"):
            return (s.shape, str(getattr(s, "dtype", "")))
        return str(s)


def _normalize_entity_map(entities: Any) -> dict[str, Any]:
    """Normalizes various entity representations into a flat mapping of key -> state/coord."""
    if not entities:
        return {}
    if isinstance(entities, dict):
        normalized: dict[str, Any] = {}
        for k, v in entities.items():
            if isinstance(v, (set, list, tuple)):
                items = list(v)
                if items and isinstance(items[0], (tuple, list)):
                    for i, coord in enumerate(sorted(items, key=lambda c: tuple(c))):
                        normalized[f"{k}_{i}"] = tuple(coord)
                elif items and isinstance(items[0], dict):
                    for i, item in enumerate(items):
                        item_id = item.get("id", f"{k}_{i}")
                        normalized[str(item_id)] = item
                else:
                    normalized[k] = tuple(items) if items else ()
            elif isinstance(v, dict):
                normalized[k] = v
            else:
                normalized[k] = v
        return normalized
    elif isinstance(entities, (list, tuple, set)):
        normalized = {}
        for i, item in enumerate(entities):
            if isinstance(item, dict):
                item_id = item.get("id", f"entity_{i}")
                normalized[str(item_id)] = item
            elif isinstance(item, (tuple, list)):
                normalized[f"entity_{i}"] = tuple(item)
            else:
                normalized[f"entity_{i}"] = item
        return normalized
    return {"entity_0": entities}


def compute_entity_diff(
    entities_before: Any,
    entities_after: Any,
) -> dict[str, Any]:
    """Computes entity difference between two steps for VTX telemetry / curiosity.
    
    Identifies:
    - added: entities present in 'after' but not 'before'
    - removed: entities present in 'before' but not 'after'
    - moved: entities present in both whose position/state changed
    - unchanged: entities whose position/state stayed identical
    - diff_score: normalized magnitude of change in [0.0, 1.0]
    """
    before_map = _normalize_entity_map(entities_before)
    after_map = _normalize_entity_map(entities_after)

    all_keys = set(before_map.keys()) | set(after_map.keys())
    if not all_keys:
        return {
            "added": [],
            "removed": [],
            "moved": [],
            "unchanged": [],
            "diff_score": 0.0,
        }

    added = []
    removed = []
    moved = []
    unchanged = []

    for k in sorted(all_keys):
        in_before = k in before_map
        in_after = k in after_map
        if in_after and not in_before:
            added.append({"id": k, "state": after_map[k]})
        elif in_before and not in_after:
            removed.append({"id": k, "state": before_map[k]})
        else:
            s_before = before_map[k]
            s_after = after_map[k]
            if s_before != s_after:
                moved.append({"id": k, "before": s_before, "after": s_after})
            else:
                unchanged.append(k)

    changed_count = len(added) + len(removed) + len(moved)
    diff_score = float(min(1.0, max(0.0, changed_count / max(1, len(all_keys)))))

    return {
        "added": added,
        "removed": removed,
        "moved": moved,
        "unchanged": unchanged,
        "diff_score": diff_score,
    }


class SurpriseModule:
    """Computes forward-dynamics prediction error (curiosity) and empowerment
    (mutual information between actions and subsequent states), augmented with
    VTX entity difference dynamics.
    
    1. Forward Dynamics Prediction Error:
       The agent learns P(s' | s, a). Unseen or surprising transitions yield high
       prediction error E(s, a, s') in [0.0, 1.0], giving intrinsic curiosity reward.
       
    2. VTX Entity Diff Surprise:
       Unexpected entity appearances, removals, or state changes contribute to
       entity novelty in [0.0, 1.0], rewarding the agent for discovering dynamic interactions.

    3. Empowerment:
       Empowerment I(A; S') measures the degree of control the agent possesses in
       state s: the mutual information between the action choice and the resulting state.
       In an open room with 4 distinct moves, empowerment is high (~2 bits).
       In a dead end or when trapped by obstacles, empowerment drops towards 0.
    """

    def __init__(self, eta: float = 1.0, beta: float = 0.1, entity_weight: float = 0.5):
        self.eta = eta  # scaling for prediction error (curiosity)
        self.beta = beta  # scaling for empowerment reward
        self.entity_weight = entity_weight  # weight of entity diff in total surprise
        # transitions[state][action][next_state] = count
        self.transitions: dict[Any, dict[str, dict[Any, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # totals[state][action] = total transitions from (state, action)
        self.action_counts: dict[Any, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.state_visits: dict[Any, int] = defaultdict(int)
        # entity transitions history
        self.entity_diff_history: list[dict[str, Any]] = []

    def observe(
        self,
        state: Any,
        action: str,
        next_state: Any,
        entities_before: Any = None,
        entities_after: Any = None,
        entity_diff: dict[str, Any] | None = None,
    ) -> float:
        """Records transition (s, a, s'), updates model, and returns the surprise before update."""
        h_state = _hashable(state)
        h_next_state = _hashable(next_state)
        surprise = self.compute_surprise(
            h_state, action, h_next_state,
            entities_before=entities_before,
            entities_after=entities_after,
            entity_diff=entity_diff,
        )
        self.transitions[h_state][action][h_next_state] += 1
        self.action_counts[h_state][action] += 1
        self.state_visits[h_state] += 1
        return surprise

    def compute_surprise(
        self,
        state: Any,
        action: str,
        next_state: Any,
        entities_before: Any = None,
        entities_after: Any = None,
        entity_diff: dict[str, Any] | None = None,
    ) -> float:
        """Prediction error in [0.0, 1.0].
        
        If (s, a) was never observed before, error is 1.0 (maximal novelty).
        Otherwise, error is 1.0 - P(s' | s, a).
        When entities or entity_diff are provided, blends state surprise with entity diff surprise.
        """
        h_state = _hashable(state)
        h_next_state = _hashable(next_state)

        total = self.action_counts[h_state].get(action, 0)
        if total == 0:
            base_surprise = 1.0
        else:
            prob = self.transitions[h_state][action].get(h_next_state, 0) / total
            base_surprise = float(max(0.0, min(1.0, 1.0 - prob)))

        # VTX Entity diff integration
        if entities_before is not None and entities_after is not None:
            diff = compute_entity_diff(entities_before, entities_after)
            entity_score = diff["diff_score"]
            self.entity_diff_history.append(diff)
            return float(max(0.0, min(1.0, (1.0 - self.entity_weight) * base_surprise + self.entity_weight * entity_score)))
        elif entity_diff is not None:
            entity_score = entity_diff.get("diff_score", 0.0)
            self.entity_diff_history.append(entity_diff)
            return float(max(0.0, min(1.0, (1.0 - self.entity_weight) * base_surprise + self.entity_weight * entity_score)))

        return base_surprise

    def predict(self, state: Any, action: str) -> Any | None:
        """Returns the most likely next state given (state, action)."""
        h_state = _hashable(state)
        counts = self.transitions[h_state].get(action)
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    def compute_empowerment(self, state: Any, actions: Sequence[str] | None = None) -> float:
        """Empowerment I(A; S') = H(S') - H(S'|A) from state s.
        
        Assuming uniform distribution over available actions P(a) = 1 / |A|:
        - H(S'|A) = sum_a P(a) * H(S'|a)
        - H(S') = - sum_{s'} P(s') * log2(P(s')) where P(s') = sum_a P(a) * P(s'|a)
        
        Empowerment is measured in bits (>= 0.0). Higher empowerment means the agent
        can reach more distinct states deterministically.
        """
        h_state = _hashable(state)
        if actions is None:
            acts = list(self.action_counts[h_state].keys())
        else:
            acts = list(actions)

        if not acts:
            return 0.0

        p_a = 1.0 / len(acts)
        marginal_next: dict[Any, float] = defaultdict(float)
        conditional_entropy = 0.0

        for a in acts:
            total = self.action_counts[h_state].get(a, 0)
            if total == 0:
                p_s_given_a = {f"__unexplored_{a}__": 1.0}
            else:
                p_s_given_a = {ns: c / total for ns, c in self.transitions[h_state][a].items()}

            h_s_given_a = -sum(p * math.log2(p) for p in p_s_given_a.values() if p > 0)
            conditional_entropy += p_a * h_s_given_a

            for ns, p in p_s_given_a.items():
                marginal_next[ns] += p_a * p

        marginal_entropy = -sum(p * math.log2(p) for p in marginal_next.values() if p > 0)

        empowerment = marginal_entropy - conditional_entropy
        return float(max(0.0, empowerment))

    def intrinsic_reward(
        self,
        state: Any,
        action: str,
        next_state: Any,
        actions: Sequence[str] | None = None,
        entities_before: Any = None,
        entities_after: Any = None,
    ) -> float:
        """Combined intrinsic motivation reward:
        r_intrinsic = eta * surprise(s, a, s') + beta * empowerment(s')
        """
        surprise = self.compute_surprise(
            state, action, next_state,
            entities_before=entities_before,
            entities_after=entities_after,
        )
        empowerment = self.compute_empowerment(next_state, actions)
        return float(self.eta * surprise + self.beta * empowerment)

SurpriseCuriosity = SurpriseModule
