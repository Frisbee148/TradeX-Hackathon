"""Deterministic OpenEnv environment for AMM market surveillance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..baseline_policy import choose_surveillance_action
    from ..models import SurveillanceAction, SurveillanceObservation
    from ..tasks import compute_task_grade, list_task_names, task_definition
except ImportError:
    from baseline_policy import choose_surveillance_action
    from models import SurveillanceAction, SurveillanceObservation
    from tasks import compute_task_grade, list_task_names, task_definition

VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK", "MONITOR"}


class MarketSurveillanceEnvironment(Environment[SurveillanceAction, SurveillanceObservation, State]):
    """AMM-style market simulation focused on bot-aware surveillance decisions."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "burst_detection", transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self._task_name = task if task in list_task_names() else "burst_detection"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = task_definition(self._task_name)
        self._step_num = 0
        self._done = False
        self._last_reward = 0.0
        self._last_action_error: Optional[str] = None
        self._actions: List[str] = []
        self._rewards: List[float] = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> SurveillanceObservation:
        task = kwargs.get("task")
        if task in list_task_names():
            self._task_name = task
        self._task = task_definition(self._task_name)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._step_num = 0
        self._done = False
        self._last_reward = 0.0
        self._last_action_error = None
        self._actions = []
        self._rewards = []
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SurveillanceAction, timeout_s: Optional[float] = None, **kwargs: Any) -> SurveillanceObservation:
        if self._done:
            return self._build_observation(reward=0.0, done=True)

        action_type = action.action_type.strip().upper()
        if action_type not in VALID_ACTIONS:
            action_type = "MONITOR"
            self._last_action_error = "Invalid action supplied; MONITOR executed."
        else:
            self._last_action_error = None

        step_data = self._task.steps[self._step_num]
        reward = self._reward_for_action(action_type, step_data)

        self._actions.append(action_type)
        self._rewards.append(reward)
        self._last_reward = reward
        self._step_num += 1
        self._state.step_count = self._step_num
        self._done = self._step_num >= len(self._task.steps)

        return self._build_observation(reward=reward, done=self._done)

    def grade(self) -> Dict[str, Any]:
        grade = compute_task_grade(self._task_name, self._actions)
        return {
            "task": self._task_name,
            "title": self._task.title,
            "score": grade["score"],
            "detection_score": grade["detection_score"],
            "false_positive_score": grade["false_positive_score"],
            "false_negative_score": grade["false_negative_score"],
            "health_score": grade["health_score"],
            "overblocking_score": grade["overblocking_score"],
            "steps_run": len(self._actions),
            "baseline_last_action": choose_surveillance_action(self._build_observation(0.0, self._done)),
        }

    def _reward_for_action(self, action_type: str, step_data) -> float:
        severity = step_data.severity
        health = step_data.healthy_market_index
        if step_data.label == "suspicious":
            if action_type == "BLOCK":
                return round(1.0 + 0.6 * severity, 4)
            if action_type == "FLAG":
                return round(0.75 + 0.45 * severity, 4)
            if action_type == "MONITOR":
                return round(0.35 + 0.30 * severity, 4)
            return round(-1.0 - 0.7 * severity, 4)
        if action_type == "ALLOW":
            return round(0.65 + 0.20 * health, 4)
        if action_type == "MONITOR":
            return round(0.10 + 0.10 * health - 0.20 * (1.0 - health), 4)
        if action_type == "FLAG":
            return round(-0.35 - 0.30 * health, 4)
        return round(-0.80 - 0.45 * health, 4)

    def _current_step_data(self):
        index = min(self._step_num, len(self._task.steps) - 1)
        return self._task.steps[index]

    def _build_observation(self, reward: float, done: bool) -> SurveillanceObservation:
        step_data = self._current_step_data()
        trade_count = sum(1 for value in step_data.trades_in_window if value > 0)
        avg_trade_size = sum(step_data.trades_in_window) / max(1, trade_count)
        max_trade_size = max(step_data.trades_in_window) if step_data.trades_in_window else 0.0
        avg_gap = sum(step_data.recent_time_gaps) / max(1, len(step_data.recent_time_gaps))
        min_gap = min(step_data.recent_time_gaps) if step_data.recent_time_gaps else 0.0
        avg_impact = sum(step_data.recent_price_impacts) / max(1, len(step_data.recent_price_impacts))
        observation = SurveillanceObservation(
            current_amm_price=step_data.current_amm_price,
            liquidity_snapshot=step_data.liquidity_snapshot,
            recent_trade_count=trade_count,
            trades_in_window=step_data.trades_in_window,
            trade_frequency=round(trade_count / max(avg_gap, 0.1), 4),
            average_trade_size=round(avg_trade_size, 4),
            maximum_trade_size=round(max_trade_size, 4),
            recent_slippage_impact=round(avg_impact, 4),
            time_gap_mean=round(avg_gap, 4),
            time_gap_min=round(min_gap, 4),
            recent_time_gaps=step_data.recent_time_gaps,
            recent_price_impacts=step_data.recent_price_impacts,
            burst_indicator=step_data.burst_indicator,
            pattern_indicator=step_data.pattern_indicator,
            suspiciousness_score=step_data.suspiciousness_score,
            manipulation_score=step_data.manipulation_score,
            step_num=self._step_num,
            max_steps=len(self._task.steps),
            task_name=self._task_name,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "available_actions": sorted(VALID_ACTIONS),
                "available_tasks": list_task_names(),
                "last_action_error": self._last_action_error,
                "scenario_note": step_data.note,
            },
        )
        return self._apply_transform(observation)

    @property
    def state(self) -> State:
        return self._state


MeverseEnvironment = MarketSurveillanceEnvironment
