"""Stage 调度：start/stop、skip、run_state 同步。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from src.tasks.task1.io_utils import STAGES, RunState

logger = logging.getLogger(__name__)


@dataclass
class RunOptions:
    mode: str = "full"
    resume: bool = False
    start_stage: Optional[str] = None
    stop_after_stage: Optional[str] = None
    run_id: str = "task1"
    skip_optuna: bool = False
    skip_multiseed: bool = False
    skip_shap: bool = False
    audit_only: bool = False
    # 审计认定已完成、可跳过的 stage
    completed_stages: Set[str] = field(default_factory=set)


def stage_index(name: str) -> int:
    if name not in STAGES:
        raise ValueError(f"未知 stage: {name}; 可选={STAGES}")
    return STAGES.index(name)


class StageGate:
    def __init__(self, opts: RunOptions, state: RunState):
        self.opts = opts
        self.state = state
        self._start_i = stage_index(opts.start_stage) if opts.start_stage else 0
        self._stop_i = stage_index(opts.stop_after_stage) if opts.stop_after_stage else len(STAGES) - 1
        if self._stop_i < self._start_i:
            raise ValueError("stop-after-stage 不能早于 start-stage")

    def should_execute(self, stage: str) -> bool:
        i = stage_index(stage)
        if i < self._start_i or i > self._stop_i:
            return False
        if stage in {"catboost_optuna", "catboost_confirmation", "lightgbm_optuna"} and self.opts.skip_optuna:
            logger.info("skip stage=%s (--skip-optuna)", stage)
            return False
        if stage == "multiseed" and self.opts.skip_multiseed:
            logger.info("skip stage=%s (--skip-multiseed)", stage)
            return False
        if stage == "shap" and self.opts.skip_shap:
            logger.info("skip stage=%s (--skip-shap)", stage)
            return False
        if self.opts.resume and stage in self.opts.completed_stages:
            # 已完成且非 OOF 缺口阶段：跳过执行
            if stage in {
                "split",
                "baseline",
                "ablation",
                "feature_selection",
                "catboost_optuna",
                "catboost_confirmation",
                "lightgbm_optuna",
            }:
                logger.info("skip stage=%s (completed artifact from this full)", stage)
                return False
        return True

    def begin(self, stage: str, expected=None, conf_hash: str = "") -> None:
        logger.info(">>> STAGE BEGIN: %s", stage)
        self.state.begin(stage, expected_artifacts=expected, configuration_hash=conf_hash)

    def complete(self, stage: str, **extra) -> None:
        logger.info("<<< STAGE COMPLETE: %s", stage)
        self.state.complete(stage, **extra)

    def fail(self, stage: str, err: str) -> None:
        logger.error("!!! STAGE FAILED: %s | %s", stage, err)
        self.state.fail(stage, err)

    def partial(self, stage: str, error: str = "", **extra) -> None:
        logger.warning("~~~ STAGE PARTIAL: %s | %s", stage, error)
        self.state.partial(stage, error=error, **extra)

    def past_stop(self, stage: str) -> bool:
        return stage_index(stage) >= self._stop_i and not self.should_execute(
            STAGES[min(stage_index(stage) + 1, len(STAGES) - 1)]
        )


def completed_from_audit(audit: Dict) -> Set[str]:
    out: Set[str] = set()
    for name, info in (audit.get("stages") or {}).items():
        if info.get("status") == "completed":
            out.add(name)
    return out
