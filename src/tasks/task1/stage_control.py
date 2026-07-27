"""Stage 调度：任务一专用 STAGES / skip 策略。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from src.common.io_utils import RunState
from src.common.stage_control import StageGate as _StageGate
from src.common.stage_control import completed_from_audit  # noqa: F401
from src.tasks.task1.io_utils import STAGES

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
    completed_stages: Set[str] = field(default_factory=set)


def stage_index(name: str) -> int:
    if name not in STAGES:
        raise ValueError(f"未知 stage: {name}; 可选={STAGES}")
    return STAGES.index(name)


def _task1_skip(stage: str, opts: RunOptions) -> bool:
    if stage in {"catboost_optuna", "catboost_confirmation", "lightgbm_optuna"} and opts.skip_optuna:
        logger.info("skip stage=%s (--skip-optuna)", stage)
        return True
    if stage == "multiseed" and opts.skip_multiseed:
        logger.info("skip stage=%s (--skip-multiseed)", stage)
        return True
    if stage == "shap" and opts.skip_shap:
        logger.info("skip stage=%s (--skip-shap)", stage)
        return True
    return False


_RESUME_SKIP = {
    "split",
    "baseline",
    "ablation",
    "feature_selection",
    "catboost_optuna",
    "catboost_confirmation",
    "lightgbm_optuna",
}


class StageGate:
    def __init__(self, opts: RunOptions, state: RunState):
        # 适配 common StageGate（opts 字段兼容）
        from src.common.stage_control import RunOptions as CommonOpts

        common_opts = CommonOpts(
            mode=opts.mode,
            resume=opts.resume,
            start_stage=opts.start_stage,
            stop_after_stage=opts.stop_after_stage,
            run_id=opts.run_id,
            skip_optuna=opts.skip_optuna,
            skip_multiseed=opts.skip_multiseed,
            skip_shap=opts.skip_shap,
            audit_only=opts.audit_only,
            completed_stages=opts.completed_stages,
        )
        self.opts = opts
        self.state = state
        self._gate = _StageGate(
            common_opts,
            state,
            STAGES,
            skip_checker=lambda s, o: _task1_skip(s, opts),
            resume_skip_stages=_RESUME_SKIP,
        )

    def should_execute(self, stage: str) -> bool:
        return self._gate.should_execute(stage)

    def begin(self, stage: str, expected=None, conf_hash: str = "") -> None:
        self._gate.begin(stage, expected=expected, conf_hash=conf_hash)

    def complete(self, stage: str, **extra) -> None:
        self._gate.complete(stage, **extra)

    def fail(self, stage: str, err: str) -> None:
        self._gate.fail(stage, err)

    def partial(self, stage: str, error: str = "", **extra) -> None:
        self._gate.partial(stage, error=error, **extra)

    def past_stop(self, stage: str) -> bool:
        i = stage_index(stage)
        return i >= self._gate._stop_i and not self.should_execute(
            STAGES[min(i + 1, len(STAGES) - 1)]
        )
