from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from src.common.io_utils import RunState

logger = logging.getLogger(__name__)


@dataclass
class RunOptions:
    mode: str = "full"
    resume: bool = False
    start_stage: Optional[str] = None
    stop_after_stage: Optional[str] = None
    run_id: str = "run"
    skip_optuna: bool = False
    skip_multiseed: bool = False
    skip_shap: bool = False
    skip_deep: bool = False
    skip_robustness: bool = False
    audit_only: bool = False
    completed_stages: Set[str] = field(default_factory=set)


def stage_index(name: str, stages: Sequence[str]) -> int:
    if name not in stages:
        raise ValueError(f"未知 stage: {name}; 可选={list(stages)}")
    return list(stages).index(name)


class StageGate:
    """STAGES 与 skip 策略由调用方注入，避免任务间硬编码耦合。"""

    def __init__(
        self,
        opts: RunOptions,
        state: RunState,
        stages: Sequence[str],
        *,
        skip_checker: Optional[Callable[[str, RunOptions], bool]] = None,
        resume_skip_stages: Optional[Sequence[str]] = None,
    ):
        self.opts = opts
        self.state = state
        self.stages: List[str] = list(stages)
        self.skip_checker = skip_checker
        self.resume_skip_stages = set(resume_skip_stages or [])
        self._start_i = stage_index(opts.start_stage, self.stages) if opts.start_stage else 0
        self._stop_i = (
            stage_index(opts.stop_after_stage, self.stages)
            if opts.stop_after_stage
            else len(self.stages) - 1
        )
        if self._stop_i < self._start_i:
            raise ValueError("stop-after-stage 不能早于 start-stage")

    def should_execute(self, stage: str) -> bool:
        i = stage_index(stage, self.stages)
        if i < self._start_i or i > self._stop_i:
            return False
        if self.skip_checker and self.skip_checker(stage, self.opts):
            return False
        if self.opts.resume and stage in self.opts.completed_stages:
            if stage in self.resume_skip_stages:
                logger.info("skip stage=%s (completed artifact)", stage)
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


def completed_from_audit(audit: Dict) -> Set[str]:
    out: Set[str] = set()
    for name, info in (audit.get("stages") or {}).items():
        if info.get("status") == "completed":
            out.add(name)
    return out
