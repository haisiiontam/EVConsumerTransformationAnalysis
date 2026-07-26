"""主程序：电动汽车消费者转型分析三任务入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.common.data_utils import (
    check_required_columns,
    ensure_output_dirs,
    read_csv_compatible,
    save_data_checks,
)
from src.config import (
    DEFAULT_TEST_SIZE,
    SCORE_WEIGHTS,
    TASK2_ALPHA_STEP,
    TASK2_N_FOLDS,
    TASK2_THRESHOLD_MAX,
    TASK2_THRESHOLD_MIN,
    TASK2_THRESHOLD_STEP,
)
from src.tasks import task1, task2, task3


def setup_logging() -> None:
    """配置日志输出格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="电动汽车消费者转型分析（任务一/三基线 + 任务二优化）",
    )
    parser.add_argument("--data", type=str, required=True, help="有标签 CSV 路径（必填）")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录（默认 outputs）")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="最终验证集比例（默认 0.2）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "task1", "task2", "task3"],
        help="运行任务：all / task1 / task2 / task3",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["quick", "full"],
        help="任务一运行模式：quick（冒烟）/ full（论文全量，默认）",
    )
    parser.add_argument("--resume", action="store_true", help="任务一：断点续跑（冻结划分，跳过已完成阶段）")
    parser.add_argument("--start-stage", type=str, default=None, help="任务一：从指定 stage 开始")
    parser.add_argument("--stop-after-stage", type=str, default=None, help="任务一：在指定 stage 后停止")
    parser.add_argument("--run-id", type=str, default=None, help="任务一：运行 ID（日志/run_state）")
    parser.add_argument("--skip-optuna", action="store_true", help="任务一：跳过 Optuna 调参")
    parser.add_argument("--skip-multiseed", action="store_true", help="任务一：跳过多 seed 稳定性")
    parser.add_argument("--skip-shap", action="store_true", help="任务一：跳过 SHAP")
    parser.add_argument("--audit-only", action="store_true", help="任务一：仅审计输出文件，不训练")
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="可选独立测试集 CSV（仅任务二在最终验证后预测）",
    )
    parser.add_argument("--task2-folds", type=int, default=TASK2_N_FOLDS)
    parser.add_argument("--task2-alpha-step", type=float, default=TASK2_ALPHA_STEP)
    parser.add_argument("--task2-threshold-min", type=float, default=TASK2_THRESHOLD_MIN)
    parser.add_argument("--task2-threshold-max", type=float, default=TASK2_THRESHOLD_MAX)
    parser.add_argument("--task2-threshold-step", type=float, default=TASK2_THRESHOLD_STEP)
    return parser.parse_args(argv)


def compute_weighted_score(acc1: float, acc2: float, acc3: float) -> float:
    """内部验证加权分（非最终比赛成绩）。"""
    return 100.0 * (
        SCORE_WEIGHTS["ACC1"] * acc1
        + SCORE_WEIGHTS["ACC2"] * acc2
        + SCORE_WEIGHTS["ACC3"] * acc3
    )


def save_overall_summary(
    accuracies: Dict[str, Optional[float]],
    output_dir: Path,
) -> Optional[float]:
    """保存 overall_summary.csv。"""
    rows = []
    for key in ("ACC1", "ACC2", "ACC3"):
        rows.append({"metric": key, "value": accuracies.get(key)})

    weighted: Optional[float] = None
    if all(accuracies.get(k) is not None for k in ("ACC1", "ACC2", "ACC3")):
        weighted = compute_weighted_score(
            float(accuracies["ACC1"]),
            float(accuracies["ACC2"]),
            float(accuracies["ACC3"]),
        )
        rows.append({"metric": "weighted_score", "value": weighted})
        rows.append({"metric": "score_note", "value": "validation weighted score"})
    else:
        rows.append(
            {
                "metric": "score_note",
                "value": "partial run; weighted_score not computed",
            }
        )

    pd.DataFrame(rows).to_csv(
        Path(output_dir) / "overall_summary.csv", index=False, encoding="utf-8-sig"
    )
    return weighted


def main(argv: list[str] | None = None) -> int:
    """参数解析、调用所选任务并汇总成绩。"""
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args(argv)

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(f"--test-size 必须在 (0, 1) 之间，当前为 {args.test_size}")

    data_path = Path(args.data)
    output_dir = Path(args.output)
    logger.info("数据路径：%s", data_path)
    logger.info("输出目录：%s", output_dir)
    logger.info("运行任务：%s", args.task)

    output_dirs = ensure_output_dirs(output_dir)
    df = read_csv_compatible(data_path)
    check_required_columns(df)
    save_data_checks(df, output_dirs["root"])

    test_df = None
    if args.test:
        test_df = read_csv_compatible(Path(args.test))
        logger.info("已读取独立测试集：%s，行数=%d", args.test, len(test_df))

    accuracies: Dict[str, Optional[float]] = {
        "ACC1": None,
        "ACC2": None,
        "ACC3": None,
    }

    if args.task in ("all", "task1"):
        m1 = task1.run(
            df,
            output_dirs["task1"],
            test_size=args.test_size,
            mode=args.mode,
            resume=args.resume,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            run_id=args.run_id,
            skip_optuna=args.skip_optuna,
            skip_multiseed=args.skip_multiseed,
            skip_shap=args.skip_shap,
            audit_only=args.audit_only,
        )
        if args.audit_only or (isinstance(m1.get("accuracy"), float) and m1["accuracy"] != m1["accuracy"]):
            accuracies["ACC1"] = None
        else:
            accuracies["ACC1"] = float(m1["accuracy"])

    if args.task in ("all", "task2"):
        m2 = task2.run(
            df,
            output_dirs["task2"],
            test_size=args.test_size,
            n_folds=args.task2_folds,
            alpha_step=args.task2_alpha_step,
            threshold_min=args.task2_threshold_min,
            threshold_max=args.task2_threshold_max,
            threshold_step=args.task2_threshold_step,
            test_df=test_df,
        )
        accuracies["ACC2"] = float(m2["accuracy"])

    if args.task in ("all", "task3"):
        m3 = task3.run(df, output_dirs["task3"], test_size=args.test_size)
        accuracies["ACC3"] = float(m3["accuracy"])

    weighted = save_overall_summary(accuracies, output_dirs["root"])

    logger.info("=" * 60)
    logger.info("内部验证结果（validation only，非最终比赛成绩）")
    for key, val in accuracies.items():
        if val is not None:
            logger.info("%s = %.6f", key, val)
    if weighted is not None:
        logger.info("validation weighted score = %.4f", weighted)
    logger.info("全部输出已写入：%s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.error("程序异常终止：%s", exc)
        raise
