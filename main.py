"""主程序：电动汽车消费者转型分析三任务朴素基线。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_TEST_SIZE, SCORE_WEIGHTS
from src.data_utils import (
    check_required_columns,
    ensure_output_dirs,
    read_csv_compatible,
    save_data_checks,
)
from src.modeling import run_all_tasks


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
        description="电动汽车消费者转型分析三任务朴素基线（CatBoost）",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="原始 CSV 文件路径（必填）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="输出目录（默认 outputs）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="验证集比例（默认 0.2）",
    )
    return parser.parse_args(argv)


def compute_weighted_score(acc1: float, acc2: float, acc3: float) -> float:
    """
    计算内部验证加权分：
    weighted_score = 100 * (0.2 * ACC1 + 0.4 * ACC2 + 0.4 * ACC3)
    """
    return 100.0 * (
        SCORE_WEIGHTS["ACC1"] * acc1
        + SCORE_WEIGHTS["ACC2"] * acc2
        + SCORE_WEIGHTS["ACC3"] * acc3
    )


def save_overall_summary(
    accuracies: dict[str, float],
    output_dir: Path,
) -> float:
    """保存 overall_summary.csv，并返回 validation weighted score。"""
    weighted = compute_weighted_score(
        accuracies["ACC1"],
        accuracies["ACC2"],
        accuracies["ACC3"],
    )
    summary = pd.DataFrame(
        [
            {"metric": "ACC1", "value": accuracies["ACC1"]},
            {"metric": "ACC2", "value": accuracies["ACC2"]},
            {"metric": "ACC3", "value": accuracies["ACC3"]},
            {"metric": "weighted_score", "value": weighted},
            {"metric": "score_note", "value": "validation weighted score"},
        ]
    )
    out_path = Path(output_dir) / "overall_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    return weighted


def main(argv: list[str] | None = None) -> int:
    """串联数据检查与三个任务训练，输出内部验证总成绩。"""
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args(argv)

    if not 0.0 < args.test_size < 1.0:
        raise ValueError(f"--test-size 必须在 (0, 1) 之间，当前为 {args.test_size}")

    data_path = Path(args.data)
    output_dir = Path(args.output)

    logger.info("数据路径：%s", data_path)
    logger.info("输出目录：%s", output_dir)
    logger.info("验证集比例 test_size=%.4f", args.test_size)

    # 1. 准备输出目录
    output_dirs = ensure_output_dirs(output_dir)

    # 2. 读取数据
    df = read_csv_compatible(data_path)

    # 3. 必需字段检查
    check_required_columns(df)

    # 4. 数据基础检查并保存
    save_data_checks(df, output_dirs["root"])

    # 5. 依次训练三个任务
    accuracies = run_all_tasks(df, output_dirs, test_size=args.test_size)

    # 6. 汇总内部验证加权分（非最终比赛成绩）
    weighted = save_overall_summary(accuracies, output_dirs["root"])

    logger.info("=" * 60)
    logger.info("内部验证结果（validation only，非最终比赛成绩）")
    logger.info("ACC1 = %.6f", accuracies["ACC1"])
    logger.info("ACC2 = %.6f", accuracies["ACC2"])
    logger.info("ACC3 = %.6f", accuracies["ACC3"])
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
