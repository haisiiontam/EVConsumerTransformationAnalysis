"""数据读取、字段检查、概览统计与输出目录管理。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from src.config import (
    REQUIRED_COLUMNS,
    TASK1_TARGET,
    TASK2_TARGET,
    TASK3_SOURCE_SCORE,
    TASK3_THRESHOLD,
)

logger = logging.getLogger(__name__)


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """创建输出根目录及三个任务子目录，返回路径映射。"""
    output_dir = Path(output_dir)
    task_dirs = {
        "root": output_dir,
        "task1": output_dir / "task1",
        "task2": output_dir / "task2",
        "task3": output_dir / "task3",
    }
    for path in task_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录已准备：%s", output_dir.resolve())
    return task_dirs


def read_csv_compatible(data_path: Path) -> pd.DataFrame:
    """
    兼容多种编码读取 CSV。

    顺序：UTF-8 -> utf-8-sig -> gbk。
    读取后清理列名前后空格。
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{data_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding)
            # 清理列名前后空格，不擅自改成中文
            df.columns = [str(col).strip() for col in df.columns]
            logger.info(
                "成功使用编码 %s 读取数据：行数=%d，列数=%d",
                encoding,
                df.shape[0],
                df.shape[1],
            )
            logger.info("全部字段名：%s", list(df.columns))
            return df
        except Exception as exc:  # noqa: BLE001 - 需要尝试下一编码
            last_error = exc
            logger.warning("使用编码 %s 读取失败：%s", encoding, exc)

    raise RuntimeError(
        f"无法读取 CSV 文件 {data_path}，已尝试编码 {encodings}。"
        f"最后错误：{last_error}"
    )


def check_required_columns(df: pd.DataFrame, required: Sequence[str] | None = None) -> None:
    """检查三个任务所需字段是否存在；缺失则明确报错并终止。"""
    required_cols = list(required) if required is not None else REQUIRED_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "数据缺少必需字段，程序终止。缺失字段："
            + ", ".join(missing)
        )
    logger.info("必需字段检查通过，共检查 %d 个字段。", len(required_cols))


def build_data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """构建数据概览：行数、列数、重复行数、各列 dtype。"""
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())
    overview = pd.DataFrame(
        [
            {"item": "n_rows", "value": n_rows},
            {"item": "n_cols", "value": n_cols},
            {"item": "n_duplicate_rows", "value": n_duplicates},
        ]
    )
    dtype_rows = pd.DataFrame(
        {
            "item": [f"dtype::{col}" for col in df.columns],
            "value": [str(dtype) for dtype in df.dtypes],
        }
    )
    overview = pd.concat([overview, dtype_rows], ignore_index=True)
    logger.info(
        "数据概览：行数=%d，列数=%d，重复行=%d（第一版仅报告，不删除）",
        n_rows,
        n_cols,
        n_duplicates,
    )
    return overview


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """构建每列缺失数量与缺失比例报告。"""
    n_rows = len(df)
    missing_count = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_ratio": (missing_count.values / n_rows) if n_rows > 0 else 0.0,
            "dtype": [str(df[col].dtype) for col in missing_count.index],
        }
    )
    report = report.sort_values("missing_count", ascending=False).reset_index(drop=True)
    return report


def build_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计三个目标的类别分布，以及任务三转换后的 0/1 标签分布。
    """
    rows: List[Dict[str, object]] = []

    # 任务一：清理标签空格与大小写后统计，输出时仍用标准 Low/Medium/High
    if TASK1_TARGET in df.columns:
        cleaned = (
            df[TASK1_TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"low": "Low", "medium": "Medium", "high": "High"})
        )
        counts = cleaned.value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "task": "task1",
                    "target": TASK1_TARGET,
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / len(df) if len(df) else 0.0,
                }
            )

    # 任务二
    if TASK2_TARGET in df.columns:
        counts = df[TASK2_TARGET].value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "task": "task2",
                    "target": TASK2_TARGET,
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / len(df) if len(df) else 0.0,
                }
            )

    # 任务三原始评分
    if TASK3_SOURCE_SCORE in df.columns:
        score = pd.to_numeric(df[TASK3_SOURCE_SCORE], errors="coerce")
        # 原始评分分布（按唯一值汇总，便于检查）
        raw_counts = score.value_counts(dropna=False).sort_index()
        for label, count in raw_counts.items():
            rows.append(
                {
                    "task": "task3_raw_score",
                    "target": TASK3_SOURCE_SCORE,
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / len(df) if len(df) else 0.0,
                }
            )

        # 固定阈值 5 构造二分类标签
        binary = (score > TASK3_THRESHOLD).astype("Int64")
        bin_counts = binary.value_counts(dropna=False).sort_index()
        for label, count in bin_counts.items():
            rows.append(
                {
                    "task": "task3_binary",
                    "target": f"{TASK3_SOURCE_SCORE} > {TASK3_THRESHOLD}",
                    "label": label,
                    "count": int(count),
                    "ratio": float(count) / len(df) if len(df) else 0.0,
                }
            )

    return pd.DataFrame(rows)


def save_data_checks(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """保存数据概览、缺失报告与目标分布。"""
    output_dir = Path(output_dir)
    overview = build_data_overview(df)
    missing = build_missing_report(df)
    targets = build_target_distribution(df)

    overview.to_csv(output_dir / "data_overview.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output_dir / "missing_report.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output_dir / "target_distribution.csv", index=False, encoding="utf-8-sig")
    logger.info("已保存 data_overview / missing_report / target_distribution。")
