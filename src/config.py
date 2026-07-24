"""项目配置：任务定义、特征列表、模型参数与评分权重。"""

from __future__ import annotations

from typing import Any, Dict, List

# 随机种子，保证可复现
RANDOM_SEED: int = 42

# 默认验证集比例
DEFAULT_TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# 任务一：电动汽车采用意愿分类（三分类）
# ---------------------------------------------------------------------------
TASK1_NAME: str = "task1"
TASK1_TARGET: str = "ev_adoption_likelihood"
TASK1_FEATURES: List[str] = [
    "annual_income",
    "education_level",
    "environmental_awareness_score",
    "technology_affinity_score",
    "range_anxiety_score",
    "fuel_expense_per_month",
]
TASK1_LABEL_ORDER: List[str] = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# 任务二：家庭充电条件预测（二分类）
# ---------------------------------------------------------------------------
TASK2_NAME: str = "task2"
TASK2_TARGET: str = "home_charging_available"
TASK2_FEATURES: List[str] = [
    "city_type",
    "nearest_charging_station_km",
    "education_level",
    "annual_income",
    "vehicle_age_years",
]
TASK2_LABEL_ORDER: List[str] = ["0", "1"]

# ---------------------------------------------------------------------------
# 任务三：里程焦虑敏感度识别（二分类，由 range_anxiety_score 构造）
# ---------------------------------------------------------------------------
TASK3_NAME: str = "task3"
TASK3_SOURCE_SCORE: str = "range_anxiety_score"
TASK3_THRESHOLD: float = 5.0  # > 5 记为 1，<= 5 记为 0
TASK3_FEATURES: List[str] = [
    "daily_commute_km",
    "weekly_travel_distance_km",
    "charging_station_accessibility",
    "ev_knowledge_score",
    "battery_replacement_concern",
]
TASK3_LABEL_ORDER: List[str] = ["0", "1"]

# 三个任务全部必需字段（用于启动前检查）
REQUIRED_COLUMNS: List[str] = sorted(
    set(
        TASK1_FEATURES
        + [TASK1_TARGET]
        + TASK2_FEATURES
        + [TASK2_TARGET]
        + TASK3_FEATURES
        + [TASK3_SOURCE_SCORE]
    )
)

# 题目中明确的类别特征（代码会同时结合 pandas dtype 识别）
KNOWN_CATEGORICAL_FEATURES: List[str] = [
    "education_level",
    "city_type",
]

# ---------------------------------------------------------------------------
# CatBoost 基础参数
# ---------------------------------------------------------------------------
CATBOOST_COMMON: Dict[str, Any] = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": 50,
    "allow_writing_files": False,
}

TASK1_MODEL_PARAMS: Dict[str, Any] = {
    **CATBOOST_COMMON,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
}

TASK2_MODEL_PARAMS: Dict[str, Any] = {
    **CATBOOST_COMMON,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
}

TASK3_MODEL_PARAMS: Dict[str, Any] = {
    **CATBOOST_COMMON,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
}

# 官方评分权重（内部验证加权分）
SCORE_WEIGHTS: Dict[str, float] = {
    "ACC1": 0.2,
    "ACC2": 0.4,
    "ACC3": 0.4,
}
