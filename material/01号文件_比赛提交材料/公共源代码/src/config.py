from __future__ import annotations

from typing import Any, Dict, List


RANDOM_SEED: int = 42


DEFAULT_TEST_SIZE: float = 0.2


TASK1_NAME: str = "task1"
TASK1_TARGET: str = "ev_adoption_likelihood"
TASK1_LABEL_ORDER: List[str] = ["Low", "Medium", "High"]


TASK1_FEATURES: List[str] = [
    "annual_income",
    "education_level",
    "environmental_awareness_score",
    "technology_affinity_score",
    "range_anxiety_score",
    "fuel_expense_per_month",
]
TASK1_OFFICIAL_FEATURES: List[str] = list(TASK1_FEATURES)


TASK1_STRONG_FEATURES: List[str] = [
    "ev_knowledge_score",
    "range_anxiety_score",
    "technology_affinity_score",
    "environmental_awareness_score",
    "annual_income",
]


TASK1_SAFE_CANDIDATE_FEATURES: List[str] = [
    "annual_income",
    "education_level",
    "environmental_awareness_score",
    "technology_affinity_score",
    "range_anxiety_score",
    "fuel_expense_per_month",
    "ev_knowledge_score",
    "battery_replacement_concern",
    "government_incentive_awareness",
    "charging_station_accessibility",
    "nearest_charging_station_km",
    "previous_ev_experience",
    "age",
    "city_type",
    "current_vehicle_type",
    "vehicle_age_years",
    "daily_commute_km",
    "weekly_travel_distance_km",
    "electricity_cost_per_kwh",
]

TASK1_EXCLUDED_FEATURES: List[str] = [
    "ev_adoption_likelihood",
    "home_charging_available",
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
]

TASK1_MECHANISM_FEATURES: List[str] = [
    "log_annual_income",
    "knowledge_deficit",
    "eco_tech_mean",
    "eco_minus_tech",
    "anxiety_knowledge_gap",
    "fuel_income_pressure",
    "commute_ratio",
]


TASK1_MECH_GROUPS: Dict[str, List[str]] = {
    "C1_economy": [
        "log_annual_income",
        "monthly_income",
        "fuel_income_pressure",
        "fuel_monthly_income_ratio",
    ],
    "C2_eco_tech": [
        "eco_tech_mean",
        "eco_tech_interaction",
        "eco_minus_tech",
    ],
    "C3_cognition": [
        "knowledge_tech_gap",
        "knowledge_tech_mean",
        "experience_tech_interaction",
        "knowledge_deficit",
    ],
    "C4_anxiety": [
        "anxiety_concern_interaction",
        "anxiety_knowledge_ratio",
        "concern_knowledge_ratio",
        "anxiety_knowledge_gap",
    ],
    "C5_travel": [
        "average_daily_travel",
        "travel_pressure",
        "commute_ratio",
    ],
    "C6_charging": [
        "charging_gap",
        "charging_convenience",
        "incentive_income_interaction",
    ],
}

TASK1_UNCERTAIN_FEATURES: List[str] = [
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
]

TASK1_KNOWN_CATEGORICAL: List[str] = [
    "education_level",
    "city_type",
    "current_vehicle_type",
    "previous_ev_experience",
]

TASK1_N_FOLDS: int = 5
TASK1_EARLY_STOPPING_ROUNDS: int = 80
TASK1_TIE_EPS: float = 0.0005
TASK1_STABILITY_SEEDS: List[int] = [42, 52, 62]
TASK1_OPTUNA_TRIALS_CB: int = 40
TASK1_OPTUNA_TRIALS_LGB: int = 25
TASK1_OPTUNA_COARSE_FOLDS: int = 3


TASK2_NAME: str = "task2"
TASK2_TARGET: str = "home_charging_available"
TASK2_LABEL_ORDER: List[str] = ["0", "1"]


TASK2_BASE_FEATURES: List[str] = [
    "city_type",
    "nearest_charging_station_km",
    "education_level",
    "annual_income",
    "vehicle_age_years",
]


TASK2_SAFE_CANDIDATE_FEATURES: List[str] = [
    "age",
    "annual_income",
    "education_level",
    "city_type",
    "daily_commute_km",
    "weekly_travel_distance_km",
    "current_vehicle_type",
    "vehicle_age_years",
    "fuel_expense_per_month",
    "charging_station_accessibility",
    "nearest_charging_station_km",
    "electricity_cost_per_kwh",
    "environmental_awareness_score",
    "government_incentive_awareness",
    "technology_affinity_score",
    "battery_replacement_concern",
    "ev_knowledge_score",
    "previous_ev_experience",
]


TASK2_EXCLUDED_FEATURES: List[str] = [
    "home_charging_available",
    "ev_adoption_likelihood",
    "range_anxiety_score",
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
]


TASK2_LEAK_RISK_PROBE_FEATURES: List[str] = [
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
    "ev_adoption_likelihood",
    "range_anxiety_score",
]


TASK2_MECHANISM_FEATURES: List[str] = [
    "charging_gap_index",
    "charging_convenience_index",
    "commute_ratio",
    "fuel_income_pressure",
    "replacement_pressure",

    "log_annual_income",
    "city_is_urban",
    "city_is_suburban",
    "city_is_rural",
    "income_x_urban",
    "income_x_suburban",
    "income_x_rural",
    "near_x_rural",
    "access_x_prev_ev",
    "income_over_near",
    "electricity_x_income",
    "noncommute_km",

    "income_ge_train_median",
    "urban_high_income",
    "suburban_high_income",
]


TASK2_WEAK_SIGNAL_AUC_GAP: float = 0.02


TASK2_DIAG_CATEGORICAL: List[str] = [
    "city_type",
    "education_level",
    "current_vehicle_type",
    "previous_ev_experience",
]


TASK2_KNOWN_CATEGORICAL: List[str] = [
    "education_level",
    "city_type",
    "current_vehicle_type",
    "previous_ev_experience",
]


TASK2_N_FOLDS: int = 5
TASK2_ALPHA_STEP: float = 0.05
TASK2_THRESHOLD_MIN: float = 0.30
TASK2_THRESHOLD_MAX: float = 0.80
TASK2_THRESHOLD_STEP: float = 0.005
TASK2_EARLY_STOPPING_ROUNDS: int = 80
TASK2_TIE_EPS: float = 0.0005
TASK2_STABILITY_SEEDS: List[int] = [42, 52, 62]
TASK2_ENABLE_TABNET: bool = False
TASK2_ENABLE_CROSS_TASK_AUX: bool = True
TASK2_ENABLE_CLUSTERING: bool = True

TASK2_ALLOW_ANXIETY_ABLATION: bool = True
TASK2_UNCERTAIN_FEATURES: List[str] = ["range_anxiety_score"]
TASK2_PERMUTATION_N: int = 200
TASK2_BOOTSTRAP_N: int = 1000
TASK2_ACC_BOOST_MIN_LIFT: float = 0.0005


TASK2_STAGES: List[str] = [
    "audit",
    "split",
    "baseline",
    "label_audit",
    "separability",
    "missingness",
    "segment_analysis",
    "feature_ablation",
    "empirical_bayes",
    "clustering_prior",
    "m1_logistic_gam",
    "m2_catboost",
    "m3_lightgbm",
    "m4_xgboost",
    "m5_extra_trees",
    "m6_tabnet",
    "cross_task_aux",
    "diversity",
    "soft_blend",
    "stacking",
    "calibration",
    "threshold",
    "permutation_test",
    "multiseed",
    "bootstrap",
    "final_lock",
    "final_validation",
    "explain",
    "report",
    "submission",
]


TASK3_NAME: str = "task3"
TASK3_SOURCE_SCORE: str = "range_anxiety_score"
TASK3_THRESHOLD: float = 5.0
TASK3_LABEL_ORDER: List[str] = ["0", "1"]


TASK3_FEATURES: List[str] = [
    "daily_commute_km",
    "weekly_travel_distance_km",
    "charging_station_accessibility",
    "ev_knowledge_score",
    "battery_replacement_concern",
]
TASK3_OFFICIAL_FEATURES: List[str] = list(TASK3_FEATURES)


TASK3_STRONG_FEATURES: List[str] = [
    "environmental_awareness_score",
    "ev_knowledge_score",
    "technology_affinity_score",
    "annual_income",
]


TASK3_SAFE_CANDIDATE_FEATURES: List[str] = [
    "daily_commute_km",
    "weekly_travel_distance_km",
    "charging_station_accessibility",
    "ev_knowledge_score",
    "battery_replacement_concern",
    "environmental_awareness_score",
    "technology_affinity_score",
    "annual_income",
    "age",
    "education_level",
    "city_type",
    "current_vehicle_type",
    "vehicle_age_years",
    "fuel_expense_per_month",
    "nearest_charging_station_km",
    "electricity_cost_per_kwh",
    "government_incentive_awareness",
    "previous_ev_experience",
]

TASK3_EXCLUDED_FEATURES: List[str] = [
    "range_anxiety_score",
    "home_charging_available",
    "ev_adoption_likelihood",
    "monthly_energy_consumption_kwh",
    "monthly_charging_cost",
]

TASK3_MECHANISM_FEATURES: List[str] = [
    "log_annual_income",
    "commute_ratio",
    "knowledge_deficit",
    "eco_tech_mean",
    "eco_minus_tech",
    "fuel_income_pressure",
]


TASK3_MECH_GROUPS: Dict[str, List[str]] = {
    "T_travel": [
        "avg_daily_weekly_travel",
        "travel_pressure",
        "commute_share",
        "weekly_minus_commute",
        "long_trip_pressure",
        "log_daily_commute",
        "log_weekly_travel",
        "commute_ratio",
    ],
    "C_charging": [
        "charging_gap",
        "charging_convenience",
        "distance_access_interaction",
        "log_station_distance",
    ],
    "K_cognition": [
        "knowledge_deficit",
        "knowledge_tech_gap",
        "knowledge_tech_mean",
        "knowledge_tech_interaction",
        "experience_knowledge",
    ],
    "B_concern": [
        "concern_knowledge_ratio",
        "concern_tech_ratio",
        "concern_knowledge_gap",
        "concern_tech_gap",
    ],
    "BT_concern_travel": [
        "concern_commute",
        "concern_weekly_travel",
        "charging_travel_pressure",
        "distance_travel_interaction",
    ],
    "E_economy": [
        "log_income",
        "log_annual_income",
        "monthly_income",
        "fuel_income_pressure",
        "fuel_monthly_income_ratio",
    ],
    "A_eco": [
        "eco_tech_mean",
        "eco_tech_interaction",
        "eco_minus_tech",
    ],
    "D_demo": [
        "vehicle_age_ratio",
        "experience_age_interaction",
        "vehicle_familiarity",
    ],
}

TASK3_KNOWN_CATEGORICAL: List[str] = [
    "education_level",
    "city_type",
    "current_vehicle_type",
]

TASK3_N_FOLDS: int = 5
TASK3_THRESHOLD_MIN: float = 0.30
TASK3_THRESHOLD_MAX: float = 0.70
TASK3_THRESHOLD_STEP: float = 0.005
TASK3_EARLY_STOPPING_ROUNDS: int = 80
TASK3_TIE_EPS: float = 0.0005
TASK3_STABILITY_SEEDS: List[int] = [42, 52, 62]
TASK3_OPTUNA_TRIALS_CB: int = 40
TASK3_OPTUNA_TRIALS_LGB: int = 25
TASK3_OPTUNA_COARSE_FOLDS: int = 3
TASK3_ENABLE_GAM: bool = False
TASK3_ENABLE_TABNET: bool = False
TASK3_ENABLE_CROSS_TASK_AUX: bool = False

TASK3_STAGES: List[str] = [
    "audit",
    "split",
    "baseline",
    "separability",
    "feature_ablation",
    "feature_selection",
    "m1_catboost_classifier",
    "m1_optuna",
    "m2_catboost_regressor",
    "m3_ordinal_score",
    "m4_lightgbm",
    "m5_xgboost",
    "m6_extra_trees",
    "m7_gam",
    "m8_tabnet",
    "cross_task_aux",
    "diversity",
    "classifier_regressor_fusion",
    "soft_blend",
    "stacking",
    "calibration",
    "threshold_postprocess",
    "multiseed",
    "robustness",
    "final_lock",
    "final_validation",
    "shap",
    "report",
    "submission",
]


REQUIRED_COLUMNS: List[str] = sorted(
    set(
        TASK1_FEATURES
        + [TASK1_TARGET]
        + TASK2_BASE_FEATURES
        + [TASK2_TARGET]
        + TASK3_FEATURES
        + [TASK3_SOURCE_SCORE]
    )
)


KNOWN_CATEGORICAL_FEATURES: List[str] = [
    "education_level",
    "city_type",
]


CATBOOST_BASELINE_COMMON: Dict[str, Any] = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": 50,
    "allow_writing_files": False,
}

TASK1_MODEL_PARAMS: Dict[str, Any] = {
    **CATBOOST_BASELINE_COMMON,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
}


TASK1_CATBOOST_COMMON: Dict[str, Any] = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
}


TASK3_MODEL_PARAMS: Dict[str, Any] = {
    **CATBOOST_BASELINE_COMMON,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
}


TASK3_CATBOOST_COMMON: Dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
}


TASK2_CATBOOST_COMMON: Dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
}


TASK2_MODEL_PARAMS: Dict[str, Any] = dict(TASK2_CATBOOST_COMMON)
TASK2_FEATURES: List[str] = list(TASK2_BASE_FEATURES)


SCORE_WEIGHTS: Dict[str, float] = {
    "ACC1": 0.2,
    "ACC2": 0.4,
    "ACC3": 0.4,
}
