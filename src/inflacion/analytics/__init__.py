"""Cálculos analíticos sobre el INPC."""
from inflacion.analytics.alt_core import trimmed_mean_yoy, weighted_median_yoy
from inflacion.analytics.base_effect import identify_outliers_expanding_window
from inflacion.analytics.contributions import (
    contributions_mom,
    contributions_yoy,
    incidencias,
)
from inflacion.analytics.core_breakdown import (
    breakdown_core_food_nonfood,
    breakdown_core_goods_services,
    breakdown_core_services,
)
from inflacion.analytics.distributions import bucket_distribution
from inflacion.analytics.zscore import rolling_zscore

__all__ = [
    "breakdown_core_food_nonfood",
    "breakdown_core_goods_services",
    "breakdown_core_services",
    "bucket_distribution",
    "contributions_mom",
    "contributions_yoy",
    "identify_outliers_expanding_window",
    "incidencias",
    "rolling_zscore",
    "trimmed_mean_yoy",
    "weighted_median_yoy",
]
