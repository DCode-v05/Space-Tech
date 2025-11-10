"""
Evaluation package for GNSS error prediction model.

This package provides functionality for evaluating model performance,
including metrics calculation and visualization tools.
"""

from .metrics import (
    calculate_metrics,
    calculate_crps,
    calculate_calibration,
    evaluate_predictions,
)

from .visualization import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_feature_importance,
    plot_correlation_matrix,
)

__all__ = [
    'calculate_metrics',
    'calculate_crps',
    'calculate_calibration',
    'evaluate_predictions',
    'plot_training_history',
    'plot_predictions',
    'plot_error_distribution',
    'plot_feature_importance',
    'plot_correlation_matrix',
]
