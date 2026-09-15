"""Reporting package: tables + plots."""

from experiments.reporting.axis_analysis import axis_analysis, report_axis_analysis
from experiments.reporting.json_export import write_results_json
from experiments.reporting.plots import save_plots
from experiments.reporting.sparse_plots import (
    plot_convergence,
    plot_metrics_bar,
    plot_pareto,
)
from experiments.reporting.tables import report_tables

__all__ = [
    "axis_analysis",
    "plot_convergence",
    "plot_metrics_bar",
    "plot_pareto",
    "report_axis_analysis",
    "report_tables",
    "save_plots",
    "write_results_json",
]
