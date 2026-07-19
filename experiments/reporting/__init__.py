"""Reporting package: tables + plots."""

from experiments.reporting.tables import report_tables
from experiments.reporting.plots import save_plots
from experiments.reporting.json_export import write_results_json
from experiments.reporting.axis_analysis import axis_analysis, report_axis_analysis
from experiments.reporting.sparse_plots import (
    plot_convergence,
    plot_pareto,
    plot_metrics_bar,
)

__all__ = [
    "report_tables",
    "save_plots",
     "write_results_json",
     "axis_analysis",
     "report_axis_analysis",
    "plot_convergence",
    "plot_pareto",
    "plot_metrics_bar",
]