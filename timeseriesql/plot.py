from dataclasses import dataclass
from typing import Callable, Any, Dict
from .timeseries import TimeSeries


def generate_title(ts: TimeSeries) -> str:
    """Generate a title based on common tags"""
    # get common labels
    labels = ts._get_unique_keys()
    if len(labels) > 0:
        return ".".join([str(v) for k, v in labels.items()])
    return "TimeSeries"


def generate_xlabel(ts: TimeSeries) -> str:
    return "Value"


def generate_ylabel(ts: TimeSeries) -> str:
    return "Value"


def generate_legend_labels(ts: TimeSeries) -> Dict[str, str]:
    common_labels = ts._get_unique_keys().keys()
    labels = []
    for l in ts.labels:
        labels.append(".".join([str(v) for k, v in l.items() if k not in common_labels]))
    return labels


def generate_chart_options(ts: TimeSeries) -> Dict[str, str]:
    return {}


@dataclass
class Plot:

    title_func: Callable[[Any], str] = generate_title
    xlabel_func: Callable[[Any], str] = generate_xlabel
    ylabel_func: Callable[[Any], str] = generate_ylabel
    legend_labels_func: Callable[[Any], str] = generate_legend_labels
    chart_options_func: Callable[[Any], str] = generate_chart_options

    def line_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all TimeSeries columns as a line """
        return NotImplementedError

    def dist_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all TimeSeries columns as a distribution """
        return NotImplementedError

    def stacked_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all TimeSeries columns as a stacked chart """
        return NotImplementedError

    def timebox_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all TimeSeries columns as a time box chart """
        return NotImplementedError

    def correlogram_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all auto-correlation chart """
        return NotImplementedError

    def text_plot(self, ts: TimeSeries, **kwargs):
        """ Plot a single value """
        return NotImplementedError

    def heatmap_plot(self, ts: TimeSeries, **kwargs):
        """ Plot all TimeSeries columns as a heatmap """
        raise NotImplementedError

    def pacf(self, ts: TimeSeries, **kwargs):
        """ Plot a pacf chart """
        raise NotImplementedError

    def acf(self, ts: TimeSeries, **kwargs):
        """ Plot a acf chart """
        raise NotImplementedError

    def qq_plot(self, ts: TimeSeries, **kwargs):
        """ Plot a qq plot """
        raise NotImplementedError

    def probability_plot(self, ts: TimeSeries, **kwargs):
        """ Plot a probablility plot """
        raise NotImplementedError

    def polar_plot(self, ts: TimeSeries, **kwargs):
        """ Plot a polar plot """
        raise NotImplementedError
