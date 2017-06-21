# coding=utf-8
"""metric util for application"""
import logbook as logging
import functools
from appmetrics import metrics
from appmetrics import reporter


def with_meter(name, value=1, interval=-1):
    """
    Call-counting decorator: each time the wrapped function is called
    the named meter is incremented by value.
    """

    try:
        mmetric = AppMetric(name=name, interval=interval)

    except metrics.DuplicateMetricError as e:
        mmetric = AppMetric(metric=metrics.metric(name), interval=interval)

    def wrapper(f):

        @functools.wraps(f)
        def fun(*args, **kwargs):
            res = f(*args, **kwargs)

            mmetric.notify(value)
            return res

        return fun

    return wrapper


class AppMetric(object):
    """ metric util for application
    Parameters
    ----------
    metric: "meter" metric
        custom metric
    name: str
        create a metric with name if metric is None
    interval: int
        report the metric for every interval seconds, if interval < 0, not report the metric
    """
    def __init__(self, metric=None, name='metric', interval=-1):
        self.metric = metric or metrics.new_meter(name)

        self.interval = interval
        if interval > 0:
            reporter.register(self.log_metrics, reporter.fixed_interval_scheduler(interval))

    @staticmethod
    def log_metrics(metrics):
        """log the metric"""
        logging.info(metrics)

    def notify(self, value):
        """Add a new observation to the metric"""
        self.metric.notify(value)
