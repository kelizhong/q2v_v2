# coding=utf-8
"""Collect the tokenize sentence from worker"""
import os
import logging
import zmq
import itertools
from utils.config_decouple import config
from utils.retry_util import retry
from utils.appmetric_util import AppMetric

logger = logging.getLogger(__name__)


class DataCollectorProcess(object):

    def __init__(self, ip='127.0.0.1', port=5557, tries=20, metric_interval=5, file_suffix="default", name="Collector"):
        """Collect the tokenized data from worker

        Parameters
        ----------
        ip : {str}, optional
            The ip address string without the port to pass to ``Socket.bind()``
            (the default is '127.0.0.1')
        port : {number}, optional
            The port to receive the tokenize sentence from worker
            (the default is 5557)
        tries : {number}, optional
            Number of times to retry, set to 0 to disable retry
            (the default is 20)
        metric_interval : {number}, optional
            interval to print/log metric, unit is second (the default is 60)
        name : {str}, optional
            Collertor name (the default is "Collector")
        """
        self.ip = ip
        self.port = port
        self.tries = tries
        self.metric_interval = metric_interval
        self.file_suffix = file_suffix
        self.name = name
        self._init_writer_handler()

    @retry(lambda x: x.tries, exception=zmq.ZMQError,
           name="DataCollector", report=logger.error)
    def _on_recv(self, receiver):
        """Receive the py object from zmq

        Parameters
        ----------
        receiver : {zmq object}
            zmq socket to receiver pbj

        Returns
        -------
        [tuple]
            raw data item str, parsed data str
        """
        data_item_str, parsed_data_str = receiver.recv_pyobj(zmq.NOBLOCK)
        return data_item_str, parsed_data_str

    def _init_writer_handler(self):
        raw_file_path = os.path.join(config('rawdata_dir'), "train_data_raw_{}".format(self.file_suffix))
        self.raw_handler = open(raw_file_path, mode='w', encoding="utf-8")
        parsed_file_path = os.path.join(config('rawdata_dir'), "train_data_parsed_{}".format(self.file_suffix))
        self.parsed_handler = open(parsed_file_path, mode='w', encoding="utf-8")

    def collect(self):
        """Generator that receive the tokenize sentence from worker and produce the words"""
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.port))
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        for _ in itertools.count():
            try:
                data_item_str, parsed_data_str = self._on_recv(receiver)
                self.raw_handler.write(data_item_str + '\n')
                self.parsed_handler.write(parsed_data_str + '\n')
                metric.notify(1)
            except zmq.ZMQError as e:
                logger.error(e)
                break
        self.terminate()

    def terminate(self):
        """Close the writer handler
        """
        logger.info("Closing saver file handler")
        self.raw_handler.close()
        self.parsed_handler.close()
