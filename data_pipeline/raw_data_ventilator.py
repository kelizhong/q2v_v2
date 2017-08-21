# coding=utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""ventilator that read/produce the corpus data"""
from multiprocessing import Process
import logging
import json
from itertools import combinations
import zmq
from zmq.decorators import socket
import smart_open
from utils.appmetric_util import AppMetric
from utils.random_dict import RandomDict
from utils.common_util import nslice

logger = logging.getLogger(__name__)


class DataVentilatorProcess(Process):

    AKSIS_KEYWORDS_TYPE = ["KeywordsByPurchases", "KeywordsByAdds", "KeywordsByClicks"]

    def __init__(self, s3_uris, ip='127.0.0.1', port=5555, max_query_num=6, asin_tag="Asin", query_dict_min_size=1000,
                 metric_interval=60, max_num_every_item=8, min_item_length=2, pos_number=3, neg_number=2, name='VentilatorProcess'):
        """Process to read the corpus data

        Parameters
        ----------
        file_pattern : {tuple}
            File pattern use to distinguish different corpus, every file pattern will start
            a ventilator process.
            File pattern is tuple type(file pattern, ).
        ip : {str}, optional
            The ip address string without the port to pass to ``Socket.bind()``.
            (the default is '127.0.0.1')
        port : {number}, optional
            Port to produce the raw data (the default is 5555)
        query_dict_min_size : {number}, optional
            min size for query radom dict
        metric_interval : {number}, optional
            interval to print/log metric, unit is second (the default is 60)
        max_num_every_item : {number}, optional
            define the max number for item combinations (the default is 8)
        min_item_length : {number}, optional
            ignore the item  which length <  `min_item_length` (the default is 2)
        pos_number : {number}, optional
            Number of positive sample  (the default is 3)
        neg_number : {number}, optional
            Number of negative sample (the default is 2)
        name : {str}, optional
            Proicess Name (the default is 'VentilatorProcess')
        """
        Process.__init__(self)
        self.s3_uris = s3_uris
        self.max_num_every_item = max_num_every_item
        self.min_item_length = min_item_length
        self.pos_number = pos_number
        self.asin_tag = asin_tag
        self.query_dict_min_size = query_dict_min_size
        # pylint: disable=invalid-name
        self.ip = ip
        self.port = port
        self.metric_interval = metric_interval
        self.max_query_num = max_query_num
        self.neg_number = neg_number
        self.name = name

    # pylint: disable=arguments-differ, no-member
    @socket(zmq.PUSH)
    def run(self, sender):
        sender.connect("tcp://{}:{}".format(self.ip, self.port))
        logger.info(
            "process %s connect %s:%d and start produce data", self.name, self.ip, self.port)
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        rd = RandomDict()
        data_stream = self.get_s3_data_stream()
        for line in data_stream:
            data = json.loads(line.decode("utf-8"))
            queries = self.extract_queries(data)
            asin = data[self.asin_tag]
            for query, _ in queries:
                if len(query.split()) > self.min_item_length:
                    rd.setdefault(query)
            if len(rd) < self.query_dict_min_size:
                continue
            for nu, comb_item in enumerate(nslice(queries, self.pos_number, truncate=True)):
                data_item = list()
                data_item.append(asin)
                for query, count in comb_item:
                    data_item.append(query)
                    data_item.append(str(count))
                src_and_pos_size = len(data_item)
                while len(data_item) - src_and_pos_size < 2 * self.neg_number:
                    neg_item = rd.random_key()
                    data_item.append(neg_item)
                    data_item.append("0")
                sender.send_pyobj(data_item)
                metric.notify(1)

    def extract_queries(self, data):
        queries = [(item['keywords'], item['count']) for field in self.AKSIS_KEYWORDS_TYPE for item in data[field] if len(item['keywords'].split()) >= self.min_item_length]
        return queries

    def get_s3_data_stream(self):
        for s3_uri in self.s3_uris:
            logger.info("Reading file: %s", s3_uri)
            with smart_open.smart_open(s3_uri) as fin:
                for line in fin:
                    yield line
