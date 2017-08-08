# coding=utf-8
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""ventilator that read/produce the corpus data"""
from multiprocessing import Process
import glob
import logging
from enum import Enum, unique
from itertools import combinations
import random
import zmq
from zmq.decorators import socket
from utils.appmetric_util import AppMetric
from utils.data_util import sentence_gen
from utils.random_dict import RandomDict
from utils.context_util import elapsed_timer

logger = logging.getLogger(__name__)


class DataVentilatorProcess(Process):

    def __init__(self, file_pattern, build_item_dict_status, ip='127.0.0.1', port=5555,
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
        self.file_pattern = file_pattern
        self.max_num_every_item = max_num_every_item
        self.min_item_length = min_item_length
        self.pos_number = pos_number
        self.build_item_dict_status = build_item_dict_status
        # pylint: disable=invalid-name
        self.ip = ip
        self.port = port
        self.metric_interval = metric_interval
        self.neg_number = neg_number
        self.name = name

    # pylint: disable=arguments-differ, no-member
    @socket(zmq.PUSH)
    def run(self, sender):
        sender.connect("tcp://{}:{}".format(self.ip, self.port))
        logger.info(
            "process %s connect %s:%d and start produce data", self.name, self.ip, self.port)
        metric = AppMetric(name=self.name, interval=self.metric_interval)
        rd = self.build_item_random_dict()
        data_stream = self.get_data_stream()
        for line in data_stream:
            items = line.split("\t")
            items = items[1:]
            if len(items) < 3:
                continue
            for nu, comb_item in enumerate(combinations(items, self.pos_number)):
                data_item = list()
                # This avoid the train contain too much
                if nu > self.max_num_every_item:
                    break
                comb_item = list(comb_item)
                random.shuffle(comb_item)
                if any([len(ele.split()) < self.min_item_length for ele in comb_item]):
                    continue
                source = comb_item[0]
                data_item.append(source)
                for ele in comb_item[1:]:
                    data_item.append(ele)
                    data_item.append(str(self.data_label.positive_label.value))
                src_and_pos_size = len(data_item)
                while len(data_item) - src_and_pos_size < 2 * self.neg_number:
                    neg_item = rd.random_key()
                    if comb_item[0] not in neg_item and neg_item not in comb_item[0]:
                        data_item.append(neg_item)
                        data_item.append(str(self.data_label.negative_label.value))
                sender.send_pyobj(data_item)
                metric.notify(1)

    @unique
    class data_label(Enum):
        negative_label = 0
        positive_label = 1

    def build_item_random_dict(self):
        """Build random dict with all items for choosing negative sample

        Returns
        -------
        [RandomDict]
            RandomDict with all items
        """
        logging.info("Building item dict")
        data_stream = self.get_data_stream()
        rd = RandomDict()
        with elapsed_timer() as elapsed:
            for i, line in enumerate(data_stream):
                items = line.split("\t")[1:]
                for item in items:
                    if len(item.split()) >= self.min_item_length:
                        rd.setdefault(str(item))

                if i % 10000 == 0:
                    logger.info("Ventilator finished: %d, %.4f sents/s", i, i / elapsed())
        logger.info("%s finish item dict, item dict size:%d", self.name, len(rd))
        self.build_item_dict_status.value = True
        return rd

    def get_data_stream(self):
        """data stream generate the query, title data"""

        logger.info("Reading file: %s", self.file_pattern[0])
        data_files = glob.glob(self.file_pattern[0])

        if len(data_files) <= 0:
            raise FileNotFoundError(
                "no files are found for file pattern {}".format(self.file_pattern))
        # action_files = [os.path.join(self.data_dir, filename) for filename in data_files]

        for line in sentence_gen(data_files):
            yield line
