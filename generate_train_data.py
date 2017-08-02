import glob
import random
import argparse
from itertools import combinations
import time
import tempfile
import logging
import logging.config
import multiprocessing as mp
import itertools
import yaml
from enum import Enum, unique
from tqdm import tqdm
from utils.config_decouple import config
from utils.data_util import sentence_gen
from utils.random_dict import RandomDict
from helper.tokenizer_helper import TextBlobTokenizerHelper
from utils.data_util import is_number
from utils.context_util import elapsed_timer


def parse_args():
    parser = argparse.ArgumentParser(description='Program to generate train data')

    # vocabulary parameter
    parser.add_argument('file_pattern', type=str, help='corpus file pattern')

    return parser.parse_args()


class SaverProcess(mp.Process):
    def __init__(self, queue, info_interval=10000):
        super(SaverProcess, self).__init__()
        self.queue = queue
        self.info_interval = info_interval
        self.init_writer_handler()

    def init_writer_handler(self):
        self.raw_handler = tempfile.NamedTemporaryFile(suffix='_%s' % time.strftime("%Y%m%d%H%M%S"),
                                                       prefix='train_data_raw_',
                                                       dir=config('rawdata_dir'),
                                                       delete=False, mode='w', encoding="utf-8")
        self.parsed_handler = tempfile.NamedTemporaryFile(
            suffix='_%s' % time.strftime("%Y%m%d%H%M%S"), prefix='train_data_parsed_', dir=config('rawdata_dir'), delete=False,
            mode='w', encoding="utf-8")

    def close_writer_handler(self):
        logging.info("Closing saver file handler")
        self.raw_handler.close()
        self.parsed_handler.close()

    def run(self):
        logging.info("Starting Saver")
        with elapsed_timer() as elapsed:
            for i in itertools.count():
                try:
                    data_item_str, parsed_data_str = self.queue.get()
                    self.raw_handler.write(data_item_str + '\n')
                    self.parsed_handler.write(parsed_data_str + '\n')
                finally:
                    if i % self.info_interval == 0:
                        logging.info("Saver finished: %d, %.4f sents/s", i, i / elapsed())
                    self.queue.task_done()

    def terminate(self):
        super(SaverProcess, self).terminate()
        self.close_writer_handler()


class WorkerProcess(mp.Process):
    def __init__(self, fronted_queue, backend_queue, name="worker", info_interval=10000):
        super(WorkerProcess, self).__init__()
        self.fronted_queue = fronted_queue
        self.backend_queue = backend_queue
        self.info_interval = info_interval
        self.name = name

    def run(self):
        logging.info("Starting %s", self.name)
        tokenizer = TextBlobTokenizerHelper()
        with elapsed_timer() as elapsed:
            for i in itertools.count():
                try:
                    data_item = self.fronted_queue.get()
                    parsed_data = map(lambda data: data if is_number(data) else ' '.join(tokenizer.tokenize(data)),
                                      data_item)
                    data_item_str = '\t'.join(data_item) + '\n'
                    parsed_data_str = '\t'.join(parsed_data) + '\n'
                    self.backend_queue.put((data_item_str, parsed_data_str))
                finally:
                    if i % self.info_interval == 0:
                        logging.info("%s finished: %d, %.4f sents/s", self.name, i, i / elapsed())
                    self.fronted_queue.task_done()


class ProducerProcess(mp.Process):
    def __init__(self, queue, corpus_files, max_num_every_item=8, min_item_length=2, pos_number=3, neg_number=2,
                 info_interval=10000, name="producer"):
        super(ProducerProcess, self).__init__()
        self.queue = queue
        self.corpus_files = corpus_files
        self.max_num_every_item = max_num_every_item
        self.min_item_length = min_item_length
        self.min_item_length = min_item_length
        self.info_interval = info_interval
        self.pos_number = pos_number
        self.neg_number = neg_number
        self.name = name

    def run(self):
        logging.info("Starting %s", self.name)
        rd = self.build_item_random_dict()
        for num, line in enumerate(sentence_gen(corpus_files)):
            items = line.split("\t")
            items = items[1:]
            if len(items) < 3:
                continue
            for nu, comb_item in enumerate(combinations(items, self.pos_number)):
                data_item = list()
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
                self.queue.put(data_item)

            if num % self.info_interval == 0:
                logging.info("Producer finished: %d", num)

    @unique
    class data_label(Enum):
        negative_label = 0
        positive_label = 1

    def build_item_random_dict(self):
        logging.info("Building item dict")
        rd = RandomDict()
        for line in tqdm(sentence_gen(self.corpus_files)):
            items = line.split("\t")[1:]
            for item in items:
                if len(item.split()) >= self.min_item_length:
                    rd.setdefault(str(item))
        logging.info("Finish item dict, item dict size:%d", len(rd))
        return rd


def setup_logger():
    logging_config_path = config('logging_config_path')
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


if __name__ == '__main__':
    data_queue = mp.JoinableQueue()
    parsed_data_queue = mp.JoinableQueue()
    setup_logger()
    args = parse_args()
    # corpus_files = glob.glob('/Users/keliz/Downloads/aksis.purchased.pair/part*')
    corpus_files = glob.glob(args.file_pattern)
    for i in range(min(4, mp.cpu_count())):
        w = WorkerProcess(data_queue, parsed_data_queue, "worker_%d" % i)
        w.daemon = True
        w.start()
    s = SaverProcess(parsed_data_queue)
    s.start()
    p = ProducerProcess(data_queue, corpus_files)
    p.start()
    p.join()
    data_queue.join()
    parsed_data_queue.join()
    s.terminate()
    p.terminate()
