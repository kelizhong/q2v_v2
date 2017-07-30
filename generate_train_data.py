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
from utils.data_util import sentence_gen
from utils.random_dict import RandomDict
from config.config import rawdata_dir
from helper.tokenizer_helper import TextBlobTokenizerHelper
from config.config import unk_token, _PUNC, _NUM
from utils.data_util import is_number
from config.config import logging_config_path
from utils.context_util import elapsed_timer


def parse_args():
    parser = argparse.ArgumentParser(description='Program to generate train data')

    # vocabulary parameter
    parser.add_argument('file_pattern', type=str, help='corpus file pattern')

    return parser.parse_args()


class WorkerProcess(mp.Process):
    def __init__(self, receiver, sender, name="worker", info_interval=10000):
        super(WorkerProcess, self).__init__()
        self.receiver = receiver
        self.sender = sender
        self.info_interval = info_interval
        self.name = name

    def close_unused_pipe_conn(self):
        # fork pipe: WorkerProcess receiver <- pipe <- WorkerProcess sender
        # close the forked pipe sender, worker not use the sender
        self.sender.close()
        # fork pipe: WorkerProcess receiver <- pipe
        # In here only WorkerProcess receiver and ProducerProcess sender connect pipe
        # WorkerProcess receiver <- pipe <- ProducerProcess sender

    def run(self):
        self.close_unused_pipe_conn()
        logging.info("Starting %s", self.name)
        tokenizer = TextBlobTokenizerHelper(unk_token=unk_token, num_word=_NUM, punc_word=_PUNC)
        with elapsed_timer() as elapsed, tempfile.NamedTemporaryFile(suffix='_%s' % time.strftime("%Y%m%d%H%M%S"),
                                                                     prefix='train_data_raw_', dir=rawdata_dir,
                                                                     delete=False, mode='w',
                                                                     encoding="utf-8") as raw_handler, tempfile.NamedTemporaryFile(
                suffix='_%s' % time.strftime("%Y%m%d%H%M%S"), prefix='train_data_parsed_', dir=rawdata_dir,
                delete=False, mode='w', encoding="utf-8") as parsed_handler:
            for i in itertools.count():
                try:
                    data_item = self.receiver.recv()
                    parsed_data = map(lambda data: data if is_number(data) else ' '.join(tokenizer.tokenize(data)),
                                      data_item)
                    data_item_str = '\t'.join(data_item) + '\n'
                    parsed_data_str = '\t'.join(parsed_data) + '\n'
                    raw_handler.write(data_item_str + '\n')
                    parsed_handler.write(parsed_data_str + '\n')
                except EOFError:
                    # when receiver can not receiver any data and the sender was closed, EORFError will be thrown
                    self.receiver.close()
                    break
                finally:
                    if i % self.info_interval == 0:
                        logging.info("%s finished: %d, %.4f sents/s", self.name, i, i / elapsed())


class ProducerProcess(mp.Process):
    def __init__(self, sender, receiver, corpus_files, max_num_every_item=8, min_item_length=2, pos_number=3,
                 neg_number=2, info_interval=10000, name="producer"):
        super(ProducerProcess, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.corpus_files = corpus_files
        self.max_num_every_item = max_num_every_item
        self.min_item_length = min_item_length
        self.min_item_length = min_item_length
        self.info_interval = info_interval
        self.pos_number = pos_number
        self.neg_number = neg_number
        self.name = name

    def close_unused_pipe_conn(self):
        # do not put it in __init__
        # In every new process, pip will be forked, so need to close some unused connect manually
        # fork pipe: ProducerProcess receiver<- pipe <-ProducerProcess sender
        # close the forked pipe receiver, producer not use the receiver
        self.receiver.close()
        # fork pipe: pipe <-ProducerProcess sender
        # In here only ProducerProcess sender connect pipe

    def run(self):
        self.close_unused_pipe_conn()
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
                self.sender.send(data_item)

            if num % self.info_interval == 0:
                logging.info("Producer finished: %d", num)
        # close sender
        self.sender.close()

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
            if len(rd) > 10000:
                break
        logging.info("Finish item dict, item dict size:%d", len(rd))
        return rd


def setup_logger():
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


if __name__ == '__main__':
    # receiver <- pipe <- sender
    receiver, sender = mp.Pipe()

    setup_logger()
    args = parse_args()
    # corpus_files = glob.glob('/Users/keliz/Downloads/aksis.purchased.pair/part*')
    corpus_files = glob.glob(args.file_pattern)

    w = WorkerProcess(receiver, sender, "worker_%d" % 0)
    w.start()
    # close main process receiver connect
    receiver.close()

    p = ProducerProcess(sender, receiver, corpus_files)
    p.start()
    p.join()
    # close main process sender connect
    sender.close()

    # wait worker
    w.join()
