import multiprocessing
import glob
import random
import argparse
from itertools import combinations
import time
import tempfile
import logging
import logging.config
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
from multiprocessing import Process, JoinableQueue, Manager, Queue,Pool


def parse_args():
    parser = argparse.ArgumentParser(description='Program to generate train data')

    # vocabulary parameter
    parser.add_argument('file_pattern', type=str, help='corpus file pattern')

    return parser.parse_args()


@unique
class data_label(Enum):
    negative_label = 0
    positive_label = 1


def build_item_random_dict(corpus_files, min_item_length=2):
    rd = RandomDict()
    for line in tqdm(sentence_gen(corpus_files)):
        items = line.split("\t")[1:]
        for item in items:
            if len(item.split()) >= min_item_length:
                rd.setdefault(str(item))
    logging.info("item dict size:%d", len(rd))
    return rd


def saver(parsed_data_queue):
    fd = tempfile.NamedTemporaryFile(suffix='_%s' % time.strftime("%Y%m%d%H%M%S"), prefix='train_data_raw_', dir=rawdata_dir,
                                delete=False, mode='w', encoding="utf-8")
    fp = tempfile.NamedTemporaryFile(
            suffix='_%s' % time.strftime("%Y%m%d%H%M%S"), prefix='train_data_parsed_', dir=rawdata_dir, delete=False,
            mode='w', encoding="utf-8")
    i = 0
    while True:
        try:
            data_item_str, parsed_data_str = parsed_data_queue.get()
            fd.write(data_item_str + '\n')
            fp.write(parsed_data_str + '\n')
        finally:
            i += 1
            if i % 10000 == 0:
                logging.info("Finish: %d", i)
            parsed_data_queue.task_done()


def worker(data_queue, parsed_data_queue):
    tokenizer = TextBlobTokenizerHelper(unk_token=unk_token, num_word=_NUM, punc_word=_PUNC)
    while True:
        try:
            data_item = data_queue.get()
            parsed_data = map(lambda data: data if is_number(data) else ' '.join(tokenizer.tokenize(data)), data_item)
            data_item_str = '\t'.join(data_item) + '\n'
            parsed_data_str = '\t'.join(parsed_data) + '\n'
            parsed_data_queue.put((data_item_str, parsed_data_str))
        finally:
            data_queue.task_done()


def generate_train_data(taskq, corpus_files, max_num_every_item=8, min_item_length=2, pos_number=3, neg_number=2):
    rd = build_item_random_dict(corpus_files, min_item_length)
    for num, line in tqdm(enumerate(sentence_gen(corpus_files))):
        items = line.split("\t")
        items = items[1:]
        if len(items) < 3:
            continue
        for nu, comb_item in enumerate(combinations(items, pos_number)):
            data_item = list()
            if nu > max_num_every_item:
                break
            comb_item = list(comb_item)
            random.shuffle(comb_item)
            if any([len(ele.split()) < min_item_length for ele in comb_item]):
                continue
            source = comb_item[0]
            data_item.append(source)
            for ele in comb_item[1:]:
                data_item.append(ele)
                data_item.append(str(data_label.positive_label.value))
            src_and_pos_size = len(data_item)
            while len(data_item) - src_and_pos_size < 2 * neg_number:
                neg_item = rd.random_key()
                if comb_item[0] not in neg_item and neg_item not in comb_item[0]:
                    data_item.append(neg_item)
                    data_item.append(str(data_label.negative_label.value))
            taskq.put(data_item)


def setup_logger():
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)

if __name__ == '__main__':
    data_queue = JoinableQueue()
    parsed_data_queue = JoinableQueue()
    setup_logger()
    tokenizer = TextBlobTokenizerHelper(unk_token=unk_token, num_word=_NUM, punc_word=_PUNC)
    args = parse_args()
    corpus_files = glob.glob(args.file_pattern)
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=worker, args=(data_queue, parsed_data_queue, ))
        p.daemon = True
        p.start()
    s = multiprocessing.Process(target=saver, args=(parsed_data_queue, ))
    s.daemon = True
    s.start()
    generate_train_data(data_queue, corpus_files)
    data_queue.join()
    parsed_data_queue.join()
