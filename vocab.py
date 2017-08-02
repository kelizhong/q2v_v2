import argparse
import signal
import sys

import logging.config
import yaml

from argparser.customArgType import FileType
from utils.config_decouple import config
from helper.vocabulary_helper import VocabularyHelper
from helper.data_parser import QueryPairParser


def parse_args():
    parser = argparse.ArgumentParser(description='Vocabulary tools')

    # vocabulary parameter
    parser.add_argument('-vd', '--vocab-data-dir', type=FileType, default=config('vocabulary_dir'),
                                  help='the file with the words which are the most command words in the corpus')
    parser.add_argument('-vs', '--vocab-size', type=int, default=sys.maxsize,
                                  help='vocabulary size')
    parser.add_argument('files', nargs='+', help='the corpus input files')

    return parser.parse_args()


def signal_handler(signal, frame):
    logging.info('Stop!!!')
    sys.exit(0)


def setup_logger():
    logging_config_path = config('logging_config_path')
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    parser = QueryPairParser()
    v = VocabularyHelper(vocabulary_data_dir=args.vocab_data_dir)
    v.build_word_counter(args.files, parser=parser)
    v.build_vocabulary_from_counter(vocabulary_size=args.vocab_size, special_words=config(section='vocabulary_symbol'))
