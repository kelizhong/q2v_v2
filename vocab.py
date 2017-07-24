import argparse
import os
import signal
import sys

import logging
import yaml

from argparser.customArgType import FileType
from config.config import project_dir
from utils.data_util import sentence_gen
from vocabulary.vocab import VocabFromZMQ, VocabularyFromCustomStringTrigram
from config.config import logging_config_path


def parse_args():
    parser = argparse.ArgumentParser(description='Vocabulary tools')
    parser.add_argument('--log-file-name', default=os.path.join(os.getcwd(), 'data/logs', 'vocab.log'),
                        type=FileType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('--metric-interval', default=6, type=int,
                        help='metric reporting frequency is set by seconds param')
    subparsers = parser.add_subparsers(help='build vocabulary')

    q2v_vocab_parser = subparsers.add_parser("query2vec_vocab")
    q2v_vocab_parser.set_defaults(action='query2vec_vocab')

    # vocabulary parameter

    q2v_vocab_parser.add_argument('--overwrite', action='store_true', help='overwrite earlier created files, also forces the\
                        program not to reuse count files')
    q2v_vocab_parser.add_argument('-vf', '--vocab-data-dir',
                                  type=FileType,
                                  default=os.path.join(project_dir, 'data', 'vocabulary'),
                                  help='the file with the words which are the most command words in the corpus')
    q2v_vocab_parser.add_argument('-w', '--workers-num',
                                  type=int,
                                  default=10,
                                  help='the file with the words which are the most command words in the corpus')
    q2v_vocab_parser.add_argument('-files', nargs='+',
                                  help='the corpus input files')

    q2v_vocab_parser.add_argument('-string', type=str, default='abcdefghijklmnopqrstuvwxyz1234567890#.&\\',
                                  help='the corpus input files')
    return parser.parse_args()


def signal_handler(signal, frame):
    logging.info('Stop!!!')
    sys.exit(0)


def setup_logger():
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    if args.files:
        VocabFromZMQ(vocabulary_data_dir=args.vocab_data_dir, workers_num=args.workers_num,
                         sentence_gen=sentence_gen,
                         overwrite=args.overwrite).build_words_frequency_counter(args.files)
    else:
        VocabularyFromCustomStringTrigram(vocabulary_data_dir=args.vocab_data_dir).build_words_frequency_counter(args.string)
