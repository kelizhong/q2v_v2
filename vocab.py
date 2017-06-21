import os
import sys
import argparse
import logbook as logging
import signal
from vocabulary.vocab import Vocab
from argparser.customArgType import FileType
from utils.data_util import aksis_sentence_gen


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
    q2v_vocab_parser.add_argument('-vf', '--vocab-file',
                                  type=FileType,
                                  default=os.path.join(os.path.dirname(__file__), 'data', 'vocabulary', 'vocab.pkl'),
                                  help='the file with the words which are the most command words in the corpus')
    q2v_vocab_parser.add_argument('-w', '--workers-num',
                                  type=int,
                                  default=10,
                                  help='the file with the words which are the most command words in the corpus')
    q2v_vocab_parser.add_argument('files', nargs='+',
                                  help='the corpus input files')

    return parser.parse_args()


def signal_handler(signal, frame):
    logging.info('Stop!!!')
    sys.exit(0)


def setup_logger():
    from utils.log_util import Logger
    log = Logger()
    log.set_stream_handler()
    log.set_time_rotating_file_handler(args.log_file_name)


if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)

    vocab = Vocab(args.files, args.vocab_file, workers_num=args.workers_num, sentence_gen=aksis_sentence_gen,
                  overwrite=args.overwrite)
    vocab.create_dictionary()
