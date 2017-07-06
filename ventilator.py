# coding=utf-8
"""
produce train data for query2vec model
"""

import os
import sys
from argparser.customArgType import FileType
from argparser.customArgAction import AppendTupleWithoutDefault
import argparse
import signal
from utils.log_util import Logger
import logbook


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq query2vec for query2vec')
    parser.add_argument('--log-file-name', default=os.path.join(os.getcwd(), 'data/logs', 'aksis_data.log'),
                        type=FileType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('--metric-interval', default=6, type=int,
                        help='metric reporting frequency is set by seconds param')
    subparsers = parser.add_subparsers(help='train vocabulary')

    q2v_aksis_ventilator_parser = subparsers.add_parser("q2v_aksis_ventilator")
    q2v_aksis_ventilator_parser.set_defaults(action='q2v_aksis_ventilator')

    q2v_aksis_ventilator_parser.add_argument('data_dir', type=str,
                                              help='the file name of the encoder input for training')
    q2v_aksis_ventilator_parser.add_argument('--vocabulary-data-dir',
                                              default=os.path.join(os.getcwd(), 'data', 'vocabulary'),
                                              type=str,
                                              help='vocabulary with he most common words')
    q2v_aksis_ventilator_parser.add_argument('-ap', '--action-patterns', nargs=2, action=AppendTupleWithoutDefault,
                                              default=[('*add', -1), ('*search', 0.5), ('*click', 0.4),
                                                       ('*purchase', -1)])
    q2v_aksis_ventilator_parser.add_argument('--ip-addr', type=str, help='ip address')
    q2v_aksis_ventilator_parser.add_argument('--port', type=str, help='zmq port')
    q2v_aksis_ventilator_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                                              help='batch size for each databatch')
    q2v_aksis_ventilator_parser.add_argument('--top-words', default=sys.maxsize, type=int,
                                              help='the max sample num for training')
    q2v_aksis_ventilator_parser.add_argument('--worker-num', default=1, type=int,
                                              help='number of parser worker')
    return parser.parse_args()


def signal_handler(signal, frame):
    logbook.warn('Stop!!!')
    sys.exit(0)


def setup_logger():
    log = Logger()
    log.set_stream_handler()
    log.set_time_rotating_file_handler(args.log_file_name)

if __name__ == "__main__":
    args = parse_args()
    setup_logger()
    signal.signal(signal.SIGINT, signal_handler)
    if args.action == 'q2v_aksis_ventilator':
        from data_io.distribute_stream.aksis_data_pipeline import AksisDataPipeline

        p = AksisDataPipeline(args.data_dir, args.vocabulary_data_dir, args.top_words, args.action_patterns,
                              args.batch_size, worker_num=args.worker_num)
        p.start_all()
