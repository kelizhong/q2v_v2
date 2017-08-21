import glob
import argparse
import os
import signal
import sys

import logging
import logging.config
import yaml
from utils.config_decouple import config
from helper.data_record_helper import DataRecordHelper
from helper.data_parser import QueryPairParser
from utils.file_util import ensure_dir_exists

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate tf records files')

    parser.add_argument('-fs', '--file-suffix', type=str, help='suffix for tfrecord train data', default="default")

    parser.add_argument('-mw', '--min-words', type=int, default=2, help='ignore the sequence that length < min_words')
    parser.add_argument('file_pattern', type=str, help='the corpus input files pattern')

    return parser.parse_args()


def signal_handler(signal, frame):
    logger.info('Stop!!!')
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
    files = glob.glob(args.file_pattern)
    d = DataRecordHelper()
    gen = parser.siamese_sequences_to_tokens_generator(files, args.min_words)
    tfrecord_path = os.path.join(config('traindata_dir'), "train.tfrecords.{}".format(args.file_suffix))
    ensure_dir_exists(tfrecord_path, is_dir=False)
    d.create_sequence(gen, record_path=tfrecord_path)
