import time
import argparse
import logging.config
import random
from multiprocessing import Value
import yaml
import boto3

from utils.config_decouple import config
from utils.network_util import local_ip
from utils.common_util import split_list
from argparser.customArgAction import AppendTupleWithoutDefault
from data_pipeline.raw_data_broker import RawDataBroker
from data_pipeline.data_parser_worker import DataParserWorker
from data_pipeline.raw_data_ventilator import DataVentilatorProcess
from data_pipeline.data_collector import DataCollectorProcess
from utils.socket_util import select_random_port


def parse_args():
    parser = argparse.ArgumentParser(description='Generate train data')

    parser.add_argument('-sp', '--s3-prefix', nargs=2, action=AppendTupleWithoutDefault,
                        default=[('*add', 1), ('*search', 1), ('*click', 1), ('*purchase', 1)])
    parser.add_argument('--ip', type=str, help='ip address', default=None)
    parser.add_argument('-w', '--worker-num', default=4, type=int,
                        help='number of parser worker')
    parser.add_argument('-fs', '--file-suffix', type=str, help='parsed train data file name', default="default")
    parser.add_argument('-b', '--bucket', type=str, help='s3 bucket for query2vec data', default='q2vdata')

    return parser.parse_args()


def start_data_ventilator_process(ip, port, s3_bucket, s3_prefixes):
    """start the ventilator process which read the corpus data"""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=s3_bucket)
    i = 0
    for s3_prefix, porcess_num in s3_prefixes:
        s3_uri_lst = ['s3://{0}/{1}'.format(bucket.name, obj.key) for obj in bucket.objects.filter(Prefix=s3_prefix)]
        for s3_uris in split_list(s3_uri_lst, int(porcess_num)):
            ventilator = DataVentilatorProcess(s3_uris, ip=ip, port=port, name="data_ventilator_{}".format(i))
            ventilator.daemon = True
            ventilator.start()
            i += 1


def start_raw_data_broker(ip='127.0.0.1', frontend_port=5555, backend_port=5556):
    """start the raw data broker between ventilator and parser worker process"""
    # TODO bind the random port not use the defined port
    raw_data_broker = RawDataBroker(ip=ip, frontend_port=frontend_port, backend_port=backend_port)
    raw_data_broker.start()


def start_parser_worker_process(ip='127.0.0.1', frontend_port=5556, backend_port=5557, worker_num=4):
    """start the parser worker process which tokenize the corpus data"""
    for i in range(worker_num):
        worker = DataParserWorker(ip=ip, frontend_port=frontend_port, backend_port=backend_port,
                                  name="data_parser_worker_{}".format(i))
        worker.daemon = True
        worker.start()


def setup_logger():
    logging_config_path = config('logging_config_path')
    with open(logging_config_path) as f:
        dictcfg = yaml.load(f)
        logging.config.dictConfig(dictcfg)


if __name__ == '__main__':
    """Data pipeline to pre process train data
    Start the raw_data_broker, data_ventilator_process, parser_worker_process
    and collector_process. The data pipeline is data_ventilator_process->raw_data_broker
    ->parser_worker_process->collector_process
    Parameters
    ----------
    file_patterns : {list}
        file pattern use to distinguish different corpus, every file pattern will start a
        ventilator process.
        e.g. there are four action type(KeywordsByAdds, KeywordsBySearches, KeywordsByPurchases,
        KeywordsByClicks) in aksis data, if split the aksis data to four files, like aksis.add,
        aksis.search, aksis.purchase and aksis.click, each file store the corresponding data,
        than can use these four patterns(*add, *search, *purchase, *click) to read the related
        file
    worker_num : {number}, optional
        Number of worker to parse  data (the default is 4)
    ip : {str}, optional
        The ip address string without the port to pass to ``Socket.bind()``.
        (the default is None, for None ip, will get ip automatically)
    """
    args = parse_args()
    setup_logger()
    ip = args.ip or local_ip()

    rawdata_frontend_port = select_random_port()

    rawdata_backend_port = select_random_port()

    start_raw_data_broker(ip=ip, frontend_port=rawdata_frontend_port, backend_port=rawdata_backend_port)

    start_data_ventilator_process(ip=ip, port=rawdata_frontend_port, s3_bucket=args.bucket, s3_prefixes=args.s3_prefix)

    collector_frontend_port = select_random_port()

    start_parser_worker_process(ip=ip, frontend_port=rawdata_backend_port,
                                backend_port=collector_frontend_port, worker_num=args.worker_num)
    collector = DataCollectorProcess(ip=ip, port=collector_frontend_port, file_suffix=args.file_suffix)
    collector.collect()
