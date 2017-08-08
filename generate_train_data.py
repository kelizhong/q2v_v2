import time
import argparse
import logging.config
from multiprocessing import Value
import yaml
from utils.config_decouple import config
from utils.network_util import local_ip
from argparser.customArgAction import AppendTupleWithoutDefault
from data_pipeline.raw_data_broker import RawDataBroker
from data_pipeline.data_parser_worker import DataParserWorker
from data_pipeline.raw_data_ventilator import DataVentilatorProcess
from data_pipeline.data_collector import DataCollectorProcess


def parse_args():
    parser = argparse.ArgumentParser(description='Generate train data')

    parser.add_argument('-fp', '--file-patterns', nargs=1, action=AppendTupleWithoutDefault,
                        default=['*add', '*search', '*click', '*purchase'])
    parser.add_argument('--ip', type=str, help='ip address', default=None)
    parser.add_argument('--port', type=str, help='zmq port')
    parser.add_argument('-w', '--worker-num', default=4, type=int,
                        help='number of parser worker')
    parser.add_argument('--rawdata-frontend-port', default=5555, type=int,
                        help='train rawdata frontend port')
    parser.add_argument('--rawdata-backend-port', default=5556, type=int,
                        help='train rawdata backend port')
    parser.add_argument('--collector-frontend-port', default=5557, type=int,
                        help='collector frontend port')

    return parser.parse_args()


def start_data_ventilator_process(ip, port, file_patterns, build_item_dict_status):
    """start the ventilator process which read the corpus data"""
    for i, file_pattern in enumerate(file_patterns):
        ventilator = DataVentilatorProcess(file_pattern, build_item_dict_status, ip=ip, port=port, name="data_ventilator_{}".format(i))
        ventilator.daemon = True
        ventilator.start()


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
    rawdata_frontend_port : {number}, optional
        raw data broker frontend port (the default is 5555)
    rawdata_backend_port : {number}, optional
        raw data broker backend port (the default is 5556)
    collector_fronted_port : {number}, optional
        collector port (the default is 5557)
    """
    args = parse_args()
    setup_logger()
    ip = args.ip or local_ip()
    # ip = '127.0.0.1'
    # True, when finish item dict building
    build_item_dict_status = Value('d', False)
    period = 60

    start_data_ventilator_process(ip=ip, port=args.rawdata_frontend_port, file_patterns=args.file_patterns, build_item_dict_status=build_item_dict_status)

    start_raw_data_broker(ip=ip, frontend_port=args.rawdata_frontend_port, backend_port=args.rawdata_backend_port)

    start_parser_worker_process(ip=ip, frontend_port=args.rawdata_backend_port,
                                backend_port=args.collector_frontend_port, worker_num=args.worker_num)
    while not build_item_dict_status.value:
        # waiting ventiliator to build item dict
        time.sleep(period)
    collector = DataCollectorProcess(ip=ip, port=args.collector_frontend_port)
    collector.collect()
