# coding=utf-8
# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments
"""aksis data pieline to start the raw_data_broker, data_ventilator_process,
parser_worker_process and collector_process"""
from utils.network_util import local_ip
from data_io.distribute_stream.aksis_raw_data_broker import AksisRawDataBroker
from data_io.distribute_stream.aksis_parser_worker import AksisParserWorker
from data_io.distribute_stream.aksis_ventilator import AksisDataVentilatorProcess
from data_io.distribute_stream.aksis_data_collector import AksisDataCollector


class AksisDataPipeline(object):
    """Start the raw_data_broker, data_ventilator_process, parser_worker_process
    and collector_process. The data pipeline is data_ventilator_process->raw_data_broker
    ->parser_worker_process->collector_process

    Parameters
    ----------
        vocabulary_path: str
            Path for vocabulary from aksis corpus data
        top_words: int
            Only use the top_words in vocabulary
        file_patterns: list
            file pattern use to distinguish different corpus, every file pattern will start a
            ventilator process.
            e.g. there are four action type(KeywordsByAdds, KeywordsBySearches, KeywordsByPurchases,
            KeywordsByClicks) in aksis data, if split the aksis data to four files, like aksis.add,
            aksis.search, aksis.purchase and aksis.click, each file store the corresponding data,
            than can use these four patterns(*add, *search, *purchase, *click) to read the related
            file
        batch_size: int
            Batch size for each databatch
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        worker_num: int
            number of parser worker which tokenize the sentence and convert the sentence to id
        rawdata_frontend_port: int
            Port for the incoming traffic of ventilator which produce the raw data
        rawdata_backend_port: int
            Port for the outbound traffic of ventilator which produce the raw data
        collector_fronted_port: int
            Port for the incoming traffic of collector which collect the data from worker
        collector_backend_port: int
            Port for the outbound traffic of collector which collect the data from worker
        num_epoch: int
            end epoch of producing the data
    Notes
    -----
        All the processes except the ventilator process can not be terminated automatically
        since the trainer is not planned to stop. If feed the data to the trainer, the train
        will continue to train.
        Send the CTRL+C signal will stop all the processes
    """

    def __init__(self, vocabulary_path, top_words, file_patterns, batch_size,
                 worker_num=1, ip=None, num_epoch=65535, words_list_file=None,
                 rawdata_frontend_port=5555, rawdata_backend_port=5556,
                 collector_fronted_port=5557, collector_backend_port=5558):
        self.vocabulary_path = vocabulary_path
        self.top_words = top_words
        self.ip = ip or local_ip()
        self.rawdata_frontend_port = rawdata_frontend_port
        self.rawdata_backend_port = rawdata_backend_port
        self.collector_fronted_port = collector_fronted_port
        self.collector_backend_port = collector_backend_port
        self.file_patterns = file_patterns
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.worker_num = worker_num
        self.words_list_file = words_list_file

    def start_collector_process(self, join=False):
        """start the collector process which collect the data from parser worker"""
        collector = AksisDataCollector(self.ip, self.batch_size,
                                       frontend_port=self.collector_fronted_port,
                                       backend_port=self.collector_backend_port)
        collector.start()
        # all process will not be terminated util receive the CTRL+C signal
        # TODO add pub-sub pattern to send the command to stop the process
        if join:
            collector.join()

    def start_parser_worker_process(self):
        """start the parser worker process which tokenize the copus data and convert them to id"""
        for i in range(self.worker_num):
            worker = AksisParserWorker(self.ip, self.vocabulary_path, self.top_words, batch_size=self.batch_size,
                                       frontend_port=self.rawdata_backend_port,
                                       backend_port=self.collector_fronted_port,
                                       words_list_file=self.words_list_file,
                                       first_worker=(i==0),
                                       name="aksis_parser_worker_{}".format(i))
            worker.start()

    def start_data_ventilator_process(self):
        """start the ventilator process which read the corpus data"""
        for i, (file_pattern, dropout) in enumerate(self.file_patterns):
            ventilator = AksisDataVentilatorProcess(file_pattern, dropout=dropout,
                                                    ip=self.ip,
                                                    port=self.rawdata_frontend_port,
                                                    name="aksis_ventilator_{}".format(i))
            ventilator.start()

    def start_raw_data_broker(self):
        """start the raw data broker between ventilator and parser worker process"""
        # TODO bind the random port not use the defined port
        raw_data_broker = AksisRawDataBroker(self.ip, self.rawdata_frontend_port,
                                             self.rawdata_backend_port)
        raw_data_broker.start()

    def start_all(self):
        """start all the process"""
        self.start_raw_data_broker()
        self.start_data_ventilator_process()
        self.start_parser_worker_process()
        self.start_collector_process(join=True)
