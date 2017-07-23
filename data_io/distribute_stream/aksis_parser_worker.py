# coding=utf-8
# pylint: disable=too-many-arguments, arguments-differ
"""worker to parse the raw data from ventilator"""
from multiprocessing import Process
import logbook as logging
import pickle
# pylint: disable=ungrouped-imports
import zmq
from utils.decorator_util import memoized
from vocabulary.vocab import VocabularyFromWordList
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream
from zmq.decorators import socket
from ..batch_data_handler import BatchDataTrigramHandler
from config.config import special_words


class AksisParserWorker(Process):
    """Parser worker to tokenize the aksis data and convert them to id

    Parameters
    ----------
        vocabulary_data_dir: str
            Path for vocabulary from aksis corpus data
        top_words: int
            Only use the top_words in vocabulary
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        frontend_port: int
            Port for the incoming traffic
        backend_port: int
            Port for the outbound traffic
    """

    def __init__(self, ip, vocabulary_data_dir, top_words, batch_size, words_list_file=None, source_maxlen=30,
                 target_maxlen=100, frontend_port=5556, backend_port=5557, first_worker=0,
                 name="AksisWorkerProcess"):
        Process.__init__(self)
        # pylint: disable=invalid-name
        self.ip = ip
        self.vocabulary_data_dir = vocabulary_data_dir
        self.top_words = top_words
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name
        self.batch_size = batch_size
        self.source_maxlen = source_maxlen
        self.target_maxlen = target_maxlen
        self.words_list_file = words_list_file if first_worker else None
        self.batch_data = BatchDataTrigramHandler(self.vocabulary, batch_size)

    # pylint: disable=no-member
    @socket(zmq.PULL)
    @socket(zmq.PUSH)
    def run(self, receiver, sender):
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender.connect("tcp://{}:{}".format(self.ip, self.backend_port))
        logging.info("process {} connect {}:{} and start parse data", self.name, self.ip,
                     self.frontend_port)
        ioloop.install()
        loop = ioloop.IOLoop.instance()
        pull_stream = ZMQStream(receiver, loop)

        def _on_recv(msg):
            try:
                source, target, label = pickle.loads(msg[0])
                self.batch_data.parse_and_insert_data_object(source, target, label)
                if self.batch_data.data_object_length == self.batch_size:
                    sender.send_pyobj(self.batch_data.data_object)
            except Exception as e:
                logging.info("{} failed to load msg. Error: {}", self.name, e)

        pull_stream.on_recv(_on_recv)
        loop.start()

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        vocab = VocabularyFromWordList(self.vocabulary_data_dir, special_words=special_words,
                                       top_words=self.top_words).build_vocabulary_from_words_list(self.words_list_file)
        return vocab
