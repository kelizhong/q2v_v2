# coding=utf-8
# pylint: disable=too-many-arguments, arguments-differ
"""worker to parse the raw data from ventilitor"""
from multiprocessing import Process
import logbook as logging
import pickle
# pylint: disable=ungrouped-imports
import zmq
from utils.decorator_util import memoized
from utils.data_util import tokenize, sentence_to_padding_tokens
from utils.data_util import load_vocabulary_from_pickle
from common.constant import special_words
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream
from zmq.decorators import socket


class AksisParserWorker(Process):
    """Parser worker to tokenzie the aksis data and convert them to id

    Parameters
    ----------
        vocabulary_path: str
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

    def __init__(self, ip, vocabulary_path, top_words, frontend_port=5556, backend_port=5557,
                 name="AksisWorkerProcess"):
        Process.__init__(self)
        # pylint: disable=invalid-name
        self.ip = ip
        self.vocabulary_path = vocabulary_path
        self.top_words = top_words
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

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
            source, target, label = pickle.loads(msg[0])
            source_len, source_tokens = sentence_to_padding_tokens(source, self.vocabulary, 55, tokenize)
            target_len, target_tokens = sentence_to_padding_tokens(target, self.vocabulary, 55, tokenize)

            if source_len == 0 or target_len == 0:
                return

            sender.send_pyobj((source_tokens, source_len, target_tokens, target_len, label))
        pull_stream.on_recv(_on_recv)
        loop.start()

    @property
    @memoized
    def vocabulary(self):
        """load vocabulary"""
        logging.info("loading vocabulary for process {}", self.name)
        vocab = load_vocabulary_from_pickle(self.vocabulary_path, top_words=self.top_words,
                                            special_words=special_words)
        return vocab

