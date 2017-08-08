# coding=utf-8
# pylint: disable=too-many-arguments, arguments-differ
"""worker to parse the raw data from ventilator"""
from multiprocessing import Process
import logging
import pickle
# pylint: disable=ungrouped-imports
import zmq
from zmq.eventloop import ioloop
from zmq.eventloop.zmqstream import ZMQStream
from zmq.decorators import socket
from helper.tokenizer_helper import TextBlobTokenizerHelper
from utils.data_util import is_number


logger = logging.getLogger(__name__)


class DataParserWorker(Process):
    """Parser worker to tokenize the aksis data and convert them to id

    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        frontend_port: int
            Port for the incoming traffic
        backend_port: int
            Port for the outbound traffic
    """

    def __init__(self, ip='127.0.0.1', frontend_port=5556, backend_port=5557,
                 name="DataWorkerProcess"):
        Process.__init__(self)
        self.logger = logging.getLogger("data")
        # pylint: disable=invalid-name
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name

    # pylint: disable=no-member
    @socket(zmq.PULL)
    @socket(zmq.PUSH)
    def run(self, receiver, sender):
        receiver.connect("tcp://{}:{}".format(self.ip, self.frontend_port))
        sender.connect("tcp://{}:{}".format(self.ip, self.backend_port))
        logger.info("process %s connect %s:%d and start parse data", self.name, self.ip, self.frontend_port)
        ioloop.install()
        loop = ioloop.IOLoop.instance()
        pull_stream = ZMQStream(receiver, loop)
        tokenizer = TextBlobTokenizerHelper()

        def _on_recv(msg):
            try:
                data_item = pickle.loads(msg[0])
                parsed_data = map(lambda data: data if is_number(data) else ' '.join(tokenizer.tokenize(data)),
                                  data_item)
                data_item_str = '\t'.join(data_item) + '\n'
                parsed_data_str = '\t'.join(parsed_data) + '\n'
                sender.send_pyobj((data_item_str, parsed_data_str))
            except Exception as e:
                logger.error("%s failed to load msg.", self.name, exc_info=True, stack_info=True)

        pull_stream.on_recv(_on_recv)
        loop.start()
