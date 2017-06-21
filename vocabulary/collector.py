# coding=utf-8
"""collect the tokenized sentence from worker"""
import logbook as logging
import zmq
from utils.appmetric_util import with_meter
from utils.retry_util import retry


class CollectorProcess(object):
    """collect the tokenized sentence from worker
    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        port:
            The port to receive the tokenized sentence from worker
        tries: int
            Number of times to retry, set to 0 to disable retry
    """

    def __init__(self, ip, port, tries=20):
        self.ip = ip
        self.port = port
        self.tries = tries

    @retry(lambda x: x.tries, exception=zmq.ZMQError,
           name="vocabulary_collector", report=logging.error)
    @with_meter('vocabulary_collector', interval=30)
    def _on_recv(self, receiver):
        words = receiver.recv_pyobj(zmq.NOBLOCK)
        return words

    def collect(self):
        """Generator that receive the tokenized sentence from worker and produce the words"""
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind("tcp://{}:{}".format(self.ip, self.port))
        while True:
            try:
                words = self._on_recv(receiver)
            except zmq.ZMQError as e:
                logging.error(e)
                break
            for word in words:
                if len(word):
                    yield word
