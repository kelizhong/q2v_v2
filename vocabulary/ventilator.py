# coding=utf-8
"""Ventilator process to read the data from sentence_gen generator"""
from __future__ import print_function
import logbook as logging
from multiprocessing import Process
from collections import Counter
import zmq
from zmq.decorators import socket
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from utils.data_util import sentence_gen


class VentilatorProcess(Process):
    """process to read the data from sentence_gen generator
    Parameters
    ----------
        corpus_files: list or str
            corpus file paths, convert it to list for str type
        ip: str
            the ip address string without the port to pass to ``Socket.bind()``.
        port: int
            port for s]the sender socket
        sentence_gen: generator
            generator which produce the sentence in corpus data
        name: str
            process name
    """

    def __init__(self, corpus_files, ip, port, sentence_gen=sentence_gen, name='VentilatorProcess'):
        Process.__init__(self)
        self.ip = ip
        self.port = port
        self.corpus_files = [corpus_files] if not isinstance(corpus_files, list) else corpus_files
        self.sentence_gen = sentence_gen
        self.name = name

    @socket(zmq.PUSH)
    def run(self, sender):
        """read the sentence from sentence generator and send to the worker"""
        sender.bind("tcp://{}:{}".format(self.ip, self.port))

        logging.info("start sentence producer {}", self.name)
        for filename in self.corpus_files:
            logging.info("Counting words in {}", filename)
            for sentence in self.sentence_gen(filename):
                sender.send_string(sentence)


if __name__ == '__main__':
    """for test"""
    v = VentilatorProcess('../data/query2vec/train_corpus/search.keyword.enc', '127.0.0.1', '5555')
    for _ in xrange(8):
        w = WorkerProcess('127.0.0.1', '5555', '5556')
        w.start()
    c = CollectorProcess('127.0.0.1', '5556')
    v.start()
    counter = Counter(c.collect())

    print(v.is_alive())
