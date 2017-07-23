# coding=utf-8
"""Ventilator process to read the data from sentence_gen generator"""
from __future__ import print_function
import logbook as logging
import glob
from multiprocessing import Process
from collections import Counter
import zmq
from zmq.decorators import socket
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from utils.data_util import sentence_gen
from utils.pickle_util import save_obj_pickle


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
            for num, sentence in enumerate(self.sentence_gen(filename)):
                if num % 10000 == 0:
                    print(num)
                sender.send_string(sentence)


if __name__ == '__main__':
    """for test"""
    files = glob.glob("/Users/keliz/Downloads/aksis.purchased.pair/part*")
    v = VentilatorProcess(files, '127.0.0.1', '5555')
    for _ in range(8):
        w = WorkerProcess('127.0.0.1', '5555', '5556')
        w.start()
    c = CollectorProcess('127.0.0.1', '5556')
    v.start()
    counter = Counter(c.collect())
    save_obj_pickle(counter, './counter.pkl', overwrite=True)
    print(v.is_alive())
