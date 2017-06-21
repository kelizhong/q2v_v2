# coding=utf-8
"""Generate vocabulary"""
import logbook as logging
from collections import Counter
from utils.pickle_util import save_obj_pickle
from vocabulary.ventilator import VentilatorProcess
from vocabulary.worker import WorkerProcess
from vocabulary.collector import CollectorProcess
from utils.data_util import sentence_gen


class Vocab(object):
    """
    Create vocabulary file (if it does not exist yet) from data file.
    Data file should have one sentence per line.
    Each sentence will be tokenized.
    Vocabulary contains the most-frequent tokens up to top_words.
    We write it to vocab_file in pickle format.
    Parameters
    ----------
        corpus_files: list
            corpus files list that will be used to create vocabulary
        vocab_save_path: str
            vocab file name where the vocabulary will be created
        sentence_gen: generator
            generator which produce the sentence in corpus data
        top_words: int
            limit on the size of the created vocabulary
        workers_num: int
            numbers of workers to parse the sentence
        ip: str
            the ip address string without the port to pass to ``Socket.bind()``
        ventilator_port: int
            port for ventilator process socket
        collector_port: int
            port for collector process socket
        overwrite: bool
            whether to overwrite the existed vocabulary
    """

    def __init__(self, corpus_files, vocab_save_path, sentence_gen=sentence_gen, workers_num=1, top_words=100000,
                 ip='127.0.0.1', ventilator_port='5555', collector_port='5556',
                  overwrite=True):
        self.corpus_files = corpus_files
        self.vocab_save_path = vocab_save_path
        self.sentence_gen = sentence_gen
        self.workers_num = workers_num
        self.top_words = top_words
        self.ip = ip
        self.ventilator_port = ventilator_port
        self.collector_port = collector_port
        self.overwrite = overwrite

    def create_dictionary(self):
        process_pool = []
        v = VentilatorProcess(self.corpus_files, self.ip, self.ventilator_port, sentence_gen=self.sentence_gen)
        v.start()
        process_pool.append(v)
        for i in xrange(self.workers_num):
            w = WorkerProcess(self.ip, self.ventilator_port, self.collector_port, name='WorkerProcess_{}'.format(i))
            w.start()
            process_pool.append(w)
        c = CollectorProcess(self.ip, self.collector_port)
        counter = Counter(c.collect())
        self._terminate_process(process_pool)
        logging.info("Finish counting. {} unique words, a total of {} words in all files."
                     , len(counter), sum(counter.values()))

        counter = counter.most_common(self.top_words)
        logging.info("store vocabulary with most_common_words file, vocabulary size: {}", len(counter))
        save_obj_pickle(counter, self.vocab_save_path, self.overwrite)

    def _terminate_process(self, pool):
        for p in pool:
            p.terminate()
            logging.info('terminated process {}', p.name)
