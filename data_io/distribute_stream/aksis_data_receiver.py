# coding=utf-8
"""receive the data from collector for training model"""
from __future__ import print_function
import zmq
from utils.network_util import local_ip
from utils.appmetric_util import with_meter


class AksisDataReceiver(object):
    """Receiver the data from collector

    Parameters
    ----------
        ip : str
            The ip address string without the port to pass to ``Socket.bind()``.
        port: int
            Port to receive the data from collector
        stop_freq: int
            Frequency to raise the StopIteration error. If the trainer iter receive
            the StopIteration, the trainer will save the checkpoint. If stop_freq < 0,
            will not raise the StopIteration error
    """
    def __init__(self, ip=None, port=5556, stop_freq=-1):
        context = zmq.Context()
        # pylint: disable=no-member
        ip = ip or local_ip()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://{}:{}".format(ip, port))
        self.num = 0
        self.stop_freq = stop_freq

    def __iter__(self):
        return self

    @with_meter('aksis_data_receiver', interval=30)
    def next(self):
        """return the data from collector"""
        if 0 < self.stop_freq < self.num:
            raise StopIteration
        data = self.receiver.recv_pyobj()
        self.num += 1
        return data

    def reset(self):
        """reset the num to zero"""
        self.num = 0

if __name__ == '__main__':
    # just for test
    # pylint: disable=invalid-name
    receiver = AksisDataReceiver(local_ip(), port=5558)
    for x in receiver:
        print(x[1])
