# coding=utf-8
"""Broker between ventilator process and worker process"""
import logging
import zmq
from zmq.devices import ProcessDevice


class RawDataBroker(object):

    def __init__(self, ip='127.0.0.1', frontend_port=5555, backend_port=5556, name="RawDataBroker"):
        """Broker between ventilator process and worker process

        Parameters
        ----------
        ip : {str}, optional
            The ip address string without the port to pass to ``Socket.bind()``.
            (the default is '127.0.0.1')
        frontend_port : {number}, optional
            Port for the incoming traffic
            (the default is 5555)
        backend_port : {number}, optional
            Port for the outbound traffic
            (the default is 5556)
        name : {str}, optional
            Broker process name (the default is "RawDataBroker")
        """
        self.ip = ip
        self.frontend_port = frontend_port
        self.backend_port = backend_port
        self.name = name
        self.logger = logging.getLogger("data")

    def run(self):
        """start device that will be run in a background Process."""
        # pylint: disable=no-member
        dev = ProcessDevice(zmq.STREAMER, zmq.PULL, zmq.PUSH)
        dev.bind_in("tcp://{}:{}".format(self.ip, self.frontend_port))
        dev.bind_out("tcp://{}:{}".format(self.ip, self.backend_port))
        dev.setsockopt_in(zmq.IDENTITY, b'PULL')
        dev.setsockopt_out(zmq.IDENTITY, b'PUSH')
        dev.start()
        self.logger.info(
            "start broker %s, ip:%s, frontend port:%d, backend port:%d",
            self.name, self.ip, self.frontend_port, self.backend_port)

    def start(self):
        """start the processDevice process"""
        self.run()
