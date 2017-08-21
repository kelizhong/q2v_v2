# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, with_statement
import contextlib
import socket
import errno
import random
import itertools
from exception.socket_exception import PortNotFoundException


def select_random_port(ports=None, exclude_ports=None):
    """
    Returns random unused port number.
    """
    if ports is None:
        ports = available_ports(exclude_ports=exclude_ports)

    for port in random.sample(ports, min(len(ports), 100)):
        if not port_is_used(port):
            return port
    raise PortNotFoundException("Can't select a port")


def is_available(port):
    """
    Returns if port is good to choose.
    """
    return port in available_ports() and not port_is_used(port)


def available_ports(low=1024, high=65535, exclude_ports=None):
    """
    Returns a set of possible ports (low-high).
    Pass ``high`` and/or ``low`` to limit the port range.
    """
    if exclude_ports is None:
        exclude_ports = []
    available = set(itertools.chain(range(low, high)))
    return available.difference(exclude_ports)


def port_is_used(port, host='127.0.0.1'):
    """
    Returns if port is used. Port is considered used if the current process
    can't bind to it or the port doesn't refuse connections.
    """
    unused = _can_bind(port, host) and _refuses_connection(port, host)
    return not unused


def _can_bind(port, host):
    sock = socket.socket()
    with contextlib.closing(sock):
        try:
            sock.bind((host, port))
        except socket.error:
            return False
    return True


def _refuses_connection(port, host):
    sock = socket.socket()
    with contextlib.closing(sock):
        sock.settimeout(1)
        err = sock.connect_ex((host, port))
        return err == errno.ECONNREFUSED
