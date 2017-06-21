# coding=utf-8
"""util for network operation. e.g. get ip"""
import socket


def local_ip():
    """get the IP address (a string of the form '255.255.255.255') for a host
    Returns
    -------
    ip: host ip
        return "127.0.0.1" if fail to get ip
    """
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except:
        ip = "127.0.0.1"
    return ip
