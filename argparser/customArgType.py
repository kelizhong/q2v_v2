# coding: utf-8
# pylint: disable=invalid-name
"""custom argparse type"""
import sys
import os
import errno


def IntegerType(value):
    """convert str to int and convert 'inf' to sys.maxsize"""
    return sys.maxsize if value == 'inf' else int(value)


def DirectoryType(value):
    """directory type, create directory if not exist"""
    if not os.path.exists(value):
        os.makedirs(value)
    return value


def FileType(value):
    """file type, create file if not exist"""
    if not os.path.exists(os.path.dirname(value)):
        try:
            os.makedirs(os.path.dirname(value))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return value
