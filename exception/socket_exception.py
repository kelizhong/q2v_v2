# -*- coding: utf-8 -*-
class PortNotFoundException(Exception):
    def __init__(self, err="socket port are found"):
        Exception.__init__(self, err)
