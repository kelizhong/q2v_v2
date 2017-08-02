# -*- coding:utf-8 -*-

from __future__ import absolute_import, unicode_literals, print_function

import logging
import time

from grpc import StatusCode
from grpc._channel import _Rendezvous, _UnaryUnaryMultiCallable
from grpc.beta import implementations

from utils.config_decouple import config
from external.tf_serving.protocol import prediction_service_pb2

logger = logging.getLogger(__name__)

# The maximum number of retries
_MAX_RETRIES_BY_CODE = {
    StatusCode.INTERNAL: config('gprc_retry_internal', section='gprc_retry'),
    StatusCode.ABORTED: config('gprc_retry_aborted', section='gprc_retry'),
    StatusCode.UNAVAILABLE: config('gprc_retry_unavailable', section='gprc_retry'),
    StatusCode.DEADLINE_EXCEEDED: config('gprc_retry_deadline_exceeded', section='gprc_retry'),
}

# The minimum seconds (float) of sleeping
_MIN_SLEEPING = config('gprc_retry_min_sleeping', section='gprc_retry')
_MAX_SLEEPING = config('gprc_retry_max_sleeping', section='gprc_retry')


class TFServingClientHelper:
    def __init__(self, host='localhost', port='9000'):
        self.host = host
        self.port = port
        self._init_predict_service_stub()

    def _init_predict_service_stub(self):
        # Create gRPC client and request
        channel = implementations.insecure_channel(self.host, self.port)
        self.predict_stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        retrying_stub_methods(self.predict_stub)

    def predict(self, request, request_timeout=10):
        # Send request
        result = self.predict_stub.Predict(request, request_timeout)
        return result


class RetriesExceeded(Exception):
    """docstring for RetriesExceeded"""
    pass


def retry(f, transactional=False):
    def wraps(*args, **kwargs):
        retries = 0
        while True:
            try:
                return f(*args, **kwargs)
            except _Rendezvous as e:
                code = e.code()

                max_retries = _MAX_RETRIES_BY_CODE.get(code)
                if max_retries is None or transactional and code == StatusCode.ABORTED:
                    raise

                if retries > max_retries:
                    raise RetriesExceeded(e)

                backoff = min(_MIN_SLEEPING * 2 ** retries, _MAX_SLEEPING)
                logger.debug("sleeping %r for %r before retrying failed request...", backoff, code)

                retries += 1
                time.sleep(backoff)

    return wraps


def retrying_stub_methods(obj):
    for key, attr in obj.__dict__.items():
        if isinstance(attr, _UnaryUnaryMultiCallable):
            setattr(obj, key, retry(attr))