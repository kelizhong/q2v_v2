

import contextlib
import logging
import socket
import threading

import queue

from thrift.Thrift import TException
from thrift.protocol import TBinaryProtocol, TCompactProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from external.nms.protocol import QueryService
logger = logging.getLogger(__name__)

THRIFT_TRANSPORTS = dict(
    buffered=TTransport.TBufferedTransport,
    framed=TTransport.TFramedTransport,
)
THRIFT_PROTOCOLS = dict(
    binary=TBinaryProtocol.TBinaryProtocol,
    compact=TCompactProtocol.TCompactProtocol,
)

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 9090
DEFAULT_TRANSPORT = 'buffered'
DEFAULT_PROTOCOL = 'binary'


class ThriftClientHelper:
    def __init__(self, host='127.0.0.1', port=10000, pool_size=8, retObj=False, retExternId=False):
        self.host = host
        self.port = port
        self.retObj = retObj
        self.retExternId = retExternId
        self.pool = ConnectionPool(size=pool_size, host=self.host, port=self.port)

    def find_k_nearest(self, k, queryObj):
        with self.pool.connection() as connection:
            logging.info("Running %d-NN search", k)
            res = connection.knn_query(k, queryObj, self.retObj, self.retExternId)
            res_list = list()
            for e in res:
                s = ''
                if self.retExternId:
                    s = 'externId=' + e.externId
                if self.retObj:
                    res_list.append((e.id, e.dist, e.obj))
                else:
                    res_list.append((e.id, e.dist))
            return res_list


class ConnectionPool(object):
    """
    Thread-safe connection pool.

    .. versionadded:: 0.5

    The `size` argument specifies how many connections this pool
    manages. Additional keyword arguments are passed unmodified to the
    :py:class:`happybase.Connection` constructor, with the exception of
    the `autoconnect` argument, since maintaining connections is the
    task of the pool.

    :param int size: the maximum number of concurrently open connections
    :param kwargs: keyword arguments passed to
                   :py:class:`happybase.Connection`
    """
    def __init__(self, size, **kwargs):
        if not isinstance(size, int):
            raise TypeError("Pool 'size' arg must be an integer")

        if not size > 0:
            raise ValueError("Pool 'size' arg must be greater than zero")

        logger.debug(
            "Initializing connection pool with %d connections", size)

        self._lock = threading.Lock()
        self._queue = queue.LifoQueue(maxsize=size)
        self._thread_connections = threading.local()

        connection_kwargs = kwargs
        connection_kwargs['autoconnect'] = False

        for i in range(size):
            connection = Connection(**connection_kwargs)
            self._queue.put(connection)

        # The first connection is made immediately so that trivial
        # mistakes like unresolvable host names are raised immediately.
        # Subsequent connections are connected lazily.
        with self.connection():
            pass

    def _acquire_connection(self, timeout=None):
        """Acquire a connection from the pool."""
        try:
            return self._queue.get(True, timeout)
        except queue.Empty:
            raise NoConnectionsAvailable(
                "No connection available from pool within specified "
                "timeout")

    def _return_connection(self, connection):
        """Return a connection to the pool."""
        self._queue.put(connection)

    @contextlib.contextmanager
    def connection(self, timeout=None):
        """
        Obtain a connection from the pool.

        This method *must* be used as a context manager, i.e. with
        Python's ``with`` block. Example::

            with pool.connection() as connection:
                pass  # do something with the connection

        If `timeout` is specified, this is the number of seconds to wait
        for a connection to become available before
        :py:exc:`NoConnectionsAvailable` is raised. If omitted, this
        method waits forever for a connection to become available.

        :param int timeout: number of seconds to wait (optional)
        :return: active connection from the pool
        :rtype: :py:class:`happybase.Connection`
        """

        connection = getattr(self._thread_connections, 'current', None)

        return_after_use = False
        if connection is None:
            # This is the outermost connection requests for this thread.
            # Obtain a new connection from the pool and keep a reference
            # in a thread local so that nested connection requests from
            # the same thread can return the same connection instance.
            #
            # Note: this code acquires a lock before assigning to the
            # thread local; see
            # http://emptysquare.net/blog/another-thing-about-pythons-
            # threadlocals/
            return_after_use = True
            connection = self._acquire_connection(timeout)
            with self._lock:
                self._thread_connections.current = connection

        try:
            # Open connection, because connections are opened lazily.
            # This is a no-op for connections that are already open.
            connection.open()

            # Return value from the context manager's __enter__()
            yield connection

        except (TException, socket.error):
            # Refresh the underlying Thrift client if an exception
            # occurred in the Thrift layer, since we don't know whether
            # the connection is still usable.
            logger.info("Replacing tainted pool connection")
            connection._refresh_thrift_client()
            connection.open()

            # Reraise to caller; see contextlib.contextmanager() docs
            raise

        finally:
            # Remove thread local reference after the outermost 'with'
            # block ends. Afterwards the thread no longer owns the
            # connection.
            if return_after_use:
                del self._thread_connections.current
                self._return_connection(connection)


class NoConnectionsAvailable(RuntimeError):
    """
    Exception raised when no connections are available.

    This happens if a timeout was specified when obtaining a connection,
    and no connection became available within the specified timeout.

    .. versionadded:: 0.5
    """
    pass


class Connection(object):
    """Connection to an HBase Thrift server.

    The `host` and `port` arguments specify the host name and TCP port
    of the HBase Thrift server to connect to. If omitted or ``None``,
    a connection to the default port on ``localhost`` is made. If
    specifed, the `timeout` argument specifies the socket timeout in
    milliseconds.

    If `autoconnect` is `True` (the default) the connection is made
    directly, otherwise :py:meth:`Connection.open` must be called
    explicitly before first use.

    The optional `table_prefix` and `table_prefix_separator` arguments
    specify a prefix and a separator string to be prepended to all table
    names, e.g. when :py:meth:`Connection.table` is invoked. For
    example, if `table_prefix` is ``myproject``, all tables tables will
    have names like ``myproject_XYZ``.

    The optional `compat` argument sets the compatibility level for
    this connection. Older HBase versions have slightly different Thrift
    interfaces, and using the wrong protocol can lead to crashes caused
    by communication errors, so make sure to use the correct one. This
    value can be either the string ``0.90``, ``0.92``, ``0.94``, or
    ``0.96`` (the default).

    The optional `transport` argument specifies the Thrift transport
    mode to use. Supported values for this argument are ``buffered``
    (the default) and ``framed``. Make sure to choose the right one,
    since otherwise you might see non-obvious connection errors or
    program hangs when making a connection. HBase versions before 0.94
    always use the buffered transport. Starting with HBase 0.94, the
    Thrift server optionally uses a framed transport, depending on the
    argument passed to the ``hbase-daemon.sh start thrift`` command.
    The default ``-threadpool`` mode uses the buffered transport; the
    ``-hsha``, ``-nonblocking``, and ``-threadedselector`` modes use the
    framed transport.

    The optional `protocol` argument specifies the Thrift transport
    protocol to use. Supported values for this argument are ``binary``
    (the default) and ``compact``. Make sure to choose the right one,
    since otherwise you might see non-obvious connection errors or
    program hangs when making a connection. ``TCompactProtocol`` is
    a more compact binary format that is  typically more efficient to
    process as well. ``TBinaryProtocol`` is the default protocol that
    Happybase uses.

    .. versionadded:: 0.9
       `protocol` argument

    .. versionadded:: 0.5
       `timeout` argument

    .. versionadded:: 0.4
       `table_prefix_separator` argument

    .. versionadded:: 0.4
       support for framed Thrift transports

    :param str host: The host to connect to
    :param int port: The port to connect to
    :param int timeout: The socket timeout in milliseconds (optional)
    :param bool autoconnect: Whether the connection should be opened directly
    :param str table_prefix: Prefix used to construct table names (optional)
    :param str table_prefix_separator: Separator used for `table_prefix`
    :param str compat: Compatibility mode (optional)
    :param str transport: Thrift transport mode (optional)
    """
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=None,
                 autoconnect=True, table_prefix=None,
                 table_prefix_separator=b'_',
                 transport=DEFAULT_TRANSPORT, protocol=DEFAULT_PROTOCOL):

        if transport not in THRIFT_TRANSPORTS:
            raise ValueError("'transport' must be one of %s"
                             % ", ".join(THRIFT_TRANSPORTS.keys()))

        if protocol not in THRIFT_PROTOCOLS:
            raise ValueError("'protocol' must be one of %s"
                             % ", ".join(THRIFT_PROTOCOLS))

        # Allow host and port to be None, which may be easier for
        # applications wrapping a Connection instance.
        self.host = host or DEFAULT_HOST
        self.port = port or DEFAULT_PORT
        self.timeout = timeout
        self.table_prefix = table_prefix
        self.table_prefix_separator = table_prefix_separator

        self._transport_class = THRIFT_TRANSPORTS[transport]
        self._protocol_class = THRIFT_PROTOCOLS[protocol]
        self._refresh_thrift_client()

        if autoconnect:
            self.open()

        self._initialized = True

    def _refresh_thrift_client(self):
        """Refresh the Thrift socket, transport, and client."""
        # Make socket
        socket = TSocket.TSocket(host=self.host, port=self.port)
        socket.setTimeout(self.timeout)

        # Buffering is critical. Raw sockets are very slow
        self.transport = self._transport_class(socket)
        # Wrap in a protocol
        protocol = self._protocol_class(self.transport)

        self.client = QueryService.Client(protocol)

    def open(self):
        """Open the underlying transport to the HBase instance.

        This method opens the underlying Thrift transport (TCP connection).
        """
        if self.transport.isOpen():
            return

        logger.debug("Opening Thrift transport to %s:%d", self.host, self.port)
        self.transport.open()

    def close(self):
        """Close the underyling transport to the HBase instance.

        This method closes the underlying Thrift transport (TCP connection).
        """
        if not self.transport.isOpen():
            return

        if logger is not None:
            # If called from __del__(), module variables may no longer
            # exist.
            logger.debug(
                "Closing Thrift transport to %s:%d",
                self.host, self.port)

        self.transport.close()

    def __del__(self):
        try:
            self._initialized
        except AttributeError:
            # Failure from constructor
            return
        else:
            self.close()

    def knn_query(self, k, query, return_object=False, return_external_id=False):
        res = self.client.knnQuery(k, query, return_object, return_external_id)
        return res

