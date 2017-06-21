# coding=utf-8
"""log util"""
import sys
import logbook
import signal
from logbook.compat import redirect_logging
from utils.file_util import ensure_dir_exists


class Logger(object):
    def __init__(self, date_format='%Y-%m-%d', level=logbook.DEBUG,
                 format_string='{record.time:%Y-%m-%d %H:%M:%S}|{record.level_name}|{record.message}'):
        self.format_string = format_string
        self.date_format = date_format
        self.level = level
        # all logging calls that happen after this call will transparently be redirected to Logbook
        redirect_logging()

    def set_stream_handler(self, level=None, format_string=None, bubble=True):
        """set a handler that writes to what is currently at stderr.
        Parameters
        ----------
        level: logbook level
            the level for the handler.
        format_string: str
            string format for msg
        bubble: Boolean
            By default messages sent to that handler will not go to a handler on
            an outer level on the stack, if handled.  This can be changed by
            setting bubbling to `True`.
        """
        level = level or self.level
        format_string = format_string or self.format_string
        handler = logbook.StderrHandler(level=level, bubble=bubble)
        handler.formatter.format_string = format_string
        handler.push_application()
        return self

    def set_time_rotating_file_handler(self, file_name, date_format=None, level=None, format_string=None, bubble=True,
                                       backup_count=10):
        """set a handler that rotates based on dates.  It will name the file
        after the filename you specify and the `date_format` pattern.
        So for example if you configure your handler like this::

        set_time_rotating_file_handler('/var/log/foo.log',
                                           date_format='%Y-%m-%d', ...)

        The file_name for the logfiles will look like this::

        /var/log/foo-2010-01-10.log
        /var/log/foo-2010-01-11.log
        ...

        By default it will keep all these files around, if you want to limit
        them, you can specify a `backup_count`.
        Parameters
        ----------
        file_name: str:
            log file name
        date_format: str
            date for file name
        level: logbook level
            the level for the handler.
        format_string: str
            string format for msg
        bubble: Boolean
            By default messages sent to that handler will not go to a handler on
            an outer level on the stack, if handled.  This can be changed by
            setting bubbling to `True`.
        backup_count: int
            the number of backup log files
        """
        ensure_dir_exists(file_name, is_dir=False)
        level = level or self.level
        date_format = date_format or self.date_format
        format_string = format_string or self.format_string
        handler = logbook.TimedRotatingFileHandler(file_name, level=level, bubble=bubble, date_format=date_format,
                                                   backup_count=backup_count)
        handler.formatter.format_string = format_string
        handler.push_application()


def signal_handler(signal, frame):
    logbook.info('Stop!!!')
    sys.exit(0)


def setup_logger(log_file_name):
    log = Logger()
    log.set_stream_handler()
    log.set_time_rotating_file_handler(log_file_name)
    signal.signal(signal.SIGINT, signal_handler)
