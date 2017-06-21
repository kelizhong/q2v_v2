"""exception about file.e.g. not find files resource"""


class FileNotFoundError(Exception):
    def __init__(self, err="no files are found"):
        Exception.__init__(self, err)
