"""exception about resource.e.g. not find resource"""


class ResourceNotFoundError(Exception):

    def __init__(self, err="no resource are found"):
        Exception.__init__(self, err)
