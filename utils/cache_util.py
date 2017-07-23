import sys
import random


class RandomSet(object):
    def __init__(self, capacity=sys.maxsize):
        self.capacity = capacity
        self.set = set()
        self._list = None

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set.add(item)
        if self.__len__() > self.capacity:
            self.pop()

    def update(self, second_set):
        for ele in second_set:
            self.add(ele)

    def get_n_items(self):
        if self._list is None:
            self._list = list(self.set)

        return self._list[random.randint(0,len(self._list)-1)]
        # return random.sample(self.set, n if self.__len__() > n else self.__len__())

    def pop(self):
        elem = self.get_n_items()
        if len(elem) > 0:
            self.set.remove(elem[0])
            return elem[0]
        return None

    def remove(self, ele):
        self.set.remove(ele)
