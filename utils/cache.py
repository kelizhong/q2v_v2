from sets import Set
import random


class RandomSet(object):
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.set = Set()

    def __len__(self):
        return len(self.set)

    def add(self, item):
        if self.__len__() > self.capacity:
            self.pop()
        self.set.add(item)

    def get_n_items(self, n):
        return random.sample(self.set, n if self.__len__() > n else self.__len__())

    def pop(self):
        elem = self.get_n_items(1)
        if len(elem) > 0:
            self.set.remove(elem[0])
