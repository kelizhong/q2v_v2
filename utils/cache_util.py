import sys
import random


class RandomSet(object):
    def __init__(self, capacity=sys.maxsize):
        self.capacity = capacity
        self.set = set()

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set.add(item)
        if self.__len__() > self.capacity:
            self.pop()

    def update(self, second_set):
        for ele in second_set:
            self.add(ele)

    def get_n_items(self, n):
        return random.sample(self.set, n if self.__len__() > n else self.__len__())

    def pop(self):
        elem = self.get_n_items(1)
        if len(elem) > 0:
            self.set.remove(elem[0])
            return elem[0]
        return None
