# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-few-public-methods, attribute-defined-outside-init
"""custom argparse action"""
import argparse


class AppendTupleWithoutDefault(argparse._AppendAction):
    """Ignore the default tuple value when append the tuple"""
    def __call__(self, parser, namespace, values, option_string=None):

        items = argparse._copy.copy(argparse._ensure_value(namespace, self.dest, []))
        try:
            self._not_first
        except AttributeError:
            self._not_first = True
            del items[:]
        items.append(tuple(values))
        setattr(namespace, self.dest, items)
