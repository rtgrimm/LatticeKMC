import inspect
import os.path
from typing import Dict
import deepdish as dd

class HDF5:
    _root : str

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError()

        self._root = path

    def _construct_path(self, name):
        return os.path.join(self._root, name)

    def load(self, name):
        return dd.io.load(self._construct_path(name))

    def save(self, name, data):
        dd.io.save(self._construct_path(name), data)

    def invalidate(self, name):
        os.remove(self._construct_path(name))

    def valid(self, name):
        return os.path.exists(self._construct_path(name))

def connect(func, name):
    return task(func, deps=[name])

def build_func_info(func, deps, name, skip_cache = False):
    return {
        "func" : func,
        "name" : name,
        "deps" : deps,
        "skip_cache" : skip_cache
    }

def task(func, deps = None, name = None):
    return build_func_info(func = func,
                           deps = [
                                param for param in inspect.signature(func).parameters
                            ] if deps is None else deps,
                           name = func.__name__ if name is None else name)


class Graph:
    _registry : Dict

    def __init__(self, persistence):
        self._registry = {}
        self._persistence = persistence

    def __call__(self, name):
        return self.request(name)

    def register_all(self, func_list):
        for func in func_list:
            self.register(func)

    def register(self, func_info):
        self._registry[func_info["name"]] = {
            "func" : func_info["func"],
            "deps" : func_info["deps"]
        }

    def invalidate(self, name):
        self._persistence.invalidate(name)

    def request(self, name):
        if self._persistence.valid(name):
            return self._persistence.load(name)

        func_info = self._registry[name]

        arg_data_list = [self.request(dep) for dep in func_info["deps"]]

        data = func_info["func"](*arg_data_list)

        if not (data is None):
            self._persistence.save(name, data)

        return data

