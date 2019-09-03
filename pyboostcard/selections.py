from __future__ import annotations
from typing import Dict, Type, Tuple, Union, Callable
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np


class Selection(ABC):
    @abstractmethod
    def in_selection(self, x: np.ndarray) -> np.ndarray:
        pass


Bounds = namedtuple("Bounds", ["left", "right"])


class Interval(Selection):

    charmap: Dict[Tuple[bool, bool], str] = {
        (False, False): "({}, {})",
        (False, True): "({}, {}]",
        (True, False): "[{}, {})",
        (True, True): "[{}, {}]",
    }

    testmap: Dict[Tuple[bool, bool], Callable[[np.ndarray, Tuple[float, float]], np.ndarray]] = {
        (False, False): lambda x, b: (x > b[0]) & (x < b[1]),
        (False, True): lambda x, b: (x > b[0]) & (x <= b[1]),
        (True, False): lambda x, b: (x >= b[0]) & (x < b[1]),
        (True, True): lambda x, b: (x >= b[0]) & (x <= b[1]),
    }

    def __init__(self, values: Tuple[float, float], bounds: Tuple[bool, bool]):
        """Bounds are tuple of bools where each indicates closed boundary"""
        self.values = values
        self.bounds = Bounds(*bounds)
        self.repr = self.charmap[bounds]

    def __repr__(self) -> str:
        return self.repr.format(*self.values)

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        test = self.testmap[self.bounds]
        return test(x, self.values)


class Exception(Selection):
    def __init__(self, value: Union[str, float, int]):
        self.value = value

    def __repr__(self) -> str:
        return f"|{self.value}|"

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        return (x == self.value) & ~np.isnan(x)


class Missing(Selection):
    def __repr__(self) -> str:
        return "Missing"

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        return np.isnan(x)


if __name__ == "__main__":
    print(Interval((10.0, 20.0), (False, False)))
    print(Exception(-1))
    print(Missing())

    i = Interval((5.0, 7.0), (False, False))
    x = np.array(list(range(10)))
    print(i.in_selection(x))
