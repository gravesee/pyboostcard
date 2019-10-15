from __future__ import annotations
from pyboostcard.constants import *

from typing import Dict, Type, Tuple, Union, Callable, Optional, Any, cast
from collections import namedtuple
from abc import ABC, abstractmethod, abstractproperty
import numpy as np # typing: ignore
import operator as op
import json
import re

np.warnings.filterwarnings('ignore')

Comparator = Callable[[np.ndarray, float], np.ndarray]

class Selection(ABC):

    priority: int

    def __init__(self, order: int = 0):
        self.order = order
        self.value: Optional[float] = np.nan

    @staticmethod
    def bounds_from_string(chars: str) -> Tuple[bool, bool]:
        m = {"[]": (True, True), "(]": (False, True), "[)": (True, False), "()": (False, False)}
        out = m.get(chars, None)
        if out is None:
            raise ValueError(f"Bounds string not recognized: {chars}.")
        return out
    
    @staticmethod 
    def interval_from_string(chars: str) -> Tuple[Tuple[float, float], Tuple[bool, bool]]:
        bounds = re.sub("[^\[\(\]\)]", "", chars)
        tmp = re.sub("[\[\(\]\)]", "", chars).split(",")
        if len(tmp) != 2:
            raise ValueError(f"Invalid specification for interval: {chars}")        
        
        ## convert to floats
        vals = []
        for v in tmp:
            if v == 'inf':
                vals.append(np.inf)
            elif v == '-inf':
                vals.append(-np.inf)
            else:
                vals.append(float(v))

        values = cast(Tuple[float, float], tuple(vals))
        return values, Selection.bounds_from_string(bounds)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Selection:
        if d["type"] == "interval":
            tmp = Selection.interval_from_string(d["values"])
            out: Selection = Interval(tmp[0], tmp[1], d["order"], d["mono"])
        elif d["type"] == "override":
            out = Override(d["override"], d["order"])
        elif d["type"] == "missing":
            out = Missing(d["order"])
        elif d["type"] == "identity":
            out = Identity()
        else:
            raise ValueError(f"Selection type, {d['type']}, not recognized.")

        return out

    @staticmethod
    def from_json(s: str) -> Selection:
        res: Selection = json.loads(s, object_hook=Selection.from_dict)
        if res is None:
            raise IOError("Could not convert json to Selection")
        return res

    @abstractmethod
    def in_selection(self, x: np.ndarray) -> np.ndarray:
        pass

    @property
    def sort_value(self) -> Tuple[int, int, float]:
        return self.priority, self.order, -np.inf
    
Bounds = namedtuple("Bounds", ["left", "right"])

class FittedSelection:
    def __init__(self, selection: Selection, value: Optional[float] = None):
        self.selection = selection
        self.value = value

    def transform(self, x: np.ndarray, result: np.ndarray) -> np.ndarray:
        replace = x if self.value is None else self.value
        # make sure to only update output vector where filter is true AND result == np.nan
        f = self.selection.in_selection(x) & np.isnan(result)
        return np.where(f, replace, result)

    @property
    def sort_value(self) -> Tuple[int, int, float]:
        return self.selection.sort_value

    @property
    def fitted(self) -> bool:
        return (self.value is not None) and (self.value != np.nan)


class Identity(Selection):
    """Selection responsible for only passing through -- no constraint in other words"""

    priority = 100

    def __init__(self, order: int = 0):
        self.order = 0
        self.value: Optional[float] = None

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        """Always return true for identity selections"""
        return np.full_like(x, True, dtype="bool")

    def __repr__(self) -> str:
        return "I"


class Interval(Selection):

    priority = 0

    testmap: Dict[Tuple[bool, bool], Tuple[Comparator, Comparator]] = {
        (False, False): (op.gt, op.lt),
        (False, True): (op.gt, op.le),
        (True, False): (op.ge, op.lt),
        (True, True): (op.ge, op.le),
    }

    def __init__(self, values: Tuple[float, float], bounds: Tuple[bool, bool], order: int = 0, mono: int = 0):
        """Bounds are tuple of bools where each indicates closed boundary"""
        super().__init__(order)
        self.values = tuple(sorted(values))
        self.bounds = Bounds(*bounds)
        self.mono = mono

    def __repr__(self) -> str:
        return "i"

    @property
    def sort_value(self) -> Tuple[int, int, float]:
        return self.priority, self.order, self.values[0]

    @property
    def mono(self) -> int:
        return self._mono

    @mono.setter
    def mono(self, value: int) -> None:
        if value not in [-1, 0, 1]:
            raise ValueError("Invalid monotonicity.")
        else:
            self._mono = value

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        z = np.ma.masked_invalid(x)
        ltest, rtest = self.testmap[self.bounds]
        return ltest(z, self.values[0]) & rtest(z, self.values[1])


class Override(Selection):

    priority = 1

    def __init__(self, override: float, order: int = 0):
        super().__init__(order)
        self.override = override

    def __repr__(self) -> str:
        return "O"

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        return (x == self.override) & ~np.isnan(x)


class Missing(Selection):

    priority = 2

    def __init__(self, order: int = 0):
        super().__init__(order)

    def __repr__(self) -> str:
        return "M"

    def in_selection(self, x: np.ndarray) -> np.ndarray:
        return np.isnan(x)


if __name__ == "__main__":

    i = Interval.from_json('{"type":"interval", "bounds":"[]", "values": [10.0, 20.0], "order":0, "mono":0}')
    print(i)

    res = Selection.interval_from_string("[20.0, 30, 40)")
    print(res)
