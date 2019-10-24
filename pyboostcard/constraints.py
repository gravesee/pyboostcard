from __future__ import annotations

from pyboostcard.selections import *
from pyboostcard.constants import *
from pyboostcard.util import indices

import copy
from typing import List, Optional, Any, cast, Tuple, Iterable, cast, Type
from operator import attrgetter
import numpy as np
from sklearn.utils import check_array
from itertools import tee
import scipy as sp


class Blueprint:
    """A collection of fitted selections that together produced columns for ML"""

    def __init__(self, selections: List[FittedSelection], mono: Optional[int] = 0):
        self.selections = sorted(selections, key=attrgetter("sort_value"), reverse=True)
        self.mono = mono
    
    def __len__(self) -> int:
        return len(self.selections)


def check_valid_intervals(selections: List[Interval]) -> None:
    ## sort the ranges in ascending order by ll and pair up (1,2), (2,3), (3,4), etc...
    a, b = tee(sorted(selections, key=attrgetter("values")), 2)
    next(b)
    for prev, curr in zip(a, b):
        ## subsequent boundaries must have the same value AND different types (open-closed or closed-open)
        if not ((prev.values[1] == curr.values[0]) & (prev.bounds.right != curr.bounds.left)):
            raise ValueError(f"Disjoint or overlapping intervals: {str(prev)}, {str(curr)}")
    return None


## a constraint is a managed list of selections with other methods
class Constraint:
    """A constraint is a collection of selections"""

    @staticmethod
    def from_json(s: str) -> Constraint:
        res: Dict[str, List[Any]] = json.loads(s)

        sels: List[Selection] = []
        for sel in list(res.values())[0]:
            sels.append(Selection.from_dict(sel))

        return Constraint(*sels, name=list(res.keys())[0])

    @staticmethod
    def filter_types(selections: List[Selection], type: Type[Selection]) -> List[Selection]:
        """Return filtered list of a single, specified type"""
        return [x for x in selections if isinstance(x, type)]

    def __init__(self, *args: Selection, name: Optional[str] = None):
        if not name:
            raise ValueError("Constraints must be named")

        if not all(isinstance(x, Selection) for x in args):
            raise ValueError("All args must be Selection objects.")

        self.name = name

        # TODO: Check if theres a clamp selection... if so, update the mins and maxes of the intervals to be 
        # within the bounds of the clamp

        self.selections = sorted(args, key=attrgetter("sort_value"), reverse=True)
        self._blueprints: List[Blueprint] = []

        # check if contains identity, if so, can only have one
        if (len(self.filter_types(self.selections, Identity)) > 0) & (len(self.selections) > 1):
            raise ValueError("Constraint with identity selection most be sole selector")

        # Check Missing Selections
        if len(self.filter_types(self.selections, Missing)) > 1:
            raise ValueError("Constraint arguments can only have 1 Missing selection.")

        if len(self.filter_types(self.selections, Clamp)) > 1:
            raise ValueError("Constraint arguments can only have 1 Clamp selection.")
    
        ## clip intervals if clamp present
        # if len(self.filter_types(self.selections, Clamp)) == 1:
        #     clamp = self.filter_types(self.selections, Clamp)[0]
        #     for interval in self.get_intervals():
        #         interval._clip(cast(Clamp, clamp))

        # Check Override Selections
        overrides = cast(List[Override], self.filter_types(self.selections, Override))
        vals = [e.value for e in overrides]
        if len(set(vals)) != len(vals):
            raise ValueError("Override selections must have unique values.")

        # Check Interval Selections
        if self.num_intervals > 0:
            check_valid_intervals(self.get_intervals())

        self.__fit()

    def get_intervals(self) -> List[Interval]:
        return cast(List[Interval], Constraint.filter_types(self.selections, Interval))

    @property
    def num_intervals(self) -> int:
        return len(self.get_intervals())

    @property
    def fitted(self) -> bool:
        return len(self._blueprints) > 0

    def order(self, desc: bool = False) -> List[int]:
        mul = -1 if desc else 1
        res = []
        for sel in self.selections:
            if sel.order is None:
                res.append(sel.order)
            else:
                res.append(sel.order * mul)
        return res
    
    def __len__(self) -> int:
        return len(self._blueprints)

    def __fit_interval(self, interval: Interval) -> List[Blueprint]:

        tmp: List[Clamp] = cast(List[Clamp], self.filter_types(self.selections, Clamp))
        if len(tmp) > 0:
            clamp: Optional[Clamp] = tmp[0]
        else:
            clamp = None

        if interval.mono == 0:
            monos: Tuple[int, ...] = (1, 1, -1, -1)
        elif interval.mono == 1:
            monos = (1, 1)
        else:
            monos = (-1, -1)

        out: List[Blueprint] = []
        for mi, mono in enumerate(monos):

            order = self.order(desc=False if mono == 1 else True)
            ll, ul = interval.values

            if clamp is not None:
                ll = max(ll, clamp.ll)
                ul = min(ul, clamp.ul)

            # need the index order of the current interval, not the original order
            pos = self.selections.index(interval)
            i = order[pos]

            # this sets what the value of the mapping will be based on
            # where the current interval index is and the relative positions
            # of the other constraints
            vals: List[Optional[float]] = []
            for j in order:
                if j is None: # clamp case
                    vals.append(None)
                elif j < i:
                    vals.append(ll - 1 - (i - j))
                elif j == i:
                    if mi % 2 == 0:
                        vals.append(ll - 1)
                    else:
                        vals.append(ul + 1)
                else:
                    vals.append(ul + 1 - (i - j))

            # current interval gets None value to signal pass-through predictions
            vals[pos] = None

            fs = [FittedSelection(sel, val) for sel, val in zip(self.selections, vals)]
            out.append(Blueprint(fs, mono))

        return out

    def __fit(self) -> None:
        self._blueprints.clear()

        intervals = self.get_intervals()

        # if intervals in constraints, have to do a lot of shuffluing
        if len(intervals) > 0:
            for interval in intervals:
                self._blueprints += self.__fit_interval(interval)
        else:  # if no intervals, much simpler
            tmp = []
            for sel, val in zip(self.selections, self.order()):
                if isinstance(sel, Identity) or isinstance(sel, Clamp):
                    tmp.append(FittedSelection(sel, None))
                else:
                    tmp.append(FittedSelection(sel, val))

            self._blueprints += [Blueprint(tmp, None)]

    def transform(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if not self.fitted:
            raise RuntimeError("Attempting to call `transform` on constraint that hasn't been fit.")

        tmp = cast(List[Clamp], self.filter_types(self.selections, Clamp))
        if len(tmp) == 0:
            clamp = Clamp(-np.inf, np.inf)
        else:
            clamp = tmp[0]

        out: List[np.ndarray] = []
        for blueprint in self._blueprints:
            #print("New Blueprint!")

            fitted_sels = blueprint.selections

            # start with a vector of np.nan to fill with the transformed results
            res = np.full_like(x, np.nan, dtype="float")
            for selection in fitted_sels:
                #print(f"  {type(selection.selection)}")
                res = selection.transform(x, res, clamp)

            out.append(res.reshape(-1, 1))

        monos = [x.mono if x.mono is not None else 0 for x in self._blueprints]
        out = np.hstack(out)

        if np.any(np.isnan(out)):
            raise RuntimeError("Not all values accounted for in constraint. np.nan found in transformed result.")

        return out, monos

    def __repr__(self) -> str:
        # call repr on all selections print heading
        sels = ",".join([str(x) for x in self.selections])
        return self.name + f"=[{sels}]"
