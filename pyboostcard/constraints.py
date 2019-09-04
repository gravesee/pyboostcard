from pyboostcard.selections import *
from pyboostcard.constants import *

from typing import List, Optional, Any, cast, Tuple
from operator import attrgetter
import numpy as np
from itertools import tee


# class BluePrint:
#     def __init__(self, selection: Selection, value: float):
#         pass

def check_valid_intervals(selections: List[Interval]) -> None:
    ## sort the ranges in ascending order by ll
    a, b = tee(sorted(selections, key=attrgetter("values")), 2)
    next(b)
    for prev, curr in zip(a, b):
        if ((prev.values[1] == curr.values[0]) and (prev.bounds.right == curr.bounds.left)) or \
            (prev.values[1] != curr.values[0]) or \
            (prev.values[1] > curr.values[0]):
            raise ValueError(f"Disjoint or overlapping intervals: {str(prev)}, {str(curr)}")
    return None    


## a constraint is a managed list of selections with other methods
class Constraint:
    """A constraint is a collection of selections"""

    @staticmethod
    def filter_types(selections: List[Selection], type: Type[Selection]) -> List[Selection]:
        """Return filtered list of a single, specified type"""
        return [x for x in selections if isinstance(x, type)]

    def __init__(self, *args: Selection):
        if not all(isinstance(x, Selection) for x in args):
            raise ValueError("All constraint arguments must be Selection objects.")

        selections = sorted(args, key=attrgetter("sort_value"), reverse=True)

        # Check Missing Selections
        if len(self.filter_types(selections, Missing)) > 1:
            raise ValueError("Constraint arguments can only have 1 Missing selection.")

        # Check Exception Selections
        exceptions = cast(List[Exception], self.filter_types(selections, Exception))
        vals = [e.value for e in exceptions]
        if len(set(vals)) != len(vals):
            raise ValueError("Exception selections must have unique values.")

        # Check Interval Selections
        check_valid_intervals(cast(List[Interval], self.filter_types(selections, Interval)))

        self.selections = selections
        self._fitted: bool = False

    def fit(self, X: Any = None, y: None = None) -> None:
        # easiest case, re-arrange selections and set mapped values
        self._fitted = True
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Attempting to call `transform` on constraint that hasn't been fit.")
        pass

    def __repr__(self) -> str:
        # call repr on all selections print heading
        lines = HEADER + [repr(sel) for sel in self.selections]
        return "\n".join(["|" + line + "|" for line in lines])


## need to fit a constraint which analyzes the selections/order/mono and creates a blueprint
## for outputting mutltiple vectors

## Heuristics:
# everything revolves around each interval in turn
# if an interval is mono -1, or 1 and there is only 1, then rearrange and done
# if an interval has mono 0 and OTHER selections, then it must be duplicated

if __name__ == "__main__":
    m1 = Missing()
    m2 = Exception(-1, 1)
    m3 = Exception(-2, 2)
    m4 = Interval((0.0, 10.0), (True, True))
    m5 = Interval((10.0, 20.0), (False, True))

    c1 = Constraint(m1, m2, m3, m4, m5)
    c1.fit()
    c1.transform(x=np.array([1, 2, 3]))
    print(c1)
