from pyboostcard.selections import *
from pyboostcard.constants import *
from pyboostcard.util import indices

from typing import List, Optional, Any, cast, Tuple, Iterable, cast
from operator import attrgetter
import numpy as np
from sklearn.utils import check_array
from itertools import tee
import scipy as sp


class Blueprint:
    """A collection of fitted selections that together produced columns for ML"""

    def __init__(self, selections: List[Selection], mono: Optional[int] = 0):
        ## check they are all fitted
        for sel in selections:
            if not sel.fitted:
                raise RuntimeError("Must fit selections before adding to blueprint.")

        self.selections = selections
        self.mono = mono


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
    def filter_types(selections: List[Selection], type: Type[Selection]) -> List[Selection]:
        """Return filtered list of a single, specified type"""
        return [x for x in selections if isinstance(x, type)]

    def __init__(self, *args: Selection):
        if not all(isinstance(x, Selection) for x in args):
            raise ValueError("All constraint arguments must be Selection objects.")

        self.selections = sorted(args, key=attrgetter("sort_value"), reverse=True)
        self._blueprints: List[Blueprint] = []

        # Check Missing Selections
        if len(self.filter_types(self.selections, Missing)) > 1:
            raise ValueError("Constraint arguments can only have 1 Missing selection.")

        # Check Exception Selections
        exceptions = cast(List[Exception], self.filter_types(self.selections, Exception))
        vals = [e.value for e in exceptions]
        if len(set(vals)) != len(vals):
            raise ValueError("Exception selections must have unique values.")

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
        return indices([x.order * mul for x in self.selections])

    def __fit_interval(self, interval: Interval) -> List[Blueprint]:

        if interval.mono == 0:
            monos = (1, -1)
        elif interval.mono == 1:
            monos = (1, 1)
        else:
            monos = (-1, 1)

        out: List[Blueprint] = []
        for mi, mono in enumerate(monos):
            order = self.order(desc=False if mono == 1 else True)
            ll, ul = interval.values

            # need the index order of the current interval, not the original order
            pos = self.selections.index(interval)
            i = order[pos]

            # this sets what the value of the mapping will be based on
            # where the current interval index is and the relative positions
            # of the other constraints
            vals: List[Optional[float]] = []
            for j in order:
                if j < i:
                    vals.append(ll - 1 - (i - j))
                elif j == i:
                    if mi == 0:
                        vals.append(ll - 1)
                    else:
                        vals.append(ul + 1)
                else:
                    vals.append(ul + 1 - (i - j))

            # current interval gets None value to signal pass-through predictions
            vals[pos] = None

            for (sel, val) in zip(self.selections, vals):
                sel.fit(val)

            out.append(Blueprint(self.selections, mono))

        return out

    def __fit(self) -> None:
        self._blueprints.clear()

        intervals = self.get_intervals()

        if len(intervals) > 0:
            for interval in intervals:
                self._blueprints += self.__fit_interval(interval)
        else:
            tmp = []
            for sel, val in zip(self.selections, self.order()):
                sel.fit(val)
                tmp.append(sel)

            self._blueprints += [Blueprint(tmp, None)]

    def transform(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if not self.fitted:
            raise RuntimeError("Attempting to call `transform` on constraint that hasn't been fit.")

        out: List[np.ndarray] = []
        for blueprint in self._blueprints:

            # start with a vector of np.nan to fill with the transformed results
            res = np.full_like(x, np.nan, dtype="float")
            for selection in blueprint.selections:
                res = selection.transform(x, res)

            out.append(res.reshape(-1, 1))

        monos = [x.mono if x.mono is not None else 0 for x in self._blueprints]
        out = np.hstack(out)

        if np.any(np.isnan(out)):
            raise RuntimeError("Not all values accounted for in constraint. np.nan found in transformed result.")

        return out, monos

    def __repr__(self) -> str:
        # call repr on all selections print heading
        lines = HEADER + [repr(sel) for sel in self.selections]
        return "\n".join(["|" + line + "|" for line in lines])


if __name__ == "__main__":
    # m1 = Missing()
    # m2 = Exception(-1, 2)
    # m3 = Exception(-2, 2)
    # m4 = Interval((0.0, 10.0), (True, True), order=3)
    # m5 = Interval((10.0, 21.0), (False, True), order=1)

    x = np.array([np.nan, -2, -1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 62, 80, 100])

    # c1 = Constraint(m1, m2, m3, m4, m5)

    m1 = Missing(order=0)
    m2 = Exception(-1, order=2)
    m3 = Interval((-2, 18), (True, True), 0, mono=1)
    m4 = Interval((18, 62), (False, True), 1, mono=1)
    m5 = Interval((62, 100), (False, True), 2, mono=1)

    c1 = Constraint(m1, m2, m3, m4, m5)
    tf, m = c1.transform(x)

    ## need to test that this is working

    import numpy as np

    age = np.random.randint(5, 100, 1000)
    probs = np.where(age >= 62, 0.30, np.where(age >= 18, 0.2, 0.1))
    y = np.random.binomial(n=1, p=probs)

    from xgboost import XGBClassifier

    tf, m = c1.transform(age)

    clf = XGBClassifier(
        max_depth=1, n_estimators=100, monotone_constraints=tuple(m), learning_rate=0.1, min_child_weight=50
    )
    clf.fit(tf, y=y, verbose=True)

    py = clf.predict_proba(data=tf)[:, 1]

    import pandas as pd

    plt = pd.DataFrame({"x": pd.Series(age), "y": py})

    import seaborn as sns

    sns.regplot(x="x", y="y", data=plt, logistic=True)
    # plt.plot()
