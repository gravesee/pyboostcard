from pyboostcard.selections import *
from typing import List, Optional

## a constraint is a managed list of selections with other methods
class Constraint:
    """A constraint is a collection of selections"""

    def __init__(self, *args: Selection):
        if not all(isinstance(x, Selection) for x in args):
            raise ValueError("All constraint arguments must be Mapping objects.")
        self.selections: List[Selection] = list(args)

    def __repr__(self) -> str:
        # call repr on all selections print heading
        heading = [f"{'Selection':<{SELECTION_WIDTH}}|{'Order':^{ORDER_WIDTH}}|{'Mono':^{MONO_WIDTH}}"]
        heading += [f"{'-'*SELECTION_WIDTH}|{'-'*ORDER_WIDTH}|{'-'*MONO_WIDTH}"]        
        lines = heading + [str(sel) for sel in self.selections]
        return "\n".join(["|" + line + "|" for line in lines])


if __name__ == "__main__":
    m1 = Missing()
    m2 = Exception(-1)
    m3 = Interval((0.0, 10.0), (True, True))

    c1 = Constraint(m1, m2, m3)
    print(c1)
