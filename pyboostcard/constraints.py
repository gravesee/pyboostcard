from pyboostcard.selections import *

## a constraint is a selection mapped to a relative ordering and optional monotonicity


class Mapping:

    def __init__(self, selection: Selection, order: int, mono: int = 0):
        self.selection = selection
        self.order = order
        self.mono = mono
    
    def __repr__(self) -> str:
        return f"{self.selection.__repr__():<20} => {{{self.order:>3}, {self.mono:.3}}}"


if __name__ == "__main__":
    m1 = Mapping(Missing(), 1)
    m2 = Mapping(Exception(-1), 2)
    m3 = Mapping(Interval((0., 10.), (True, True)), 3, 1)

    print(m1)
    print(m2)
    print(m3)
