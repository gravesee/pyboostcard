from typing import List, Set

# TODO: Clean this up

def indices(l: List[int]) -> List[int]:
    """return sorted positions of elements in l"""
    seen: Set = set()
    uniq = [x for x in sorted(l) if x not in seen and not seen.add(x)] # type: ignore
    lookup = {k: i for (i, k) in enumerate(uniq)}
    return [lookup[x] for x in l]
