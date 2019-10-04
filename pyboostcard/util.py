from typing import List, Set, Tuple, cast, Any
from xgboost.sklearn import XGBClassifier
from tempfile import mkstemp
from sklearn.tree._tree import Tree
import os
import re

# TODO: Clean this up

def indices(l: List[int]) -> List[int]:
    """return sorted positions of elements in l"""
    seen: Set[int] = set()
    uniq = [x for x in sorted(l) if x not in seen and not seen.add(x)]  # type: ignore
    lookup = {k: i for (i, k) in enumerate(uniq)}
    return [lookup[x] for x in l]


Features = Tuple[int, float]
LeafValues = Tuple[float, float]


def get_xgb_features_and_values(clf: XGBClassifier) -> Tuple[List[Features], List[LeafValues]]:
    """Use regex to find (features, thresholds) and (left, right) splits"""

    fd, fout = mkstemp(text=True)

    clf.get_booster().dump_model(fout, with_stats=True)

    with open(fout, "r") as fin:
        txt = fin.read()

    os.close(fd)

    pat = "\[f([0-9]+)<([0-9]+.*[0-9-e]*)\]"
    features_thresholds = list(map(lambda x: (int(x[0]), float(x[1])), re.findall(pat, txt)))

    _ = list(map(float, re.findall("leaf=(-{,1}[0-9]+.[0-9-e]+),", txt)))
    left_right = cast(List[Tuple[float, float]], list(zip(*(iter(_),) * 2)))

    return features_thresholds, left_right


# extract lists by feature indices
def filter_lists_by_fid(
    ft: List[Features], lv: List[LeafValues], fids: List[int]
) -> Tuple[List[Features], List[LeafValues]]:
    # given sets of ids, return list with just those ids
    out: Tuple[List[Features], List[LeafValues]] = ([], [])
    for x, y in zip(ft, lv):
        if x[0] in fids:
            out[0].append(x)
            out[1].append(y)

    return out


def lengths_to_indices(lens: List[int]) -> List[List[int]]:
    """[2, 3, 2] -> [[0,1], [2,3,4], [,5,6]]"""
    out = []
    curr = 0
    for l in lens:
        out.append(list(range(curr, curr + l)))
        curr = curr + l
    return out


def split_xgb_outputs(clf: XGBClassifier, lens: List[int]) -> List[Tuple[List[Features], List[LeafValues]]]:
    ft, lv = get_xgb_features_and_values(clf)

    ids = lengths_to_indices(lens)

    out = []
    for var_ids in ids:
        out.append(filter_lists_by_fid(ft, lv, var_ids))

    return out


## functions to extract split boundaries and values from a decision tree
def sklearn_tree_to_bins(tree: Tree) -> List[Tuple[float, float]]:
    """given an sklearn tree, return a set of bin breaks and mapped values"""


    pass