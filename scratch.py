from pyboostcard.constraints import *
from pyboostcard.decisionstump import *
from pyboostcard.boostcard import BoostCardClassifier

from sklearn.tree._tree import Tree
import pandas as pd
from typing import Tuple, List, Dict, cast
import numpy as np
import copy

m1 = Missing(order=0)
m2 = Override(-1, order=4)
m3 = Interval((0, 18), (True, True), 2, mono=0)
m4 = Interval((18, 62), (False, True), 2, mono=0)
m5 = Interval((62, 100), (False, True), 2, mono=0)

c1 = Constraint(m1, m2, m3, m4, m5, name="Age")
# tf, m = c1.transform(x)

## need to test that this is working

import numpy as np

age = np.random.randint(0, 100, 1000)
age[:100] = -1
probs = np.where(age >= 62, 0.30, np.where(age >= 18, 0.2, 0.1))

y = pd.Series(np.random.binomial(n=1, p=probs))
df = pd.DataFrame({"Age": age})

bst = BoostCardClassifier(constraints=[c1])

bst.fit(X=df, y=y)


from sklearn.tree import DecisionTreeRegressor, export_graphviz

clf = DecisionTreeRegressor(max_leaf_nodes=8)
clf.fit(age.reshape(-1, 1), y)

tree = copy.deepcopy(clf.tree_)

## which indices are the leaves
leaves = [i for i, t in enumerate(tree.threshold) if t == -2]

# left splits/right splits
ls = [i for i, x in enumerate(tree.children_left) if x in leaves]
rs = [i for i, x in enumerate(tree.children_right) if x in leaves]

res = [(tree.threshold[i], float(tree.value[l])) for i, l in zip(ls + rs, leaves)]
srt = sorted(res, key=lambda x: x[0])

## need to record if left or right split as well

## TODO: figure this shit out tomorrow!

## go down tree and keep track of max/min thresh of each level
## output max/min/value at each leaf


def tree_to_splits(tree: Tree) -> List[Tuple[float, ...]]:
    # inner function that recursively finds boundaries and final values
    def recurse(
    tree: Tree, node: int, bounds: Tuple[float, ...]=(-np.inf, np.inf), res: List[Tuple[float, ...]] = list()
    ) -> None:
        if tree.threshold[node] == -2:
            res.append(tuple(bounds) + (float(tree.value[node]),))
            return None

        if tree.children_left[node] != -1:
            recurse(tree, tree.children_left[node], (bounds[0], tree.threshold[node]), res)

        if tree.children_right[node] != -1:
            recurse(tree, tree.children_right[node], (tree.threshold[node], bounds[1]), res)

    result: List[Tuple[float, ...]] = []
    recurse(tree, 0, res=result)    
    return result


pd.Series(clf.predict(age.reshape(-1, 1))).value_counts()

import graphviz

dot_data = export_graphviz(clf)
graph = graphviz.Source(dot_data)

graph.render()

# clf.predict(age.reshape(-1, 1))
