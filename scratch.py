from pyboostcard.constraints import *
from typing import Tuple, List, Dict
import numpy as np

m1 = Missing(order=0)
m2 = Exception(-1, order=4)
m3 = Interval((0, 18), (True, True), 2, mono=0)
m4 = Interval((18, 62), (False, True), 2, mono=0)
m5 = Interval((62, 100), (False, True), 2, mono=0)

c1 = Constraint(m1, m2, m3, m4, m5)
# tf, m = c1.transform(x)


## need to test that this is working

import numpy as np

age = np.random.randint(0, 100, 1000)
age[:100] = -1
probs = np.where(age >= 62, 0.30, np.where(age >= 18, 0.2, 0.1))
y = np.random.binomial(n=1, p=probs)

from xgboost import XGBClassifier

tf, m = c1.transform(age)

clf = XGBClassifier(
    max_depth=1, n_estimators=100, monotone_constraints=tuple(m), learning_rate=0.1, min_child_weight=10
)
clf.fit(tf, y=y, verbose=True)

py = clf.predict_proba(data=tf)[:, 1]

import pandas as pd

plt = pd.DataFrame({"x": pd.Series(age), "y": py})

import seaborn as sns

sns.regplot(x="x", y="y", data=plt, logistic=True)

## function to get all of the features and splits of a tree as well as the final points
clf.get_booster().dump_model("xgb_model.txt", with_stats=True)

# read the contents of the file
with open("xgb_model.txt", "r") as f:
    txt_model = f.read()

# print(txt_model)

## regex to get feature #, threshold and leaf values
import re

# trying to extract all patterns like "[f2<2.45]"

split_pattern = "\[f([0-9]+)<([0-9]+.*[0-9-e]*)\]"
splits = list(map(lambda x: (int(x[0]), float(x[1])), re.findall(split_pattern, txt_model)))
values = list(map(float, re.findall("leaf=(-{,1}[0-9]+.[0-9-e]+),", txt_model)))

## create class that returns predicted value based on stumps from xgboost model

# splits
class DecisionStump:
    def __init__(self, features: List[Tuple[int, float]], values: List[Tuple[float, float]]):
        ## aggregate tuples by feature and thresholds and sum values
        fm: Dict[Tuple[int, float], Tuple[float, float]] = {}

        ## split values into groups of two
        values_ = list(zip(*(iter(values),) * 2))

        for i, f in enumerate(features):
            tmp = fm.get(f, (0.0, 0.0))
            tmp = (tmp[0] + values_[i][0], tmp[1] + values_[i][1])
            fm[f] = tmp

        ## sort the keys and reindex
        self._feature_map = {k: fm[k] for k in sorted(fm.keys())}

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape[0])

        for (feature, threshold), (left, right) in self._feature_map.items():
            out += np.where(x[:, feature] < threshold, left, right)

        return out
