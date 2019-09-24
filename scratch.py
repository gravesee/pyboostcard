from pyboostcard.constraints import *
from pyboostcard.decisionstump import *

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
    max_depth=1, n_estimators=100, monotone_constraints=tuple(m), learning_rate=0.1, min_child_weight=5
)
clf.fit(tf, y=y, verbose=True)

py = clf.predict_proba(data=tf)[:, 1]

import pandas as pd

plt = pd.DataFrame({"x": pd.Series(age), "y": py})

import seaborn as sns

sns.regplot(x="x", y="y", data=plt, logistic=True)

## function to get all of the features and splits of a tree as well as the final points

## create class that returns predicted value based on stumps from xgboost model

## function to split vars by which features they belong

# splits



# ds = DecisionStump(splits, values)

# vals = ds.transform(tf)

# data = pd.DataFrame({'Age':age, 'y':vals})

# # get new dataset to predict
# from sklearn.tree import DecisionTreeRegressor

# dt = DecisionTreeRegressor(max_leaf_nodes=8)
# dt.fit(data[['Age']], data['y'])

# p = dt.predict(data[['Age']])

# plt = pd.DataFrame({'x':data['Age'], 'y':p})
# sns.regplot(data=plt, x='x', y='y', logistic=True)

