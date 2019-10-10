from pyboostcard.constraints import *
from pyboostcard.decisionstump import *
from pyboostcard.boostcard import BoostCardClassifier
# import pyboostcard.util as util
from xgboost.sklearn import XGBClassifier
from sklearn.tree._tree import Tree
import pandas as pd
from typing import Tuple, List, Dict, cast
import numpy as np
import copy

m1 = Missing(order=0)
m2 = Override(-1, order=3)
m3 = Interval((0, 50), (True, True), 2, mono=0)
m4 = Interval((50, 100), (False, True), 3, mono=0)
# m5 = Interval((18, 100), (False, True), 4, mono=1)

c1 = Constraint(m1, m2, m3, m4, name="Age")
#c1 = Constraint(m1, m2, m3, m4, m5, name="Age")

x = np.array([np.nan, -1., 0, 50, 51, 100])

tf, m = c1.transform(x)

print(np.hstack([x.reshape(-1,1), tf]))

## need to test that this is working

age = np.random.randint(0, 100, 1000)
age[:100] = -1
probs = np.where(age >= 62, 0.80, np.where(age >= 18, 0.60, 0.55))

y = pd.Series(np.random.binomial(n=1, p=probs))
df = pd.DataFrame({"Age": age})

# testing using just xgboost
# x, m = c1.transform(age)
# clf = XGBClassifier(min_child_weight=100, max_depth=1, learning_rate=0.1,
#                     monotone_constraints='(1, 1, 1, 1, -1, -1)', objective='binary:logitraw')
# x, _ = c1.transform(age)
# clf.fit(x, y)
# yhat = clf.predict_proba(x)[:,1]


import seaborn as sns
# sns.scatterplot(x=age, y=yhat)


# pd.DataFrame(np.hstack([age.reshape(-1,1), x])).iloc[:10,:]

bst = BoostCardClassifier(constraints=[c1], min_child_weight=100, learning_rate=0.1)

bst.fit(X=df, y=y)
x, _ = c1.transform(age)
yhat = bst.xgb.predict_proba(x)[:,1]
sns.scatterplot(x=df['Age'], y=yhat)

bst.xgb

# d1 = clf.get_params()
# d2 = bst.xgb.get_params()
# for k, v in d1.items():
#     print(f"{k}: match: {d1[k] == d2[k]}")


# clf.get_params() == bst.xgb.get_params()


# ## TROUBLE SHOOT

# X = df

# xs: List[np.ndarray] = []
# monos: List[int] = []
# for constraint in bst.constraints:
#     data, mono = constraint.transform(X[[constraint.name]])
#     xs.append(data)
#     monos += mono

# # create list of number of cols produced for each feature
# lens: List[int] = [x.shape[1] for x in xs]

# clf.get_params() == bst.xgb.get_params()


# # from sklearn.tree import DecisionTreeRegressor, export_graphviz

# # clf = DecisionTreeRegressor(max_leaf_nodes=8)
# # clf.fit(age.reshape(-1, 1), y)

# # tree = copy.deepcopy(clf.tree_)

# # ## which indices are the leaves
# # leaves = [i for i, t in enumerate(tree.threshold) if t == -2]

# # # left splits/right splits
# # ls = [i for i, x in enumerate(tree.children_left) if x in leaves]
# # rs = [i for i, x in enumerate(tree.children_right) if x in leaves]

# # res = [(tree.threshold[i], float(tree.value[l])) for i, l in zip(ls + rs, leaves)]
# # srt = sorted(res, key=lambda x: x[0])

# # ## need to record if left or right split as well

# # ## TODO: figure this shit out tomorrow!

# # ## go down tree and keep track of max/min thresh of each level
# # ## output max/min/value at each leaf

# # pd.Series(clf.predict(age.reshape(-1, 1))).value_counts()

# # import graphviz

# # dot_data = export_graphviz(clf)
# # graph = graphviz.Source(dot_data)

# # graph.render()

# # # clf.predict(age.reshape(-1, 1))
