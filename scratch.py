from pyboostcard.constraints import *
from pyboostcard.decisionstump import *
from pyboostcard.boostcard import *

from xgboost.sklearn import XGBClassifier
from sklearn.tree._tree import Tree
import pandas as pd
from typing import Tuple, List, Dict, cast
import numpy as np
import copy

import seaborn as sns

df = pd.read_csv("train.csv")

k = ['Age','Fare']
data = df[k]
y = df['Survived']

## create constraints, one for Age, one for Fare

c_age = Constraint(
    Missing(order=4),
    Override(override=24.0, order=0),
    Interval((0, 30), (True, True), 2, mono=-1),
    Interval((30, 100), (False, True), 1, mono=0), name='Age')

x = np.array([np.nan, 24.0, 0, 30, 31, 100])

print(np.hstack([x.reshape(-1, 1), c_age.transform(x)[0]]))


c1 =Constraint(
    Missing(order=4),
    Override(override=24.0, order=0),
    Interval((0, 30), (True, True), 2, mono=-1),
    Interval((30, 100), (False, True), 1, mono=0), name='Age')

c2 = Constraint(
    Interval((0, 1000), (True, True), 0, mono=1),
    name='Fare')

bst = BoostCardClassifier(constraints=[c1,c2], min_child_weight=25)

bst = BoostCardClassifier(constraints="config.json", min_child_weight=25)
bst.fit(data, y)


bst.score(data, y.values)


bst.predict(data)
bst.predict_log_proba(data)
bst.decision_function(data, columns=True)


## does grid search work?
from sklearn.model_selection import GridSearchCV

parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_weight': [10, 25, 50]
    }

bst = BoostCardClassifier(constraints="config.json", n_estimators=500)

bst.fit(data, y, eval_metric='auc', verbose=True)


# clf = GridSearchCV(bst, parameters, cv=5)
# clf.fit(data, y)

# yhat = clf.estimator.decision_function(data, columns=True)

# sns.scatterplot(x=df['Age'], y=yhat[:,0])

# sns.scatterplot(x=df['Age'], y=yhat[:,0])
# sns.scatterplot(x=df['Fare'], y=yhat[:,1])

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

## debug nump warning
# from numpy.
