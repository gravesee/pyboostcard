from pyboostcard.constraints import *
from pyboostcard.decisionstump import *
from pyboostcard.boostcard import *

from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.tree._tree import Tree

from sklearn.model_selection import train_test_split, KFold, cross_val_score

import pandas as pd
from typing import Tuple, List, Dict, cast
import numpy as np
import copy

import seaborn as sns

df = pd.read_csv("train.csv")

k = ['Age','Fare', 'SibSp', 'Pclass']
data = df[k]
y = df['Survived']
data['Sex'] = df.Sex.map({'male': 0, 'female': 1})

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33)

## create constraints, one for Age, one for Fare

# c_age = Constraint(
#     Missing(order=4),
#     Interval((0, 30), (True, True), 2, mono=-1),
#     Interval((30, 100), (False, True), 1, mono=0), name='Age')

# x = np.array([np.nan, 24.0, 0, 30, 31, 100])

# print(np.hstack([x.reshape(-1, 1), c_age.transform(x)[0]]))


bst = BoostCardClassifier(constraints="config.json", min_child_weight=25, n_estimators=100)

scores = cross_val_score(clf, X=x_train, y=y_train, cv=2, scoring='roc_auc')

# bst = BoostCardClassifier(constraints=[c1,c2,c3,c4], min_child_weight=25, n_estimators=200, learning_rate=0.1)
bst.fit(x_train, y_train)

bst.transform(x_test)

bst.score(data, y.values)

bst.decision_function(data, columns=True)


from sklearn.metrics import roc_auc_score, roc_curve
fpr, tpr, _ = roc_curve(y, bst.predict_log_proba(data))
auc = roc_auc_score(y, bst.predict_log_proba(data))

sns.lineplot(fpr, tpr)






# bst.predict(data)
# bst.predict_log_proba(data)
yhat = bst.decision_function(data, columns=True)


# ## does grid search work?
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parameters = {
    'min_child_weight': [5, 10, 25, 50, 100],
    'max_leaf_nodes': list[5, 10, 25, 50, 100],

    }

# bst = BoostCardClassifier(constraints="config.json", n_estimators=500)

# bst.fit(data, y, eval_metric='auc', verbose=True)
clf = GridSearchCV(bst, parameters, cv=10)

from scipy.stats import randint as sp_randint
from scipy.stats import uniform
parameters = {
    'min_child_weight': sp_randint(5, 100),
    'max_leaf_nodes': sp_randint(2, 10),
    }


clf = RandomizedSearchCV(bst, parameters, n_iter=50, cv=5)
clf.fit(data, y, eval_metric='auc', verbose=True)

yhat = clf.best_estimator_.decision_function(data, columns=True)

sns.scatterplot(x=df['Age'], y=yhat['Age'])
sns.scatterplot(x=df['Sex'], y=yhat['Sex'])
sns.scatterplot(x=df['Pclass'], y=yhat['Pclass'])
sns.scatterplot(x=df['Fare'], y=yhat['Fare'])
sns.scatterplot(x=df['SibSp'], y=yhat['SibSp'])

# #sns.scatterplot(x=df['Age'], y=yhat[:,0])
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
