from __future__ import annotations

from pyboostcard.decisionstump import DecisionStump
from pyboostcard.constraints import *
from pyboostcard import util

from typing import Dict, List, Tuple
import copy

from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_consistent_length
from util import sklearn_tree_to_bins


class BaseBoostCard(BaseEstimator):
    def __init__(
        self,
        constraints: List[Constraint],
        objective: str = "reg:squarederror",
        eta: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        max_leaf_nodes: int = 8,
    ) -> None:

        self.constraints = constraints
        self.objective = objective
        self.eta = eta
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample

        self.max_leaf_nodes = max_leaf_nodes
        self.decision_stumps: List[DecisionStump] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostCardClassifier:
        check_consistent_length(X, y)

        ## transform featueres using the fitted constraints
        xs, monos = [], []
        for constraint in self.constraints:
            _x, _m = constraint.transform(X[[constraint.name]])
            xs.append(_x)
            monos.append(_m)

        lens: List[int] = [x.shape[1] for x in xs]

        self.xgb = XGBClassifier(
            max_depth=1,  # hard-coded
            objective=self.objective,
            eta=self.eta,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            montone_costraints=str(tuple(monos)),
        )

        # decision tree is always a regressor because we are using xgboost output as y
        clf = DecisionTreeRegressor(
            min_samples_leaf=self.min_child_weight, max_leaf_nodes=self.max_leaf_nodes
        )

        self.xgb.fit(np.concatenate(xs, axis=1), y)

        ## dump the model and generate the decision stumps
        mod_data = util.split_xgb_outputs(self.xgb, lens)

        for data in mod_data:
            self.decision_stumps.append(DecisionStump(data[0], data[1]))

        # generate data for the decision trees
        self._trees: List[Tree] = []
        for x, stump, constraint in zip(xs, self.decision_stumps, self.constraints):
            # TODO: loop over constraints, and treat each one differentlu
            # each missing, override should get a single value assigned
            # each interval should be transformed, nans filtered, and a tree fitted
            # the final results of each interval should be combined into one 
            yhat = stump.transform(x)
            clf.fit(X[[constraint.name]], yhat)
            self._trees.append(copy.deepcopy(clf.tree_))            

        # now have fitted trees...do something with them, like convert them to tuples of bins and values
        # TODO: add the variable name as well... and move this to the loop above!
        self._bins: List[Tuple[float, ...]] = [sklearn_tree_to_bins(t) for t in self._trees]

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_predictions = self.decision_function()
        pass


class BoostCardClassifier(BaseBoostCard, ClassifierMixin):
    def __init__(
        self,
        # xgb params
        constraints: List[Constraint],
        objective: str = "binary:logitraw",
        eta: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            eta=eta,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            max_leaf_nodes=max_leaf_nodes,
        )


class BoostCardRegressor(BaseBoostCard, RegressorMixin):
    def __init__(
        self,
        # xgb params
        constraints: List[Constraint],
        objective: str = "reg:squarederror",
        eta: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            eta=eta,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
        )

