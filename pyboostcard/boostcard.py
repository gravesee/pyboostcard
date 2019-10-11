from __future__ import annotations

from pyboostcard.decisionstump import DecisionStump
from pyboostcard.constraints import *
from pyboostcard import util
import operator as op
from functools import reduce
from itertools import accumulate

from typing import Dict, List, Tuple
import copy

from xgboost.sklearn import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_consistent_length


class BaseBoostCard(BaseEstimator):
    def __init__(
        self,
        constraints: List[Constraint],
        objective: str = "reg:squarederror",
        learning_rate: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        max_leaf_nodes: int = 8,
    ) -> None:

        self.constraints = constraints
        self.objective = objective
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample

        self.max_leaf_nodes = max_leaf_nodes
        self.decision_stumps: List[DecisionStump] = []

        self.xgboost = XGBRegressor

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostCardClassifier:
        check_consistent_length(X, y)

        ## transform featueres using the fitted constraints
        xs: List[np.ndarray] = []
        monos: List[int] = []
        for constraint in self.constraints:
            data, mono = constraint.transform(X[[constraint.name]])
            xs.append(data)
            monos += mono

        # create list of number of cols produced for each feature
        lens: List[int] = [x.shape[1] for x in xs]
        
        self.xgb = self.xgboost(
            max_depth=1,  # hard-coded
            objective=self.objective,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            monotone_constraints=str(tuple(monos)),
        )

        self.xgb.fit(np.concatenate(xs, axis=1), y)

        # decision tree is always a regressor because we are using xgboost output as y
        clf = DecisionTreeRegressor(min_samples_leaf=self.min_child_weight, max_leaf_nodes=self.max_leaf_nodes)

        ## dump the model and generate the decision stumps
        mod_data = util.split_xgb_outputs(self.xgb, lens)

        for data in mod_data:
            self.decision_stumps.append(DecisionStump(data[0], data[1]))

        # generate data for the decision trees
        self._trees: List[Tree] = []
        self._bins: Dict[str, List[Tuple[float, ...]]] = {}
        pos = 0
        for x, stump, constraint in zip(xs, self.decision_stumps, self.constraints):

            tuples: List[Tuple[float, ...]] = []
            ## loop over selections, transform and predict using decision tree

            intervals: List[Tuple[float, ...]] = []
            for sel in constraint.selections:

                # check if missing
                if isinstance(sel, Missing):
                    _x, _ = constraint.transform(np.array(np.nan).reshape(-1, 1))
                    tuples.append((np.nan, np.nan, float(stump.transform(_x, pos))))
                # check if override
                elif isinstance(sel, Override):
                    
                    _x, _ = constraint.transform(np.array(sel.override).reshape(-1, 1))
                    tuples.append((np.nan, sel.override, float(stump.transform(_x, pos))))
                # check if interval
                elif isinstance(sel, Interval):
                    # Push these to a separate list and combine them before
                    # adding them to the _bins list
                    yhat = stump.transform(x, pos)
                    # filter X to only where it is in selection
                    f = sel.in_selection(X[[constraint.name]]).reshape(-1)
                    clf.fit(X[[constraint.name]][f], yhat[f])

                    intervals += util.sklearn_tree_to_bins(clf.tree_, values=sel.values)
                else:
                    pass

            self._bins[constraint.name] = tuples + sorted(intervals)
            pos += x.shape[1]

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        pass    

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def decision_function(self, X: pd.DataFrame, columns: bool = False, order: str = 'F') -> np.ndarray:
        ## check that the names of the dataframe are found in self._bins
        diff = set(self._bins.keys()).difference(set(X.columns))
        if len(diff) > 0:
            raise ValueError("Required columns not found in `X`: {diff}")

        ## loop over each of the _bins
        out: List[np.ndarray] = []
        for k, v in self._bins.items():
            x = X[k]
            res = np.full_like(x,fill_value=np.nan, dtype='float')

            # loop over the bin intervals (start, stop, value)
            # TODO: refactor this so it isn't just using un-named tuples
            for el in v:
                # missing or override
                if el[0] is np.nan:
                    # missing
                    if el[1] is np.nan:
                        res[np.isnan(x)] = el[2]
                    # override
                    else:
                        res[x == el[1]] = el[2]
                # interval TODO: add bounds to the tuple and get comparison op here
                else:
                    f = (x >= el[0]) & (x <= el[1])
                    res[f] = el[2]
            
            out.append(res) 

        cols = np.hstack(out).reshape(-1, len(out), order=order)
        
        if columns:
            return cols
        else:
            # sum the columns and add the intercept ...
            return np.sum(cols, axis=1)
        

    def predict(self, X: pd.DataFrame, columns: bool = False) -> np.ndarray:
        raw_predictions = self.decision_function(X, columns)
        pass


class BoostCardClassifier(BaseBoostCard, ClassifierMixin):
    def __init__(
        self,
        # xgb params
        constraints: List[Constraint],
        objective: str = "binary:logitraw",
        learning_rate: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            max_leaf_nodes=max_leaf_nodes,
        )

        self.xgboost = XGBClassifier


class BoostCardRegressor(BaseBoostCard, RegressorMixin):
    def __init__(
        self,
        # xgb params
        constraints: List[Constraint],
        objective: str = "reg:squarederror",
        learning_rate: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            learning_rate=learning_rate,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
        )

        self.xgboost = XGBRegressor
