from __future__ import annotations

from pyboostcard.decisionstump import DecisionStump
from pyboostcard.constraints import *
from pyboostcard import util
import operator as op
from functools import reduce
from itertools import accumulate

from typing import Dict, List, Tuple, Union, cast, Any, Optional 
import copy

from xgboost.sklearn import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_consistent_length


import json


## TODO: implement preprocess mixin


class BaseBoostCard(BaseEstimator):
    @staticmethod
    def from_json(file: str) -> List[Constraint]:

        with open(file) as f:
            config = json.load(f)

        constraints: List[Constraint] = []
        for k, v in config.items():
            sels = [Selection.from_dict(sel) for sel in v]
            constraints.append(Constraint(*sels, name=k))

        return constraints

    def __init__(
        self,
        constraints: Union[str, List[Constraint]],
        objective: str = "reg:squarederror",
        n_estimators: int = 100,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        max_leaf_nodes: int = 8,
    ) -> None:

        if isinstance(constraints, str):
            self.constraints = BaseBoostCard.from_json(constraints)
        else:
            self.constraints = constraints

        self.objective = objective
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        
        self.max_leaf_nodes = max_leaf_nodes
        self.decision_stumps: List[DecisionStump] = []

        self.xgboost = XGBRegressor

    def fit(  # type: ignore
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        eval_metric=None
    ) -> BaseBoostCard:
        check_consistent_length(X, y)

        # transform input data through constraints
        xs, monos, lens = self.transform(X)

        self.xgb = self.xgboost(
            max_depth=1,  # hard-coded
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=1.0,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            subsample=1.0,
            monotone_constraints=str(tuple(monos)),
        )

        self.xgb.fit(
            np.concatenate(xs, axis=1),
            y,
            sample_weight=sample_weight,
            eval_metric=eval_metric,
        )

        # decision tree is always a regressor because we are using xgboost output as y
        clf = DecisionTreeRegressor(
            min_samples_leaf=self.min_child_weight, max_leaf_nodes=self.max_leaf_nodes
        )

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
                    tuples.append(
                        (np.nan, sel.override, float(stump.transform(_x, pos)))
                    )
                # check if interval
                elif isinstance(sel, Interval):
                    # Push these to a separate list and combine them before
                    # adding them to the _bins list
                    yhat = stump.transform(x, pos)
                    # filter X to only where it is in selection
                    f = sel.in_selection(X[[constraint.name]]).reshape(-1)
                    clf.fit(X[[constraint.name]][f], yhat[f])
                    intervals += util.sklearn_tree_to_bins(clf.tree_, values=sel.values)

                elif isinstance(sel, Identity):
                    yhat = stump.transform(x, pos)
                    clf.fit(X[[constraint.name]], yhat)
                    tuples += util.sklearn_tree_to_bins(
                        clf.tree_, values=(-np.inf, np.inf)
                    )
                else:
                    pass

            self._bins[constraint.name] = tuples + sorted(intervals)
            pos += x.shape[1]

        return self

    def transform(
        self, X: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[int], List[int]]:

        xs: List[np.ndarray] = []
        monos: List[int] = []

        for constraint in self.constraints:
            data, mono = constraint.transform(X[[constraint.name]])
            xs.append(data)
            monos += mono

        lens: List[int] = [x.shape[1] for x in xs]

        return xs, monos, lens

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def decision_function(
        self, X: pd.DataFrame, columns: bool = False, order: str = "F"
    ) -> np.ndarray:
        ## check that the names of the dataframe are found in self._bins
        diff = set(self._bins.keys()).difference(set(X.columns))
        if len(diff) > 0:
            raise ValueError("Required columns not found in `X`: {diff}")

        ## loop over each of the _bins
        out: List[np.ndarray] = []
        for k, v in self._bins.items():
            x = X[k]
            res = np.full_like(x, fill_value=np.nan, dtype="float")

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
            return {k: v for k, v in zip(self._bins.keys(), out)}
        else:
            # sum the columns and add the intercept ... # TODO add intercept
            # prob = util.sigmoid(np.sum(cols, axis=1))
            # return np.hstack([1. - prob, prob])
            return np.sum(cols, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_predictions = self.decision_function(X, False)
        pass


class BoostCardClassifier(BaseBoostCard, ClassifierMixin):
    def __init__(
        self,
        # xgb params
        constraints: Union[str, List[Constraint]],
        objective: str = "binary:logitraw",
        n_estimators: int = 100,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            n_estimators=n_estimators,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_leaf_nodes=max_leaf_nodes,
        )

        self.xgboost = XGBClassifier

    def fit(  # type: ignore
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model=None,
        sample_weight_eval_set=None,
        callbacks=None,
    ) -> BaseBoostCard:
        self.classes_, y = np.unique(y, return_inverse=True)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_metric=eval_metric,            
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        D = self.decision_function(X)
        return self.classes_[np.where(D > 0, 1, 0)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        D = self.decision_function(X)
        prob = util.sigmoid(D)
        return np.hstack([1.0 - prob, prob])

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.decision_function(X)


class BoostCardRegressor(BaseBoostCard, RegressorMixin):
    def __init__(
        self,
        # xgb params
        constraints: Union[str, List[Constraint]],
        objective: str = "reg:squarederror",
        n_estimators: int = 100,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            n_estimators=n_estimators,
            gamma=gamma,
            min_child_weight=min_child_weight,
        )

        self.xgboost = XGBRegressor
