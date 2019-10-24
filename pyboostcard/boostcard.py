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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_consistent_length


import matplotlib.pyplot as plt


import json

from collections import namedtuple

Level = namedtuple("Level", ["ll", "ul", "value"])

## TODO: implement preprocess mixin


class BinnedVar:
    def __init__(self, levels: List[Level], constraint: Constraint, x: pd.Series) -> None:
        self.levels = levels
        ## get min/max values from constraints .. if Clamp, then use those values
        self.mn = x.min(skipna=True)
        self.mx = x.max(skipna=True)
        
        for sel in constraint.selections:
            if isinstance(sel, Clamp):
                self.mn, self.mx = sel.ll, sel.ul

    def transform(self, x: np.ndarray) -> np.ndarray:
        res = np.full_like(x, fill_value=np.nan, dtype="float")

        # loop over the bin intervals (start, stop, value)
        for el in self.levels:
            # missing or override
            if el.ll is np.nan:
                # missing
                if el.ul is np.nan:
                    res[np.isnan(x)] = el.value
                # override
                else:
                    res[x == el.ul] = el.value
            # interval TODO: add bounds to the tuple and get comparison op here
            else:
                f = (x >= el.ll) & (x <= el.ul)
                res[f] = el.value

        return res

    def get_overrides(self) -> Tuple[List[float], List[float]]:
        """ return arrays for overrides and values"""
        ors: List[float] = []
        vals: List[float] = []
        for l in self.levels:
            if np.isnan(l.ll):
                if not np.isnan(l.ul):
                    ors.append(l.ul)
                    vals.append(l.value)

        return ors, vals

    def get_missing(self) -> List[float]:
        for l in self.levels:
            if np.isnan(l.ll) and np.isnan(l.ul):
                return [l.value]
        return []
    
    def plot(self, resolution: int = 50) -> None:
        ## get interval min and max (excluding np.inf)
        mn, mx = self.mn, self.mx
        space = (mx - mn) / resolution

        overrides, override_vals = self.get_overrides()
        missing = self.get_missing()
        
        points = np.linspace(mn, mx, num=resolution)
        yhats = list(self.transform(points))

        ## these should be added as points
        special = missing + overrides

        ## these added as lines
        ys = missing + override_vals + yhats
        xs = list(np.linspace(mn - space * len(special), mn, len(special))) + list(points)

        plt.plot(xs[:len(special)], ys[:len(special)], 'ro')
        plt.plot(xs[len(special):], ys[len(special):])
        plt.axis([min(xs), max(xs), min(ys), max(ys)])
        # plt.xticks(xs, )
        plt.show()

        # figure out axis labels

        # pass


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
        learning_rate: float = 0.30,
        subsample: float = 0.5,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        max_leaf_nodes: int = 8,
        **kwargs: Dict[str, Any]
    ) -> None:

        if isinstance(constraints, str):
            self.constraints = BaseBoostCard.from_json(constraints)
        else:
            self.constraints = constraints

        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.max_leaf_nodes = max_leaf_nodes
        self.decision_stumps: List[DecisionStump] = []

        self.kwargs = kwargs

        self.xgboost = XGBRegressor

    def fit(  # type: ignore
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None, eval_metric=None
    ) -> BaseBoostCard:
        check_consistent_length(X, y)

        # transform input data through constraints
        xs, monos, lens = self.transform(X)

        self.xgb = self.xgboost(
            tree_method="hist",
            grow_policy="lossguide",
            max_depth=1,  # hard-coded
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            monotone_constraints=str(tuple(monos)),
            **self.kwargs
        )

        self.xgb.fit(np.concatenate(xs, axis=1), y, sample_weight=sample_weight, eval_metric=eval_metric)

        # decision tree is always a regressor because we are using xgboost output as y
        clf = DecisionTreeRegressor(min_samples_leaf=self.min_child_weight, max_leaf_nodes=self.max_leaf_nodes)

        ## dump the model and generate the decision stumps
        mod_data = util.split_xgb_outputs(self.xgb, lens)

        for data in mod_data:
            self.decision_stumps.append(DecisionStump(data[0], data[1]))

        # generate data for the decision trees
        self._trees: List[Tree] = []
        # self._bins: Dict[str, List[Tuple[float, ...]]] = {}

        self._bins: Dict[str, BinnedVar] = {}

        pos = 0
        for x, stump, constraint in zip(xs, self.decision_stumps, self.constraints):

            tuples: List[Tuple[float, ...]] = []
            ## loop over selections, transform and predict using decision tree

            intervals: List[Tuple[float, ...]] = []
            for sel in constraint.selections:

                # TODO: Push bin creation into the selection class?

                # check if missing
                if isinstance(sel, Missing):
                    _x, _ = constraint.transform(np.array(np.nan).reshape(-1, 1))
                    tuples.append(Level(np.nan, np.nan, float(stump.transform(_x, pos))))
                # check if override
                elif isinstance(sel, Override):

                    _x, _ = constraint.transform(np.array(sel.override).reshape(-1, 1))
                    tuples.append(Level(np.nan, sel.override, float(stump.transform(_x, pos))))
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
                    tuples += util.sklearn_tree_to_bins(clf.tree_, values=(-np.inf, np.inf))
                else:
                    pass

            levels = [Level(*x) for x in tuples + sorted(intervals)]
            self._bins[constraint.name] = BinnedVar(levels, constraint, X[constraint.name])
            pos += x.shape[1]

        return self

    def lengths(self) -> List[int]:
        return [len(x) for x in self.constraints]

    @property
    def features(self) -> List[str]:
        return [x.name for x in self.constraints]

    @property
    def feature_importances_(self) -> pd.DataFrame:
        fi = self.xgb.feature_importances_
        idxs = util.lengths_to_indices(self.lengths())
        data = [(k, np.sum(fi[i])) for k, i in zip(self.features, idxs)]
        res = pd.DataFrame.from_records(data, columns=["Feature", self.xgb.importance_type])
        return res.sort_values(by=[self.xgb.importance_type], ascending=False)

    def transform(self, X: pd.DataFrame) -> Tuple[List[np.ndarray], List[int], List[int]]:

        xs: List[np.ndarray] = []
        monos: List[int] = []

        for constraint in self.constraints:
            data, mono = constraint.transform(X[constraint.name])
            xs.append(data)
            monos += mono

        lens: List[int] = [x.shape[1] for x in xs]

        return xs, monos, lens

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def decision_function(self, X: pd.DataFrame, columns: bool = False, order: str = "F") -> np.ndarray:
        ## check that the names of the dataframe are found in self._bins
        diff = set(self._bins.keys()).difference(set(X.columns))
        if len(diff) > 0:
            raise ValueError("Required columns not found in `X`: {diff}")

        out: List[np.ndarray] = []
        for key, binned_var in self._bins.items():
            out.append(binned_var.transform(X[key]))

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
        learning_rate: float = 0.30,
        subsample: float = 0.5,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
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
        return super().fit(X, y, sample_weight=sample_weight, eval_metric=eval_metric)

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
        learning_rate: float = 0.30,
        subsample: float = 0.5,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        # tree params
        max_leaf_nodes: int = 8,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            gamma=gamma,
            min_child_weight=min_child_weight,
        )

        self.xgboost = XGBRegressor

