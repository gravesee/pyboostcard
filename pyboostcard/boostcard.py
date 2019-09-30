from __future__ import annotations

from pyboostcard.decisionstump import DecisionStump
from pyboostcard.constraints import *
from pyboostcard import util

from typing import Dict

from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_consistent_length


class BaseBoostCard(BaseEstimator):
    def __init__(
        self,
        constraints: List[Constraint],
        objective: str = "reg:squarederror",
        eta: float = 0.3,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        **kwargs,
    ) -> None:

        self.constraints = constraints
        self.objective = objective
        self.eta = eta
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.kwargs = kwargs

        self.max_leaf_nodes = max_leaf_nodes
        self.decision_stumps: List[DecisionStump] = []

        params = self.kwargs.update(
            {
                "max_depth": 1,  # hard-coded
                "objective": self.objective,
                "eta": self.eta,
                "gamma": self.gamma,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
            }
        )

        self.xgb = XGBClassifier(params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostCardClassifier:
        check_consistent_length(X, y)

        ## transform featueres using the fitted constraints
        xs, monos = [], []
        for constraint in self.constraints:
            _x, _m = constraint.transform(X[[constraint.name]])
            xs.append(_x)
            monos.append(_m)

        lens: list[int] = [x.shape[1] for x in xs]

        self.xgb.set_params({"monotone_constraints": str(tuple(monos))})

        self.xgb.fit(np.concatenate(xs, axis=1), y)

        ## dump the model and generate the decision stumps
        mod_data = util.split_xgb_outputs(self.xgb, lens)

        for data in mod_data:
            self.decision_stumps.append(DecisionStump(data[0], data[1]))
        
        # generate data for the decision trees
        for x, stump, constraint in zip(xs, self.decision_stumps, self.constraints):
            yhat = stump.transform(x)
            self.clf.fit(X[[constraint.name]], yhat)        
        
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
        **kwargs,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objetive=objective,
            eta=eta,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            kwargs=kwargs,
        )

        self.clf = DecisionTreeClassifier(min_weight_fraction_leaf=min_child_weight, max_leaf_nodes=self.max_leaf_nodes)


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
        **kwargs,
    ) -> None:

        super().__init__(
            constraints=constraints,
            objetive=objective,
            eta=eta,
            gamma=gamma,
            min_child_weight=min_child_weight,
            subsample=subsample,
            kwargs=kwargs,
        )

        self.clf = DecisionTreeRegressor(min_weight_fraction_leaf=min_child_weight, max_leaf_nodes=self.max_leaf_nodes)

