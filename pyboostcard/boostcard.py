from __future__ import annotations

from pyboostcard.decisionstump import DecisionStump
from pyboostcard.constraints import *
from pyboostcard.util import *

from typing import Dict

from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_consistent_length


class BaseBoostCard(BaseEstimator):
    def __init__(self, constraints: List[Constraint], xgb_params: Dict = None, tree_params: Dict = None) -> None:
        self.constraints = constraints
        self.xgb_params = xgb_params
        self.tree_params = tree_params


class BoostCardClassifier(BaseBoostCard, ClassifierMixin):
    # list of constraints

    def __init__(self, constraints: List[Constraint], xgb_params: Dict = None, tree_params: Dict = None) -> None:
        super().__init__(constraints=constraints, xgb_params=xgb_params, tree_params=tree_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BoostCardClassifier:
        check_consistent_length(X, y)
        clf = XGBClassifier(**self.xgb_params)

        # TODO: do fitting stuff
        #
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_predictions = self.decision_function()
        pass
