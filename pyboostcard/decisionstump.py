from typing import List, Tuple, Dict
import numpy as np


class DecisionStump:
    def __init__(self, features: List[Tuple[int, float]], values: List[Tuple[float, float]]):
        ## aggregate tuples by feature and thresholds and sum values
        fm: Dict[Tuple[int, float], Tuple[float, float]] = {}

        for i, f in enumerate(features):
            tmp = fm.get(f, (0.0, 0.0))
            tmp = (tmp[0] + values[i][0], tmp[1] + values[i][1])
            fm[f] = tmp

        ## sort the keys and reindex
        self._feature_map = {k: fm[k] for k in sorted(fm.keys())}

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape[0])

        for (feature, threshold), (left, right) in self._feature_map.items():
            out += np.where(x[:, feature] < threshold, left, right)

        return out
