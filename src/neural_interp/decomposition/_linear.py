import numpy as np

from neural_interp.typing import FloatArray


def linear_contributions(inputs: FloatArray, weight: FloatArray) -> FloatArray:
    return np.asarray(inputs) * np.asarray(weight)[None, :]
