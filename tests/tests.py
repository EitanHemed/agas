"""
tspair is a small library for matching pairs of time-series while maximizing
similarity on one dimension and maximizing dissimilarity on another.
"""
import numpy as np
import tspair
import pytest

def test_array():
    length = 10
    inp_vals = np.vstack((np.zeros(length), np.ones(length) * 10,
               np.arange(length))
              )

    print(tspair.pair_from_array(inp_vals, maximize=np.std, minimize=np.mean))
