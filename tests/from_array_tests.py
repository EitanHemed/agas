"""
tspair is a small library for matching pairs of time-series while maximizing
similarity on one dimension and maximizing dissimilarity on another.
"""

import pytest
import numpy as np

import tspair


def test_toy_data():
    length = 10

    inp_vals = np.vstack((np.zeros(length), np.ones(length) * 10,
                          np.arange(5, length + 5, ))
                         )
    assert tspair.pair_from_array(inp_vals, maximize=np.std, minimize=np.mean,
                           return_indices=True) == (0, 1)


    assert tspair.pair_from_array(inp_vals, maximize=np.mean, minimize=np.std,
                                  return_indices=True) == (1, 2)
    #
    # inp_vals = np.vstack((np.zeros(length), np.ones(length) * 4.5,
    #            np.arange(length))
    #           )
    # assert tspair.pair_from_array(inp_vals, maximize=np.mean, minimize=np.std,
    #                        return_indices=True) == (1, 2)

def test_sim_data():
    rng = np.random.RandomState(2022)
    means = np.array((10, 12, 0, -2))
    sds = np.array((1, 3, 1, 7))
    N = 100

    arrays = np.vstack([rng.normal(m, sd, N) for (m, sd) in zip(means, sds)])

    assert tspair.pair_from_array(
        arrays, maximize=np.mean, minimize=np.std,
            return_indices=True) == (2, 3)

    assert tspair.pair_from_array(
        arrays, maximize=np.std, minimize=np.mean,
        return_indices=True) == (1, 3)

assert tspair.pair_from_array(
        arrays, maximize=np.std, minimize=np.mean,
        return_indices=True, maximize_weight=0.95) == (0, 2)

assert tspair.pair_from_array(
        arrays, maximize=np.mean, minimize=np.std,
        return_indices=True, maximize_weight=0) == (0, 3)