

import numpy as np
import pytest
import tspair


@pytest.mark.parametrize('return_indices', [False, True])
def test_toy_data(return_indices):
    length = 10

    inp_vals = np.vstack((np.zeros(length),
                          np.ones(length) * 10,
                          np.arange(5, length + 5, )
                          )
                         )

    expected_return = (0, 1) if return_indices else inp_vals[(0, 1), :]
    assert np.array_equal(tspair.pair_from_array(
        inp_vals, maximize=np.std, minimize=np.mean,
        return_indices=return_indices), expected_return)

    expected_return = (1, 2) if return_indices else inp_vals[(1, 2), :]
    assert np.array_equal(tspair.pair_from_array(
        inp_vals, maximize=np.mean, minimize=np.std,
        return_indices=return_indices), expected_return)


@pytest.mark.parametrize('return_indices', [False, True])
def test_sim_data(return_indices):
    rng = np.random.RandomState(2022)
    means = np.array((10, 12, 0, -2))
    sds = np.array((1, 3, 1, 7))
    N = 100

    arrays = np.vstack([rng.normal(m, sd, N) for (m, sd) in zip(means, sds)])

    expected_return = (2, 3) if return_indices else arrays[(2, 3), :]
    assert np.array(tspair.pair_from_array(
        arrays, maximize=np.mean, minimize=np.std,
        return_indices=return_indices), expected_return)

    expected_return = (1, 3) if return_indices else arrays[(1, 3), :]
    assert np.array_equal(tspair.pair_from_array(
        arrays, maximize=np.std, minimize=np.mean,
        return_indices=return_indices), expected_return)

    expected_return = (0, 2) if return_indices else arrays[(0, 2), :]
    assert np.array_equal(tspair.pair_from_array(
        arrays, maximize=np.std, minimize=np.mean, maximize_weight=0.95,
        return_indices=return_indices), expected_return)

    expected_return = (0, 3) if return_indices else arrays[(0, 3), :]
    assert np.array_equal(tspair.pair_from_array(
        arrays, maximize=np.mean, minimize=np.std, maximize_weight=0,
        return_indices=return_indices), expected_return)
