import numpy as np
import pytest

import agas
from agas._from_numpy import (RETURN_TYPE_OPTIONS, RETURN_INDICES,
                              RETURN_VALUES)

TOY_DATA = np.vstack((np.zeros(10), np.ones(10) * 10, np.arange(5, 15, )))


def _return_type_helper(array: np.ndarray, return_type: str, indices: tuple):
    if return_type == RETURN_INDICES:
        return indices
    elif return_type == RETURN_VALUES:
        return array[indices, :]
    else:
        raise RuntimeError("Unknown return_type")


@pytest.mark.parametrize('return_type', RETURN_TYPE_OPTIONS)
def test_toy_data(return_type):
    inp_vals = TOY_DATA

    expected_return = _return_type_helper(inp_vals, return_type, (0, 1))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.std, minimize_function=np.mean,
        return_type=return_type), expected_return)

    expected_return = _return_type_helper(inp_vals, return_type, (1, 2))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.mean, minimize_function=np.std,
        return_type=return_type), expected_return)


@pytest.mark.parametrize('return_type', RETURN_TYPE_OPTIONS)
def test_sim_data(return_type):
    rng = np.random.RandomState(2022)
    means = np.array((10, 12, 0, -2))
    sds = np.array((1, 3, 1, 7))
    N = 100

    inp_vals = np.vstack([rng.normal(m, sd, N) for (m, sd) in zip(means, sds)])

    expected_return = _return_type_helper(inp_vals, return_type, (2, 3))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.mean, minimize_function=np.std,
        return_type=return_type), expected_return)

    expected_return = _return_type_helper(inp_vals, return_type, (1, 3))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.std, minimize_function=np.mean,
        return_type=return_type), expected_return)

    expected_return = _return_type_helper(inp_vals, return_type, (0, 2))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.std, minimize_function=np.mean,
        maximize_weight=0.95,
        return_type=return_type), expected_return)

    expected_return = _return_type_helper(inp_vals, return_type, (0, 3))
    assert np.array_equal(agas.pair_from_array(
        inp_vals, maximize_function=np.mean, minimize_function=np.std,
        maximize_weight=0,
        return_type=return_type), expected_return)


def test__normalize_differences():
    def mean_with_some_nans(a: np.ndarray):
        res = np.mean(a, axis=1)
        res[[0, -1]] = np.nan
        return res

    def return_all_nans(a: np.ndarray):
        res = np.empty(a.shape[0])
        res.fill(np.nan)
        return res

    with pytest.raises(ValueError):
        agas.pair_from_array(TOY_DATA,
                             maximize_function=mean_with_some_nans,
                             minimize_function=np.std, )
        agas.pair_from_array(TOY_DATA,
                             maximize_function=mean_with_some_nans,
                             minimize_function=mean_with_some_nans, )

    with pytest.raises(ValueError):
        agas.pair_from_array(TOY_DATA,
                             maximize_function=return_all_nans,
                             minimize_function=np.std, )
        agas.pair_from_array(TOY_DATA,
                             maximize_function=return_all_nans,
                             minimize_function=return_all_nans, )

def test_input_arrays():
    np.array_equal(
        agas.pair_from_array(TOY_DATA.tolist(), maximize_function=np.mean,
                             minimize_function=np.std, ), (0, 3))

    with pytest.raises(ValueError):
        agas.pair_from_array(TOY_DATA.tolist()[0], maximize_function=np.mean,
                             minimize_function=np.std, )
        agas.pair_from_array(None, maximize_function=np.mean,
                             minimize_function=np.std, )

    with pytest.raises(RuntimeError):
        (agas.pair_from_array(TOY_DATA[[0], :], maximize_function=np.mean,
                              minimize_function=np.std, ))

        (agas.pair_from_array(np.empty((0,)), maximize_function=np.mean,
                              minimize_function=np.std, ))
        (agas.pair_from_array(np.empty((0, 0)), maximize_function=np.mean,
                              minimize_function=np.std, ))

def test_input_functions():

    with pytest.raises(TypeError):

        agas.pair_from_array(TOY_DATA, maximize_function=np.mean,
                             minimize_function=None, )
        agas.pair_from_array(TOY_DATA, maximize_function=None,
                             minimize_function=np.std, )
        agas.pair_from_array(TOY_DATA, maximize_function=None,
                             minimize_function=None, )


def test_maximize_weight():
    _kwargs = {'input_array': TOY_DATA, 'maximize_function': np.mean,
               'minimize_function': np.std, }

    with pytest.raises(ValueError):
        agas.pair_from_array(**_kwargs, maximize_weight=5)
        agas.pair_from_array(**_kwargs, maximize_weight=-0.5)
    with pytest.raises(TypeError):
        agas.pair_from_array(**_kwargs, maximize_weight=[5])
        agas.pair_from_array(**_kwargs, maximize_weight='0.5')


def test_return_type():
    with pytest.raises(ValueError):
        agas.pair_from_array(TOY_DATA, maximize_function=np.mean,
                             minimize_function=np.std,
                             return_type='array values')
    with pytest.raises(TypeError):
        agas.pair_from_array(TOY_DATA, maximize_function=np.mean,
                             minimize_function=np.std,
                             return_type=1)
        agas.pair_from_array(TOY_DATA, maximize_function=np.mean,
                             minimize_function=np.std,
                             return_type=[])
        agas.pair_from_array(TOY_DATA, maximize_function=np.mean,
                             minimize_function=np.std,
                             return_type=None)


def test__apply_func():
    a = np.array([[0, 1], [2, 3]])
    a_sum = np.array([1, 5])

    assert np.array_equal(agas._from_numpy._apply_func(a, sum), a_sum)
    assert np.array_equal(agas._from_numpy._apply_func(a, np.sum), a_sum)

    with pytest.raises(AttributeError):
        assert np.array_equal(agas._from_numpy._apply_func(a.tolist(),
                                                           sum), a_sum)

def test__get_diffs_matrix():
    a = np.array([1, 2, 3])
    expected_return = np.array([[np.nan, 1, 2],
                                [1, np.nan, 1],
                                [2, 1, np.nan]])

    assert np.array_equal(agas._from_numpy._get_diffs_matrix(a),
                          expected_return, equal_nan=True)

def test__normalize():
    np.array_equal(agas._from_numpy._normalize(np.array([1, 2])),
                   np.array([0, 1]))
    np.array_equal(agas._from_numpy._normalize(np.array([-100, 100])),
                   np.array([0, 1]))
    np.array_equal(agas._from_numpy._normalize(np.array([0, 10, 1000])),
                   np.array([0, 0.01, 1]))

