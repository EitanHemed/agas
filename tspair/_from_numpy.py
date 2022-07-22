"""
Functions for applying Agas on a NumPy array.
"""

import typing

import numpy as np
import numpy.typing as npt

__all__ = ['pair_from_array']


def pair_from_array(array: npt.NDArray,
                    maximize: typing.Callable,
                    minimize: typing.Callable,
                    maximize_weight: typing.Union[float, int] = 0.5,
                    return_indices: bool = False):
    """

    :param array:
    :param maximize:
    :param minimize:
    :param maximize_weight:
    :param return_indices:
    :return:
    """
    assert array.ndim == 2
    assert np.size != 0
    assert (0 <= maximize_weight) & (maximize_weight <= 1)

    minimize_weight = 1 - maximize_weight

    array = array.copy()

    normalized_max_sim_diffs_mat = _normalize_differences(array, maximize)
    normalized_min_sim_diffs_mat = _normalize_differences(array, minimize)

    optimized_differences = _optimize(normalized_max_sim_diffs_mat,
                                      normalized_min_sim_diffs_mat,
                                      maximize_weight, minimize_weight
                                      )

    optimal_pair = _find_optimal(optimized_differences)

    if return_indices is True:
        return optimal_pair

    return array[optimal_pair, :]


def _normalize_differences(array: npt.NDArray, func: typing.Callable):
    aggregated_array = func(array, axis=1)
    diffs_mat = _get_diffs_matrix(aggregated_array)
    normalized_diffs_mat = _normalize(diffs_mat)
    return normalized_diffs_mat


def _get_diffs_matrix(array: npt.NDArray):
    """
    Return a matrix of the absolute difference between each element and all
    other elements in the array.

    :param array: npt.NDArray of size 1 X n.
    :return: npt.NDArray of size n X n.
    """
    diffs_mat = np.abs(np.subtract.outer(array, array))
    # We can ignore the difference between each sample and itself
    np.fill_diagonal(diffs_mat, np.nan)
    return diffs_mat


def _normalize(a: npt.NDArray):
    """Normalize an array to the range between 0 and 1.

    :param a : npt.NDArray
        A 1D array to normalize.

    :return:
        The normalized array, between 0 and 1.
    """
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))


def _optimize(maximize_similarity_array, minimize_similarity_array,
              maximize_weight, minimize_weight):
    """
    Calculates the weighted average of the two similarity arrays, based on their
    respective weights.

    :param maximize_similarity_array:
    :param minimize_similarity_array:
    :param maximize_weight: Float in the range [0.0, 1.0]
    :param minimize_weight: Float in the range [0.0, 1.0],
        Complementary of `maximize_weight`.
    :return:

    """
    return maximize_similarity_array * maximize_weight + (
            np.reciprocal(minimize_similarity_array) * minimize_weight
    )


def _find_optimal(optimized_differences):
    return np.unravel_index(
        np.nanargmin(optimized_differences.flatten()),
        optimized_differences.shape)
