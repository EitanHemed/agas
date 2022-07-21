"""
tspair is a small library for matching pairs of time-series while maximizing
similarity on one dimension and maximizing dissimilarity on another.
"""
import itertools
import typing

import numpy as np
import numpy.typing as npt

__all__ = ['pair_from_array']


def pair_from_array(array: npt.NDArray, maximize: typing.Callable,
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

    max_similarity_aggregated = _aggregate_array(array, maximize)
    min_similarity_aggregated = _aggregate_array(array, minimize)

    max_sim_func_diffs_mat = get_diffs_matrix(max_similarity_aggregated)
    min_sim_func_diffs_mat = get_diffs_matrix(min_similarity_aggregated)

    normalized_max_sim_diffs_mat = normalize(max_sim_func_diffs_mat)
    normalized_min_sim_diffs_mat = normalize(min_sim_func_diffs_mat)

    # # For the max_mat we need to maximize similarity (i.e., get the smallest
    # # difference).
    # minimized_differences_vector = np.nanargmin(max_sim_func_diffs_mat, axis=1)
    # # For the min_mat we need to minimize similarity (i.e., get the largest
    # # difference).
    # maximized_differences_vector = np.nanargmax(min_sim_func_diffs_mat, axis=1)
    #
    # # Normalize all differences between 0 and 1
    # norm_min_diff_vec = normalize(minimized_differences_vector)
    # norm_max_diff_vec = normalize(maximized_differences_vector)

    # Return the index of pairs
    optimized_differences = optimize(normalized_max_sim_diffs_mat, normalized_min_sim_diffs_mat,
                            maximize_weight, minimize_weight
                            )

    optimal_indices = np.nanargmin(optimized_differences.flatten())

    sample_indices = np.arange(array.shape[0])
    indice_pairs = list(itertools.product(sample_indices, sample_indices))

    if return_indices is True:
        return indice_pairs[optimal_indices]

    return array[indice_pairs[optimal_indices], :]

def _aggregate_array(array: npt.NDArray, func: typing.Callable) -> npt.NDArray:
    return func(array, axis=1)

def get_diffs_matrix(array: npt.NDArray):
    """"""
    # Get the difference between each sample and all samples (including itself)
    diffs_mat = np.abs(np.subtract.outer(array, array))
    # We can ignore the difference between each sample and itself
    np.fill_diagonal(diffs_mat, np.nan)
    return diffs_mat

def normalize(a: npt.NDArray):
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

def optimize(maximize_similarity_array, minimize_similarity_array,
                      maximize_weight, minimize_weight):
    optimized = maximize_similarity_array * maximize_weight + (
            np.reciprocal(minimize_similarity_array) * minimize_weight
    )

    # Handle infinite values introduced by calculating reciprocal of 0
    # optimized[np.isinf(optimized)] = 1

    return optimized

#
# def pair_from_df():
#     pass
#
#
# def _process_dataframe(df: pd.DataFrame,
#                        maximize_weight: typing.Union[float, int] = 0.5):
#     assert df.size != 0
#
#     df = df.copy()
