""""""

import inspect
import typing
import warnings

import numpy as np
import numpy.typing as npt

__all__ = ['pair_from_array']

RETURN_INDICES = 'indices'
RETURN_VALUES = 'values'
RETURN_TYPE_OPTIONS = {RETURN_INDICES, RETURN_VALUES}


def pair_from_array(input_array,
                    maximize_function: typing.Callable,
                    minimize_function: typing.Callable,
                    maximize_weight: typing.Union[float, int] = 0.5,
                    return_type: str = 'indices'):
    r"""
    Find the optimal pair of 1D arrays given 2D array.

    The optimality of the pairing of two vectors is determined by a weighted
    average of their absolute differences, following normalization of the
    differences between an aggregated vectors and each of the other vectors
    (excluding itself).

    Aggregation is performed using each of the aggregation functions
    `maximize_function` and `minimize_function`, weighted by `maximize_function`
    and 1 - `maximize_function`, respectively.

    Parameters
    ----------
    input_array: array-like
        A 2D array of shape (N, T).
    maximize_function, minimize_function: Callable
        Used to aggregate `input_array` along the 2-nd axis and find similar
        pairs - those with minimal absolute difference between their
        aggregated values. `minimize_function` is identical
        except it is used to find pairs of arrays with maximal absolute
        differnce between aggregated values.
    maximize_weight: Int, Float, default=0.5
        Used to weight the `maximize_function` function in weighted average of
        aggragted diffrecnces. Must be between 0 and 1, inclusive. The weight of
        `minimize_function` will be 1 - `maximize weight`.
    return_type: {'indices', 'values'}, default 'indices'
        If 'indices', returns the indices the optimal pair. If 'values', returns
         the values of the optimal pair.

    Returns
    -------
    If `return_type` is 'indices', returns the indices of the
    arrays out of input_array as tuple (e.g., input_array.iloc[optimal, :])
    If `return_type` is 'values' returns a 2D array of size (2, T), where the
    rows are the pair of optimal vectors.

    Notes
    -----
    Always returns a copy of the optimal arrays.

    Normalization of absolute differences is performed by scaling between 0 and
    1 [(x - min(x)) / (max(x) - min(x))].

    Examples
    --------
    Find an optimal pair of sub-arrays which have the most similar standard
    deviation (relative to all other sub-arrays), and the most different mean
    (relative to all other sub-arrays).

    >>> a = np.vstack([[0, 0.5], [0.5, 0.5], [5, 5], [4, 10]])
    >>> agas.pair_from_array(a, maximize_function=np.std,
        minimize_function=np.mean)
    (0, 2)

    Increase the weight of standard deviation in finding the optimal pair using
    the `maximize_weight` keyword argument.

    >>> agas.pair_from_array(a, maximize_function=np.std,
        minimize_function=np.mean, maximize_weight=.7)
    (1, 2)

    Return values instead of indices using the `return_type` keyword argument.

    >>> agas.pair_from_array(a, maximize_function=np.std,
        minimize_function=np.mean, return_type='values')
    array([[0. , 0.5],
       [5. , 5. ]])

    """

    if not isinstance(input_array, np.ndarray):
        try:
            input_array = np.array(input_array)
            if input_array.ndim != 2:
                raise ValueError("input_array must be 2-dimensional")
        except TypeError as e:
            raise RuntimeError("input_array must be a 2-dimensional array or "
                               "an object that can be converted to a 2-dimensional"
                               " array")

    if input_array.size == 0:
        raise RuntimeError("input_array must not be empty")

    if input_array.shape[0] == 1:
        raise RuntimeError(
            "input_array must have more than one element on the samples"
            f" axis. If trying to pass an input_array containing {input_array.shape[1]} series"
            "and 1 sample per series, (e.g., shape == [2, 1], consider transposing the"
            "input_array (input_array.T).")

    if not ((0 <= maximize_weight) & (maximize_weight <= 1)):
        raise ValueError("maximize_weight must be between 0 and 1 (0≤x<1)")

    if not return_type in RETURN_TYPE_OPTIONS:
        if isinstance(return_type, str):
            raise ValueError(
                f"return_type must be one of {RETURN_TYPE_OPTIONS}")
        else:
            raise TypeError("return_type must be a string")

    if (not isinstance(maximize_function, typing.Callable)
    ) or not isinstance(minimize_function, typing.Callable):
        raise TypeError(
            "Both `maximize_function` and `minimize_function` must be callables,"
            f" but received {type(maximize_function)} and "
            f"{type(minimize_function)}, respectively")

    minimize_weight = 1 - maximize_weight

    input_array = input_array.copy()

    max_sim_diffs_mat = _calc_differences(input_array,
                                          maximize_function)
    min_sim_diffs_mat = _calc_differences(input_array,
                                          minimize_function)

    optimized_differences = _calculate_optimal_values(max_sim_diffs_mat,
                                                      min_sim_diffs_mat,
                                                      maximize_weight, minimize_weight
                                                      )

    optimal_pair = _find_optimal_pairs(optimized_differences)

    if return_type == RETURN_INDICES:
        return optimal_pair
    elif return_type == RETURN_VALUES:
        return input_array[optimal_pair, :].copy()


def _apply_func(array, func):
    """Apply the given function to the input array along the second dimension (1).

    Assumes that if the function does not receive the axis positional or keyword
    parameter 'axis' it is predfined (e.g., using partial).
    """

    sig = inspect.signature(func)
    args = [p.name for p in sig.parameters.values() if
            p.kind == p.POSITIONAL_OR_KEYWORD]
    if 'axis' in args:
        return func(array, axis=1)
    else:
        # It is very likely that the function will be called along first axis
        # therefore we need to transpose the array prior to applying the function
        return func(array.T)


def _calc_differences(array: npt.NDArray, func: typing.Callable):
    aggregated_array = _apply_func(array, func)
    if (np.any(np.isnan(aggregated_array))):
        warnings.warn(f"The result of aggregating the input values using the "
                      f"function {func.__name__} resulted in "
                      f"{np.isnan(aggregated_array).sum()} NaN values.",
                      RuntimeWarning)
    if (np.all(np.isnan(aggregated_array))):
        raise ValueError(
            "Aggregating the input values using the {func.__name__}"
            " function resulted in all NaN values.")

    return _get_diffs_matrix(aggregated_array)

def _get_diffs_matrix(array: npt.NDArray):
    """
    Return a matrix of the absolute difference between each element and all
    other elements in the input_array.

    :param array: npt.NDArray of size 1 X n.
    :return: npt.NDArray of size n X n.
    """

    # After taking the absolute differences, cast the differences matrix as
    # float, given we need nans on the diagonal
    diffs_mat = np.abs(np.subtract.outer(array, array), dtype=float)
    # We can ignore the difference between each sample and itself
    np.fill_diagonal(diffs_mat, np.nan)
    return diffs_mat


def _normalize(a: npt.NDArray):
    """Normalize an input_array to the range between 0 and 1.

    :param a : npt.NDArray
        A 1D input_array to normalize.

    :return:
        The normalized input_array, between 0 and 1.
    """
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))


def _calculate_optimal_values(maximize_similarity_array, minimize_similarity_array,
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
    similarity = _normalize(maximize_similarity_array) * maximize_weight
    dissimilarity = - 1 * _normalize(minimize_similarity_array) * minimize_weight
    return _normalize(similarity + dissimilarity) # np.nansum([similarity, dissimilarity], axis=[0, 1])


def _find_optimal_pairs(optimized_differences):

    # Remove repeated pairs from the differences' matrix, by asigning the lower
    #  triangle of the array to NaNs.
    optimized_differences[np.tril_indices(
        optimized_differences.shape[0], -1)] = np.nan

    not_nan_indices = ~np.isnan(optimized_differences)

    # Find all indices of values which are not NaNs, hence the first occurure of
    #  of a pair
    indices = np.argwhere(not_nan_indices)
    values = optimized_differences[not_nan_indices]

    # Sort the indices and values by the values, from most optimal (0) to least
    # (1)
    ordered_values = values.argsort()
    indices = indices[ordered_values]
    values = values[ordered_values]

    return indices[0]
