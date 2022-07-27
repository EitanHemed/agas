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


def pair_from_array(input_array: npt.NDArray,
                    maximize_function: typing.Callable,
                    minimize_function: typing.Callable,
                    maximize_weight: typing.Union[float, int] = 0.5,
                    return_type: str = 'indices'):
    f"""
    :param input_array:
        A 2D array of shape (n_series, n_times).
    :param maximize_function:
        Select the pair based on the maximal similarity (minimal absolute
        difference) between the output values of this `maximize_function`.
    :param minimize_function:
        Select the pair based on the minimal similarity (maximal absolute
        difference) between the output values of `minimize_function`.
    :param maximize_weight:
        The weight of the `maximize_function` function in the pairing of data
        series. Must be between 0 and 1, inclusive. The weight of `minimize_function`
        will be the complementy of `maximize weight`.
    :param return_type: {RETURN_TYPE_OPTIONS}, default '{RETURN_INDICES}'
        If '{RETURN_VALUES}', returns the indices of the paired data series out of
         the 'input_array'. If '{RETURN_VALUES}', returns the values of the paired
          data series. 
    :return:
        If `return_type` is '{RETURN_INDICES}', returns the indices of the 
        paired data series out of input_array as tuple. If `return_type` is 
        '{RETURN_VALUES}' returns the values of the paired data series as 
        2D array.
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

    normalized_max_sim_diffs_mat = _normalize_differences(input_array,
                                                          maximize_function)
    normalized_min_sim_diffs_mat = _normalize_differences(input_array,
                                                          minimize_function)

    optimized_differences = _optimize(normalized_max_sim_diffs_mat,
                                      normalized_min_sim_diffs_mat,
                                      maximize_weight, minimize_weight
                                      )

    optimal_pair = _find_optimal(optimized_differences)

    if return_type == RETURN_INDICES:
        return optimal_pair
    elif return_type == RETURN_VALUES:
        return input_array[optimal_pair, :]


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


def _normalize_differences(array: npt.NDArray, func: typing.Callable):
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

    diffs_mat = _get_diffs_matrix(aggregated_array)
    normalized_diffs_mat = _normalize(diffs_mat)
    return normalized_diffs_mat


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
