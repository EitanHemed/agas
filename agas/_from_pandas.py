import typing

import numpy as np
import pandas as pd

from . import _from_numpy

__all__ = ['pair_from_wide_df']


def pair_from_wide_df(df: pd.DataFrame,
                      maximize_function: typing.Callable,
                      minimize_function: typing.Callable,
                      maximize_weight: typing.Union[float, int] = 0.5,
                      return_type: str = 'values',
                      values_columns: typing.Union[
                          typing.Tuple, typing.List, np.ndarray] = None
                      ):
    """
    Find the optimal pair of 1D arrays given rows in a wide-format dataframe.
    
    The optimality of the pairing of two vectors is determined by a weighted
    average of their absolute differences, following normalization of the
    differences between an aggregated vectors and each of the other vectors
    (excluding itself).

    Aggregation is performed using each of the aggregation functions
    `maximize_function` and `minimize_function`, weighted by `maximize_function`
    and 1 - `maximize_function`, respectively.

    Parameters
    ----------
    df: pd.DataFrame
        A wide (unstacked, pivoted) dataframe, where values are stored in
        columns and unique units are stored in rows.
    maximize_function, minimize_function: Callable
        Used to aggregate `input_array` along the 2-nd axis and find similar
        pairs - those with minimal absolute differences between their
        aggregated values. `minimize_function` is identical
        except it is used to find pairs of arrays with maximal absolute
        differnce between aggregated values.
    maximize_weight: Int, Float, default=0.5
        Used to weight the `maximize_function` function in weighted average of
        aggragted diffrecnces. Must be between 0 and 1, inclusive. The weight of
        `minimize_function` will be 1 - `maximize weight`.
    return_type: {'indices', 'values'}, default 'values'
        If 'indices', returns the indices of the paired rows out of
        'df'. If 'values', returns the paired rows.
    values_columns: array-like, Default None
        List, Tuple or Array of the column names of the values to aggregate. If
        None, assumes all columns should be aggregated.

    Returns
    -------
    If `return_type` is 'indices', returns the indices of the
    optimal pair of rows out of `df` (e.g., df.iloc[optimal, :]).
    If `return_type` is 'values' returns a dataframe composed of the optima pair
    of rows out of `df`.

    See Also
    --------
    :func:`~agas.pair_from_array`.

    Notes
    -----
    Currently Agas doesn't allow usage of string function names for aggregation,
     unlike what can be done using pandas.

    Always returns a copy of the optimal rows.

    Normalization of absolute differences is performed by scaling between 0 and
    1 [(x - min(x)) / (max(x) - min(x))].

    Examples
    -----
    Setting up a small dataset of angle readings from fictitious sensors,
    collected in 3-hour intervals.

    >>> data = np.array([(0, 2, 1), (10, 11, 100), (120, 150, 179)])
    >>> df = pd.DataFrame(data, columns=['3PM', '6PM', '9PM'],
    ...             index=['Yaw', 'Pitch', 'Roll'])
    >>> df.agg([np.std, 'sum'], axis=1).round(2)
             std    sum
    Yaw     1.00    3.0
    Pitch  51.68  121.0
    Roll   29.50  449.0

    Yaw and Roll display the highest normalized similarity in mean value,
    and the lowest normalized similarity in sum value.

    >>> agas.pair_from_wide_df(df, np.std, np.sum)
          3PM  6PM  9PM
    Yaw     0    2    1
    Roll  120  150  179

    Giving standard deviation a heavier weight, leads to Pitch and Roll
    selected as the optimal value.

    >>> agas.pair_from_wide_df(df, np.std, np.sum, 0.8)
           3PM  6PM  9PM
    Pitch   10   11  100
    Roll   120  150  179

    Prioritizing small differences between sums, and large differences in
    variance

    >>> agas.pair_from_wide_df(df, np.sum, np.std)
           3PM  6PM  9PM
    Yaw      0    2    1
    Pitch   10   11  100
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'df must be a pandas DataFrame, received {type(df)}')
    else:
        if df.shape[0] < 2:  # Less than 2 rows
            raise ValueError(f'df must contain at least two rows')

    if values_columns is not None:
        _df = df.loc[:, values_columns]
    else:
        _df = df.copy()

    res = _from_numpy.pair_from_array(
        _df.values, maximize_function, minimize_function,
        maximize_weight, return_type=_from_numpy.RETURN_INDICES)

    if return_type is _from_numpy.RETURN_INDICES:
        return list(res)
    if return_type is _from_numpy.RETURN_VALUES:
        return df.iloc[list(res), :].copy()
