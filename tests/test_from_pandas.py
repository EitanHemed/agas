import typing

import numpy as np
import pandas as pd
import pytest

import agas
from . test_from_numpy import TOY_DATA

TOY_DATA_DF = pd.DataFrame(TOY_DATA,
                           columns=[f'Day {i}' for i in range(1, 11)]).assign(
    subject_ids=['Foo', 'Bar', 'Baz'])


def test_invalid_arguments():
    with pytest.raises(TypeError):
        agas.pair_from_wide_df(None, np.mean, np.std)

    with pytest.raises(ValueError):
        # Select no rows
        agas.pair_from_wide_df(
            TOY_DATA_DF.loc[TOY_DATA_DF['subject_ids'] == 'Foobaz'].filter(
                like='Day'), np.mean, np.std)
        # Select a single row
        agas.pair_from_wide_df(
            TOY_DATA_DF.loc[TOY_DATA_DF['subject_ids'] == 'Bar'].filter(
                like='Day'), np.mean, np.std)


@pytest.mark.parametrize("return_type", agas._from_numpy.RETURN_TYPE_OPTIONS)
@pytest.mark.parametrize("values_columns",
                         [None, TOY_DATA_DF.filter(like='Day').columns])
def test_return_type(return_type: str,
                     values_columns: typing.Union[None, typing.List]):
    if values_columns is None:
        inp_vals = TOY_DATA_DF.filter(like='Day')
    else:
        inp_vals = TOY_DATA_DF.copy()

    if return_type == agas._from_numpy.RETURN_VALUES:
        expected_return = inp_vals.iloc[[0, 1], :]
        pd.testing.assert_frame_equal(
            agas.pair_from_wide_df(inp_vals, np.std, np.mean,
                                   return_type=return_type,
                                   values_columns=values_columns),
            expected_return)

    if return_type == agas._from_numpy.RETURN_INDICES:
        expected_return = [0, 1]
        assert agas.pair_from_wide_df(inp_vals, np.std, np.mean,
                                      return_type=return_type,
                                      values_columns=values_columns) == expected_return
