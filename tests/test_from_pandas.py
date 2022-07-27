import numpy as np
import pytest

import agas
from agas._from_numpy import (RETURN_TYPE_OPTIONS, RETURN_INDICES,
                              RETURN_VALUES)

def test_pair_from_df():
    with pytest.raises(NotImplementedError):
        agas.pair_from_df()

