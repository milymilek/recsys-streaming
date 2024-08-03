import torch
import numpy as np
import pandas as pd
import pytest

#from recsys_streaming_ml.model.utils import *

@pytest.mark.xfail(reason="Long mongo init, to be refactored")
def test_build_input_tensor():
    a = np.array([1,2,3])
    result = build_input_tensor(a)
    assert isinstance(result, torch.tensor)
    assert result.dtype == torch.long


@pytest.mark.xfail(reason="Long mongo init, to be refactored")
def test_binarize_target():
    t = torch.tensor([0.1, 0.2, 0.5, 0.7, 1.0])
    threshold = 0.5
    result = binarize_target(t, threshold=threshold)
    assert result == torch.tensor([0, 0, 1, 1, 1])


@pytest.mark.xfail(reason="Long mongo init, to be refactored")
def test_cast_df_to_tensor():
    df = pd.DataFrame([[1,2], [3,4], [5,6]], columns=["col1", "col2"])
    df_vals = df.col1.values, df.col2.values
    result = cast_df_to_tensor(df_vals)
    for r, v in zip(result, df_vals):
        assert isinstance(r, torch.tensor)
        assert r.tolist() == v.tolist()
