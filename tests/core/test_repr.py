def test_repr(edata_333):
    edata = edata_333

    # vanilla case
    repr_str = edata.__repr__()
    target_repr_str = "EHRData object with n_obs × n_vars × n_t = 3 × 3 × 3\n    obs: 'obs_col_1'\n    var: 'var_col_1'\n    tem: 't1', 't2', 't3'\n    shape of .X: (3, 3)\n    shape of .R: (3, 3, 3)"
    assert repr_str == target_repr_str

    # more layers
    edata.layers["test_layer"] = edata.X.copy()
    repr_str_with_layer = edata.__repr__()
    target_repr_str_with_layer = "EHRData object with n_obs × n_vars × n_t = 3 × 3 × 3\n    obs: 'obs_col_1'\n    var: 'var_col_1'\n    tem: 't1', 't2', 't3'\n    layers: 'test_layer'\n    shape of .X: (3, 3)\n    shape of .R: (3, 3, 3)"
    assert repr_str_with_layer == target_repr_str_with_layer

    # view
    edata_view = edata[:2, :2, :2]
    repr_str_view = edata_view.__repr__()
    target_repr_str_view = "View of EHRData object with n_obs × n_vars × n_t = 2 × 2 × 2\n    obs: 'obs_col_1'\n    var: 'var_col_1'\n    tem: 't1', 't2'\n    layers: 'test_layer'\n    shape of .X: (2, 2)\n    shape of .R: (2, 2, 2)"
    assert repr_str_view == target_repr_str_view
