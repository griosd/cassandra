import pytest
import logging
import numpy as np
import pandas as pd
from ..pymc_toolkit.client_config import ClientConfig

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=5, freq="D"),
        "tv": [100, 120, 130, 140, 110],
        "radio": [30, 40, 35, 45, 38],
        "temp": [15, 14, 16, 15, 13],
        "y": [200, 210, 215, 220, 205],
    })

def test_ClientConfig_with_data(dummy_df, caplog):
    with caplog.at_level(logging.INFO):
        cfg = ClientConfig(
            channel_names=["tv", "radio"],
            client_data=dummy_df,
            date_column="ds",
            client_name="test_client",
            control_names=["temp"]
        )
    assert cfg.client_name == "test_client"
    assert cfg.target_name == "y"
    assert cfg.coords["n_channels"] == 2
    assert cfg.coords["n_controls"] == 1
    assert "Client data loaded for 'test_client'" in caplog.text

def test_init_missing_date_column():
    df = pd.DataFrame({'a': [1,2], 'tv':[1,2], 'radio':[3,4], 'y':[5,6]})
    with pytest.raises(ValueError):
        ClientConfig(channel_names=['tv', 'radio'], client_data=df)

def test_to_dict_contains_expected_keys(dummy_df):
    cfg = ClientConfig(
        channel_names=['tv'],
        control_names=['temp'],
        client_data=dummy_df,
        target_name='y',
        client_name='clientA'
    )
    d = cfg.to_dict()
    assert d['client_name'] == 'clientA'
    assert d['target'] == 'y'
    assert d['n_timesteps'] == len(dummy_df)
    assert d['n_channels'] == 1
    assert d['n_controls'] == 1

def test_repr_contains_client_name_and_counts(dummy_df):
    cfg = ClientConfig(
        channel_names=['tv', 'radio'],
        control_names=[],
        client_data=dummy_df,
        client_name='clientX'
    )
    rep = repr(cfg)
    assert 'clientX' in rep

def test_add_new_data_success(caplog):
    data = {
        'ds': pd.date_range('2023-01-01', periods=3),
        'channel1': [1, 2, 3],
        'control1': [0, 0, 1],
        'y': [10, 20, 30]
    }
    df_new = pd.DataFrame(data)
    original_df = pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=3),
        'channel1': [0,1,0],
        'control1': [0,1,0],
        'y': [3,1,2]
    })
    cfg = ClientConfig(
        channel_names=['channel1'], 
        control_names=['control1'], 
        client_data=original_df,
        target_name='y',
        date_column='ds'
    )
    with caplog.at_level(logging.INFO):
        cfg.add_new_data(df_new)
    assert cfg.coords['n_timesteps'] == 6
    assert "New client data updated successfully." in caplog.text

def test_missing_columns_in_add_client_data(dummy_df):
    cfg = ClientConfig(
        channel_names=['tv', 'radio'],
        control_names=['temp'],
        client_data=dummy_df
    )
    bad_df = dummy_df.drop(columns=['tv'])
    with pytest.raises(ValueError):
        cfg.add_new_data(bad_df)

@pytest.mark.parametrize("missing_col", ["ds", "tv", "y"])
def test_init_with_missing_columns_raises(missing_col, dummy_df):
    df = dummy_df.drop(columns=[missing_col])
    with pytest.raises(ValueError):
        ClientConfig(channel_names=['tv'], control_names=['temp'], client_data=df)

def test_get_covariates_with_controls():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "y": [1.0, 2.0, 3.0],
        "tv": [10, 20, 30],
        "radio": [5, 10, 15],
        "inflation": [2.1, 2.2, 2.3]
    })

    client = ClientConfig(
        client_data=data,
        date_column="date",
        target_name="y",
        channel_names=["tv", "radio"],
        control_names=["inflation"]
    )

    covariates_df = client._get_covariates()

    assert list(covariates_df.columns) == ["date", "tv", "radio", "inflation"]
    assert covariates_df.shape == (3, 4)
    assert covariates_df["inflation"].iloc[0] == 2.1

def test_get_covariates_without_controls():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame({
        "ds": dates,
        "y": [1.0, 2.0, 3.0],
        "tv": [10, 20, 30],
    })

    client = ClientConfig(
        client_data=data,
        target_name="y",
        channel_names=["tv"]
    )

    covariates_df = client._get_covariates()

    assert list(covariates_df.columns) == ["ds", "tv"]
    assert covariates_df.shape == (3, 2)

def test_get_covariates_with_and_without_inverse_scaling():
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=5, freq="D"),
        "tv": [100, 150, 200, 250, 300],
        "radio": [20, 25, 30, 35, 40],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],  # control
        "holiday": [0, 1, 0, 1, 0],     # control
        "y": [12,15,17,40,60] #target
    })

    config = ClientConfig(
        client_data=df,
        channel_names=["tv", "radio"],
        control_names=["price", "holiday"],
        scale_data=True
    )

    cov_scaled = config._get_covariates()
    assert "price" in cov_scaled.columns
    assert "holiday" in cov_scaled.columns

    assert np.all(np.abs(cov_scaled["price"]) <= 1)
    assert np.all(np.abs(cov_scaled["holiday"]) <= 1)

    cov_unscaled = config._get_covariates(original_scale=True)
    pd.testing.assert_series_equal(cov_unscaled["price"], df["price"].astype(float))
    pd.testing.assert_series_equal(cov_unscaled["holiday"], df["holiday"].astype(int))

@pytest.fixture
def config_with_scaling(dummy_df):
    return ClientConfig(
        channel_names=["tv", "radio"],
        control_names=["temp"],
        client_data=dummy_df,
        target_name="y",
        scale_data=True,
        date_column="ds"
    )

@pytest.mark.parametrize("variable", ["channels", "controls"])
def test_get_scaler_and_inverse_scaler_returns_functions(config_with_scaling, variable):
    scaler_fn = config_with_scaling.get_scaler(variable)
    inverse_fn = config_with_scaling.get_inverse_scaler(variable)

    assert callable(scaler_fn)
    assert callable(inverse_fn)

    if variable == "channels":
        data = config_with_scaling._get_channels(original_scale=True)
    else:  # controls
        data = config_with_scaling._get_controls(original_scale=True)
    scaled = scaler_fn(data)
    recovered = inverse_fn(scaled)

    np.testing.assert_allclose(recovered, data, rtol=1e-5, atol=1e-8)

def test_get_scaler_invalid_variable_raises(config_with_scaling):
    with pytest.raises(ValueError):
        config_with_scaling.get_scaler("invalid_var")

def test_get_inverse_scaler_invalid_variable_raises(config_with_scaling):
    with pytest.raises(ValueError):
        config_with_scaling.get_inverse_scaler("invalid_var")

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "tv": [100, 110, 120, 130, 140],
        "radio": [10, 20, 15, 25, 30],
        "temp": [1.5, 1.7, 1.6, 1.8, 1.9],
        "holiday": [0, 1, 0, 1, 0],
        "sales": [200, 210, 220, 230, 240]
    })

@pytest.fixture
def client_config(sample_df):
    return ClientConfig(
        client_data=sample_df,
        date_column="date",
        target_name="sales",
        channel_names=["tv", "radio"],
        control_names=["temp", "holiday"],
        scale_data=False,
    )

def test_get_controls(client_config, sample_df):
    controls = client_config._get_controls()

    assert isinstance(controls, pd.DataFrame)
    assert list(controls.columns) == ["temp", "holiday"]
    pd.testing.assert_frame_equal(controls.reset_index(drop=True), sample_df[["temp", "holiday"]])

def test_get_channels(client_config, sample_df):
    channels = client_config._get_channels()
    assert isinstance(channels, pd.DataFrame)
    assert list(channels.columns) == ["tv", "radio"]
    pd.testing.assert_frame_equal(channels.reset_index(drop=True), sample_df[["tv", "radio"]])

def test_get_covariates(client_config, sample_df):
    covariates = client_config._get_covariates()
    expected_cols = ["date", "tv", "radio", "temp", "holiday"]
   
    assert list(covariates.columns) == expected_cols
    pd.testing.assert_series_equal(covariates["date"], sample_df["date"])
    pd.testing.assert_frame_equal(covariates[["tv", "radio"]], sample_df[["tv", "radio"]])
    pd.testing.assert_frame_equal(covariates[["temp", "holiday"]], sample_df[["temp", "holiday"]])

def test_get_target(client_config, sample_df):
    target = client_config._get_target()

    assert isinstance(target, np.ndarray)
    np.testing.assert_array_equal(target, sample_df["sales"].values)

def test_get_data(client_config, sample_df):
    target, covariates = client_config._get_data()
    expected_cols = ["date", "tv", "radio", "temp", "holiday"]
    
    assert isinstance(target, np.ndarray)
    np.testing.assert_array_equal(target, sample_df["sales"].values)
    assert isinstance(covariates, pd.DataFrame)
    assert list(covariates.columns) == expected_cols
    pd.testing.assert_series_equal(covariates["date"], sample_df["date"])