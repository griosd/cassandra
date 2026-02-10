import pytest
import pandas as pd
import numpy as np

from pymc_marketing.mmm import MMM
from ..pymc_toolkit.client_config import ClientConfig
from ..pymc_toolkit.pymc_model import PymcModel, SaturationType, AdstockType

@pytest.fixture
def sample_data():
    np.random.seed(12345)  
    data = pd.DataFrame({
        "ds": pd.date_range(start="2023-01-01", periods=10),
        "channel_1": np.random.rand(10),
        "control_1": np.random.rand(10),
        "y": np.random.rand(10),
    })
    return data

def test_init_basic(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=["control_1"],
        target_name="y",
        date_column="ds",
    )

    assert isinstance(model.client_configuration, ClientConfig)
    assert model.client_name is None or isinstance(model.client_name, (str, type(None)))
    assert model.saturation_type == SaturationType.HILL
    assert model.adstock_type == AdstockType.GEOMETRIC
    assert "saturation_slope" in model.model_priors
    assert "gamma_control" in model.model_priors
    assert "intercept" in model.model_priors

def test_set_saturation_invalid(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=None,
        target_name="y",
    )
    with pytest.raises(ValueError):
        model._set_saturation("invalid_saturation")

def test_set_adstock_invalid(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=None,
        target_name="y",
    )
    with pytest.raises(ValueError):
        model._set_adstock("invalid_adstock")

def test_fit_and_predict_flow(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=["control_1"],
        target_name="y",
        date_column="ds")

    model.fit(draws=10, tune=5, chains=1, cores=1, seed=42)
    y_pred = model.predict()
    
    assert isinstance(model.model_fit, MMM)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape[1] == 10

def test_predict_before_fit_raises(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=["control_1"],
        target_name="y",
        date_column="ds",
    )
    with pytest.raises(RuntimeError):
        model.predict()

def test_validate_prediction_input_missing_cols(sample_data):
    model = PymcModel(
        client_data=sample_data,
        channel_names=["channel_1"],
        control_names=["control_1"],
        target_name="y",
        date_column="ds",
    )
    incomplete_df = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=10),
        "channel_1": np.random.rand(10),
    })
    with pytest.raises(ValueError):
        model._validate_prediction_input(incomplete_df)
    
def test_build_mmm():
    df = pd.DataFrame({
            "ds": pd.date_range("2022-01-01", periods=10),
            "Facebook": [1]*10,
            "Searches": [2]*10,
            "control1": [3]*10,
            "y": [10]*10
        })

    model = PymcModel(
        client_data=df,
        channel_names=["Facebook", "Searches"],
        control_names=["control1"],
        target_name="y",
        date_column="ds",
        lag_max=7
    )

    mmm_instance = model.build_pymc_mmm()

    assert isinstance(mmm_instance, MMM)
    priors = mmm_instance.model_config
    assert "intercept" in priors
    assert "gamma_control" in priors
    assert "adstock_alpha" in priors
    assert "saturation_beta" in priors
    assert "saturation_slope" in priors
    assert "saturation_kappa" in priors

    assert mmm_instance.adstock.l_max == 7
    assert mmm_instance.date_column == "ds"
    assert set(mmm_instance.channel_columns) == {"Facebook", "Searches"}
    assert set(mmm_instance.control_columns) == {"control1"}

def test_create_mmm_no_controls():
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=10),
        "Facebook": [1]*10,
        "Searches": [2]*10,
        "y": [10]*10
    })

    model = PymcModel(
        client_data=df,
        channel_names=["Facebook", "Searches"],
        target_name="y",
        date_column="ds",
        lag_max=7
    )

    mmm_instance = model.build_pymc_mmm()

    assert isinstance(mmm_instance, MMM)
    priors = mmm_instance.model_config
    assert "gamma_control" in priors
    assert mmm_instance.control_columns is None

def test_create_custom_saturation_mmm():

    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=10),
        "Facebook": [1]*10,
        "Searches": [2]*10,
        "control1": [3]*10,
        "y": [10]*10
    })

    model= PymcModel(
        client_data=df,
        channel_names=["Facebook", "Searches"],
        control_names=["control1"],
        target_name="y",
        date_column="ds",
        lag_max=7, 
        saturation = "logistic"
    )

    mmm_instance = model.build_pymc_mmm()

    assert isinstance(mmm_instance, MMM)
    priors = mmm_instance.model_config
    assert "saturation_beta" in priors
    assert "saturation_lam" in priors