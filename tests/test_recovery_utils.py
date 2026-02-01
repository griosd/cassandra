import pytest
import numpy as np
import arviz as az
import pandas as pd
from unittest.mock import MagicMock
 
from cassandra.pymc_toolkit import (crps_accuracy,
                                    compute_coverage,
                                    coverage_percentage,
                                    recovery_percentage,
                                    simulate_spend_channels)

def get_mock_mmm():

    mock = MagicMock()

    # Simular parámetros con 4 cadenas y 250 draws
    chains, draws, dims = 4, 250, 3
    y_len = 10

    posterior_data = {
        "saturation_beta": np.random.normal(1.0, 0.1, size=(chains, draws, dims)),
        "y_sigma": np.random.normal(1.0, 0.1, size=(chains, draws)),
        "intercept": np.random.normal(0.5, 0.1, size=(chains, draws)),
        "y": np.random.normal(100, 5, size=(y_len)),  # para y_fit_train
    }

    mock.idata = az.from_dict(posterior=posterior_data)
    mock.y = np.linspace(90, 110, y_len)
    mock.X = np.random.rand(y_len, 3)

    mock.sample_posterior_predictive.return_value = {
        "y": np.random.normal(100, 5, size=(chains * draws, y_len))  # (draws, y_len)
    }

    # Escaladores falsos
    class DummyScaler:
        def inverse_transform(self, x):
            return x * 10

    mock.channel_transformer = DummyScaler()
    mock.target_transformer = DummyScaler()
    mock.control_columns = None

    return mock
 
def test_simulate_spend_channels_output():
    ds = pd.date_range(start="2020-01-01", periods=10, freq="D")
    df = simulate_spend_channels(dates=ds, 
                            channel_names=["tv", "facebook", "search"], 
                            seed=42)
    assert df.shape == (10, 4)
    assert all(col in df.columns for col in ["tv", "facebook", "search","ds"])

def test_compute_coverage():
    posterior = np.random.normal(5, 0.2, size=(1000, 5))
    true_values = np.full(5, 5.0)
    coverage = compute_coverage(posterior, true_values, hdi_prob=0.9)
    assert isinstance(coverage, float)
    assert 0 <= coverage <= 1

def test_crps_accuracy():
    target = np.array([100, 110, 105])
    predict = np.random.normal(loc=100, scale=5, size=(1000, 3))
    score = crps_accuracy(target, predict)
    assert isinstance(score, float)
    assert 0 <= score <= 100

def test_recovery_percentage():
    mmm = get_mock_mmm()
    true_params = {"saturation_beta": [1.0, 1.0, 1.0]}
    score = recovery_percentage(mmm, "saturation_beta", true_params, original_scale=False, averaged=True)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_coverage_percentage():
    mmm = get_mock_mmm()
    true_params = {"saturation_beta": [1.0, 1.0, 1.0]}
    coverage = coverage_percentage(mmm, "saturation_beta", true_params, original_scale=False, hdi_prob=0.9)
    assert isinstance(coverage, float)
    assert 0 <= coverage <= 1
