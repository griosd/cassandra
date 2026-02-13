import pytest
import numpy as np
import pandas as pd
import xarray as xr

from unittest.mock import MagicMock
from ..pymc_toolkit.fleet_result import FleetResult

@pytest.fixture
def mock_mmm():
    """Return a mock MMM object with minimal structure for FleetResult."""
    mmm = MagicMock()
    div = xr.DataArray(
        data = np.array([[0,0],[0,0],[1,1],[1,1]]),  
        coords=[[0, 1, 2, 3], [0, 1]],  
        dims=["chain", "draw"])
    tree_depth =  xr.DataArray(
        data = np.array([[5,6],[7,8],[9,10],[5,5]]),  
        coords=[[0, 1, 2, 3], [0, 1]],  
        dims=["chain", "draw"])

    mmm.idata = {
        "sample_stats": {
            "diverging": div,
            "tree_depth": tree_depth
        }   
    }
    mmm.time_varying_intercept = False
    mmm.model_config = {
       'intercept':1, 
       'likelihood':1, 
       'gamma_control':1, 
       'gamma_fourier':1, 
       'intercept_tvp_config':1, 
       "media_tvp_config":1,
    }
    mmm.sample_posterior_predictive.return_value = {"y": np.array([[11,21], [31,12], [22,32]])}
    mmm.control_columns = None
    return mmm

@pytest.fixture
def base_data():
    """Return common X_train, y_train, y_pred for tests."""
    X_train = pd.DataFrame({"channel1": [1, 2, 3]})
    y_train = np.array([10, 20, 30])
    y_pred = np.array([[11, 21, 31], [12, 22, 32]])  
    return X_train, y_train, y_pred


def test_get_model_accuracy_train(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred)
    metrics = fr.get_model_accuracy(train=True)
    assert isinstance(metrics, dict)
    assert "mape" in metrics
    assert "crps_error" in metrics


def test_get_model_accuracy_test(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    X_test = pd.DataFrame({"channel1": [4, 5, 6]})
    y_test = np.array([40, 50, 60])
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred, X_test=X_test, y_test=y_test)
    # y_fit will be computed via _get_y_fit
    metrics = fr.get_model_accuracy(train=False)
    assert isinstance(metrics, dict)
    assert "nrmse" in metrics


def test_get_mcmc_diagnostics(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred)
    diag = fr.get_mcmc_diagnostics()
    assert "divergences" in diag
    assert "Max_treedepth" in diag
    assert isinstance(diag["divergences"], list)

def test_get_recovery_diagnostics_wrong_type(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred)
    with pytest.raises(ValueError):
        fr.get_recovery_diagnostics()

def test_summary_train(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred)
    summary = fr.summary()
    assert "crps_error" in summary['train']
    assert "mape" in summary['train']

def test_summary_production(mock_mmm, base_data):
    X_train, y_train, y_pred = base_data
    X_test = pd.DataFrame({"channel1": [4, 5, 6]})
    y_test = np.array([40, 50, 60])
    fr = FleetResult(mock_mmm, X_train, y_train, y_pred, X_test=X_test, y_test=y_test)
    summary = fr.summary()
    assert "crps_error" in summary['test']
    assert "mape" in summary['test']