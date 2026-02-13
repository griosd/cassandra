import pytest
import numpy as np
import pandas as pd

from ..pymc_toolkit.pymc_model import PymcModel
from ..pymc_toolkit.fleet_result import FleetResult

from ..pymc_toolkit.utils import simulate_spend_channels

@pytest.fixture
def sample_data():
    np.random.seed(12345)  
    data = pd.DataFrame({
        "ds": pd.date_range(start="2023-01-01", periods=20),
        "channel_1": np.random.rand(20),
        "control_1": np.random.rand(20),
        "y": np.random.rand(20),
    })
    return data

@pytest.mark.integration
def test_recovery_integration():
    seed = sum(map(ord, "PyMC-Marketing is weird"))
    n_dates = 52
    date_array = pd.date_range("2022-01-01", periods=n_dates, freq="W-MON")

    df_spends = simulate_spend_channels(
        seed=seed,
        dates=date_array,
        channel_names=["Facebook", "Searches"]
    )

    # Adding control data
    df_spends["trend"] = np.linspace(start=0.0, stop=n_dates, num=n_dates)
    df_spends["target"] = 1

    cfg = PymcModel(
        client_data=df_spends,
        channel_names=["Facebook", "Searches"],
        date_column="ds",
        control_names=["trend"],
        target_name="target"
    )

    fleet = cfg.recovery_fleet(chains=2,
                           draws=500,
                           tune=500,
                           seed=seed)

    result = fleet.get_recovery_diagnostics(use_recovery=True,hdi_prob=0.9)

    assert isinstance(result, dict)
    assert "saturation_beta" in result

    recovery = result["saturation_beta"]
    assert isinstance(recovery, float)
    assert 0 <= recovery <= 1

@pytest.mark.integration
def test_standard_fleet_integration(sample_data):  
    model = PymcModel(client_data=sample_data,
                    channel_names=["channel_1"],
                    control_names=["control_1"],
                    date_column="ds",
                    target_name="y")
    
    X = model.get_covariates()
    y = model.get_target(original_scale=True)

    X_train = X.iloc[:16]
    y_train = y[:16]
    X_test = X.iloc[16:]
    y_test = y[16:]

    fleet_result = model.standard_fleet(X_train=X_train,
                                 y_train=y_train,
                                 X_test=X_test,
                                 y_test=y_test,
                                 draws=100,
                                 tune=50,
                                 chains=2,
                                 cores=1,
                                 progressbar=False,
                                 seed=42)

    assert isinstance(fleet_result, FleetResult)
    assert fleet_result.X_train.equals(X_train)
    assert len(fleet_result.y_train) == 16
    assert fleet_result.y_pred.shape[1] == len(y_test)
    assert fleet_result.X_test.equals(X_test)
    assert len(fleet_result.y_test) == 4

@pytest.mark.integration
def test_production_fleet_integraiton(sample_data):
    model = PymcModel(client_data=sample_data,
                    channel_names=["channel_1"],
                    control_names=["control_1"],
                    date_column="ds",
                    target_name="y")

    fleet_result = model.production_fleet(n_test=4,
                                    draws=50, 
                                    tune=25, 
                                    chains=1, 
                                    cores=1, 
                                    progressbar=False, 
                                    seed=123)

    assert isinstance(fleet_result, FleetResult)
    assert fleet_result.y_pred.shape[1] == 4
