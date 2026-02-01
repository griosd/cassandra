import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from cassandra.pymc_toolkit import rolling_split
from cassandra.pymc_toolkit.fleet_result import FleetResult

@pytest.fixture
def dummy_mmm():
    """Un objeto ficticio para ocupar el lugar de mmm."""
    mmm = MagicMock(name="DummyMMM")
    mmm.time_varying_intercept = False
    return mmm

@pytest.fixture
def simulated_data():
    n_samples = 20
    n_features = 3
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = np.random.rand(n_samples)
    return X, y

def test_rolling_split_basic(simulated_data):
    X, y = simulated_data
    splits = rolling_split(X, y, n_test=4, n_splits=1)
    assert len(splits) == 1
    X_train, y_train, X_test, y_test = splits[0]
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4

def test_rolling_split_multiple_splits(simulated_data):
    X, y = simulated_data
    splits = rolling_split(X, y, n_test=4, n_splits=3)
    assert len(splits) == 3
    for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
        assert len(X_test) == 4
        assert len(X_train) == 8

@pytest.mark.parametrize("n_test", [0, 5])
def test_rolling_split_n_test_zero(simulated_data, n_test):
    X, y = simulated_data
    splits = rolling_split(X, y, n_test=n_test)
    if n_test == 0:
        assert len(splits) == 1
        X_train, y_train, X_test, y_test = splits[0]
        assert X_test is None
        assert y_test is None
    else:
        assert splits[0][2].shape[0] == n_test

def test_rolling_split_invalid_n_test(simulated_data):
    X, y = simulated_data
    with pytest.raises(ValueError):
        rolling_split(X, y, n_test=-1)
    with pytest.raises(ValueError):
        rolling_split(X, y, n_test=100, n_splits=2)


def test_fleet_result_train_only(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    y_train = np.random.rand(10)
    y_pred = np.random.rand(100, 10)  # 100 draws, 10 observaciones

    fr = FleetResult(
        mmm=dummy_mmm,
        X_train=X_train,
        y_train=y_train,
        y_pred=y_pred
    )

    assert fr.mmm is dummy_mmm
    assert fr.X_train.equals(X_train)
    assert np.array_equal(fr.y_train, y_train)
    assert np.array_equal(fr.y_pred, y_pred)
    assert fr.X_test is None
    assert fr.y_test is None


def test_fleet_result_with_test_set(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(8, 3), columns=["a", "b", "c"])
    y_train = np.random.rand(8)
    X_test = pd.DataFrame(np.random.rand(4, 3), columns=["a", "b", "c"])
    y_test = np.random.rand(4)
    y_pred = np.random.rand(50, 4)  # 50 draws, 4 observaciones

    fr = FleetResult(
        mmm=dummy_mmm,
        X_train=X_train,
        y_train=y_train,
        y_pred=y_pred,
        X_test=X_test,
        y_test=y_test
    )

    assert fr.X_test.equals(X_test)
    assert np.array_equal(fr.y_test, y_test)

def test_invalid_y_pred_shape(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
    y_train = np.random.rand(5)
    y_pred = np.random.rand(5)  # 1D array, debería ser 2D

    with pytest.raises(ValueError, match="y_pred must have shape \\(n_draws, n_obs\\)"):
        FleetResult(dummy_mmm, X_train, y_train, y_pred)


def test_train_length_mismatch(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
    y_train = np.random.rand(5)
    y_pred = np.random.rand(10, 4)  # mismatch: 4 != len(y_train)

    with pytest.raises(
        ValueError, 
        match="y_train length \\(5\\) does not match y_pred second dimension \\(4\\)\\."
    ):
        FleetResult(dummy_mmm, X_train, y_train, y_pred)


def test_test_without_y_test(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
    y_train = np.random.rand(5)
    X_test = pd.DataFrame(np.random.rand(3, 2), columns=["a", "b"])
    y_pred = np.random.rand(10, 3)

    with pytest.raises(
        ValueError, match="When providing X_test, you must also provide y_test\\."
    ):
        FleetResult(dummy_mmm, X_train, y_train, y_pred, X_test=X_test)


def test_test_length_mismatch(dummy_mmm):
    X_train = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
    y_train = np.random.rand(5)
    X_test = pd.DataFrame(np.random.rand(3, 2), columns=["a", "b"])
    y_test = np.random.rand(3)
    y_pred = np.random.rand(10, 4)  # mismatch: 4 != len(y_test)

    with pytest.raises(
        ValueError, 
        match="y_test length \\(3\\) does not match y_pred second dimension \\(4\\)\\."
    ):
        FleetResult(dummy_mmm, X_train, y_train, y_pred, X_test=X_test, y_test=y_test)