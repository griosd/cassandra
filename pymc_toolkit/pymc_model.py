import logging
from enum import Enum
from typing import Optional, Dict, Union, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    WeibullPDFAdstock,
    HillSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
)

from pymc_toolkit.client_config import ClientConfig
from pymc_toolkit.fleet_result import FleetResult
from pymc_toolkit.utils import rolling_split, recovery_summary

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class SaturationType(Enum):
    LOGISTIC = "logistic"
    HILL = "hill"
    MICHAELIS_MENTEN = "michaelis_menten"


class AdstockType(Enum):
    GEOMETRIC = "geometric"
    WEIBULL_PDF = "weibull_pdf"


class EventBasis(Enum):
    """How to convert events to numeric features."""
    GAUSSIAN = "gaussian"
    HALF_GAUSSIAN = "half_gaussian"
    ASYMMETRIC_GAUSSIAN = "asymmetric_gaussian"


class HalfGaussianMode(Enum):
    """Which side keeps mass for half-gaussian basis."""
    AFTER = "after"
    BEFORE = "before"


# =============================================================================
# Event feature builders (pure numpy)
# =============================================================================

def _gaussian_bump(x: np.ndarray, sigma: float) -> np.ndarray:
    """exp(-0.5*(x/sigma)^2) — unnormalized gaussian bump."""
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * (x / sigma) ** 2)


def _half_gaussian_bump(x: np.ndarray, sigma: float, mode: HalfGaussianMode) -> np.ndarray:
    out = _gaussian_bump(x, sigma)
    if mode == HalfGaussianMode.AFTER:
        return np.where(x >= 0, out, 0.0)
    if mode == HalfGaussianMode.BEFORE:
        return np.where(x <= 0, out, 0.0)
    raise ValueError("mode must be HalfGaussianMode.AFTER or HalfGaussianMode.BEFORE")


def _asymmetric_gaussian_bump(x: np.ndarray, sigma_before: float, sigma_after: float) -> np.ndarray:
    sigma_before = max(float(sigma_before), 1e-6)
    sigma_after = max(float(sigma_after), 1e-6)
    out = np.empty_like(x, dtype=float)
    mask = x < 0
    out[mask] = np.exp(-0.5 * (x[mask] / sigma_before) ** 2)
    out[~mask] = np.exp(-0.5 * (x[~mask] / sigma_after) ** 2)
    return out


# =============================================================================
# Main class
# =============================================================================

class PymcModel:
    """
    Class to generate Media Mix Models (MMM) in PyMC with support for
    prior configuration, saturation functions, and adstock functions.

    Optional: add event features (pymc-marketing notebook "events") by injecting
    engineered event covariates as extra control columns.
    """

    def __init__(
        self,
        client_data: pd.DataFrame,
        channel_names: List[str],
        date_column: str = "ds",
        control_names: Optional[List[str]] = None,
        target_name: str = "y",
        client_name: Optional[str] = None,
        lag_max: int = 1,
        scale_data: bool = False,
        calibrations: dict = None,
        priors: Optional[Dict[str, Prior]] = None,
        saturation: Union[str, SaturationType] = SaturationType.HILL,
        adstock: Union[str, AdstockType] = AdstockType.GEOMETRIC,
        number_of_basis: int = 50,
        time_varying_media: bool = False,
        time_varying_intercept: bool = False,
        # --- EVENTS (optional) ---
        df_events: Optional[pd.DataFrame] = None,
        events_basis: Union[str, EventBasis] = EventBasis.GAUSSIAN,
        events_sigma_days: float = 7.0,
        events_half_mode: Union[str, HalfGaussianMode] = HalfGaussianMode.AFTER,
        events_sigma_before_days: float = 7.0,
        events_sigma_after_days: float = 14.0,
        events_prefix: str = "event",
        events_reference_date: Optional[str] = None,
    ):

        # ---------------------------------------------------------------------
        # Inject event features BEFORE building ClientConfig
        # ---------------------------------------------------------------------
        if df_events is not None and len(df_events) > 0:
            client_data, control_names = self._inject_event_features(
                client_data=client_data,
                control_names=control_names,
                df_events=df_events,
                date_column=date_column,
                basis=events_basis,
                sigma_days=events_sigma_days,
                half_mode=events_half_mode,
                sigma_before_days=events_sigma_before_days,
                sigma_after_days=events_sigma_after_days,
                prefix=events_prefix,
                reference_date=events_reference_date,
            )

        logger.info("Creating the client's data configuration.")
        self.client_configuration = ClientConfig(
            client_data=client_data,
            channel_names=channel_names,
            control_names=control_names,
            calibrations=calibrations,
            target_name=target_name,
            date_column=date_column,
            lag_max=lag_max,
            scale_data=scale_data,
            client_name=client_name,
        )

        # ---------------------------------------------------------------------
        # Base config
        # ---------------------------------------------------------------------
        logger.info("Set up PyMCModel's basic configuration.")
        self.number_of_basis = number_of_basis
        self.time_varying_media = time_varying_media
        self.time_varying_intercept = time_varying_intercept

        self.lag_max = self.client_configuration.lag_max
        self.date_name = self.client_configuration.date_name
        self.client_name = self.client_configuration.client_name
        self.channel_columns = self.client_configuration.channel_names
        self.control_columns = self.client_configuration.control_names
        self.lift_tests = self.client_configuration.calibration_inputs

        self.model_fit = None
        self.has_lift_tests = False if self.lift_tests is None else True

        # ---------------------------------------------------------------------
        # Default priors & variables
        # ---------------------------------------------------------------------
        logger.info("Define PymcModel's default priors.")
        self.model_variables = ["y_sigma", "intercept"]

        logger.info("Creating default priors for intercept and likelihood.")
        self.model_priors = {
            "intercept": Prior("HalfNormal", sigma=1),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        }

        if self.control_columns:
            logger.info("Creating default priors for gamma control.")
            self.model_variables.append("gamma_control")
            self.model_priors["gamma_control"] = Prior("HalfNormal", sigma=1, dims="control")

        self._set_adstock(adstock)
        self._set_saturation(saturation)
        self._set_time_varying_model()

        if priors:
            try:
                logger.info("Updating priors with user-defined values.")
                self.model_priors.update(priors)
            except Exception as e:
                logger.error(f"Error updating priors: {e}", exc_info=True)

    def __repr__(self):
        return (
            f"PyMC Media Mix Model(client_name='{self.client_configuration.client_name}', "
            f"Adstock={self.adstock_type}, "
            f"Saturation={self.saturation_type.value})"
        )

    # =============================================================================
    # Events
    # =============================================================================

    @staticmethod
    def _coerce_enum(value, enum_cls):
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value.lower())
            except Exception:
                pass
        raise ValueError(f"Invalid value '{value}' for {enum_cls.__name__}")

    def _inject_event_features(
        self,
        client_data: pd.DataFrame,
        control_names: Optional[List[str]],
        df_events: pd.DataFrame,
        date_column: str,
        basis: Union[str, EventBasis],
        sigma_days: float,
        half_mode: Union[str, HalfGaussianMode],
        sigma_before_days: float,
        sigma_after_days: float,
        prefix: str,
        reference_date: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert df_events into numeric regressors (control columns).

        Expected df_events columns: name, start_date, end_date

        Strategy (simple and robust):
        - Compute a time index in DAYS from a reference date.
        - For each event, use the midpoint of its [start_date, end_date] as center.
        - Create a bump-shaped covariate based on basis.
        """
        required = {"name", "start_date", "end_date"}
        missing = required.difference(df_events.columns)
        if missing:
            raise ValueError(f"df_events is missing columns {missing}. Required: {required}")

        basis = self._coerce_enum(basis, EventBasis)
        half_mode = self._coerce_enum(half_mode, HalfGaussianMode)

        data = client_data.copy()
        data[date_column] = pd.to_datetime(data[date_column])

        ev = df_events.copy()
        ev["start_date"] = pd.to_datetime(ev["start_date"])
        ev["end_date"] = pd.to_datetime(ev["end_date"])

        ref = pd.to_datetime(reference_date) if reference_date else data[date_column].min()
        t_days = (data[date_column] - ref).dt.days.values.astype(float)

        new_controls: List[str] = []

        for _, row in ev.iterrows():
            raw_name = str(row["name"]).strip()
            safe_name = raw_name.replace(" ", "_")

            start = row["start_date"]
            end = row["end_date"]

            # center of event window
            center = start + (end - start) / 2
            center_days = float((pd.to_datetime(center) - ref).days)

            x = t_days - center_days  # days from event center

            col = f"{prefix}_{safe_name}"

            if basis == EventBasis.GAUSSIAN:
                values = _gaussian_bump(x, sigma_days)
            elif basis == EventBasis.HALF_GAUSSIAN:
                values = _half_gaussian_bump(x, sigma_days, mode=half_mode)
            elif basis == EventBasis.ASYMMETRIC_GAUSSIAN:
                values = _asymmetric_gaussian_bump(x, sigma_before_days, sigma_after_days)
            else:
                raise ValueError("Unsupported EventBasis")

            data[col] = values
            new_controls.append(col)

        if control_names is None:
            control_names = []
        control_names = list(control_names) + new_controls

        logger.info(f"Injected {len(new_controls)} event feature(s) into controls: {new_controls}")
        return data, control_names

    # =============================================================================
    # Saturation / Adstock / TVP
    # =============================================================================

    def _set_saturation(self, saturation: Union[str, SaturationType]):
        if isinstance(saturation, str):
            try:
                saturation = SaturationType(saturation.lower())
            except ValueError:
                logger.error(f"Unsupported saturation type string received: '{saturation}'")
                raise ValueError(f"Unsupported saturation type string: '{saturation}'")

        self.saturation_type = saturation

        if saturation == SaturationType.LOGISTIC:
            logger.info("Using Logistic Saturation.")
            self.saturation = LogisticSaturation()

            self.model_variables += ["saturation_lam", "saturation_beta"]
            self.model_priors["saturation_lam"] = Prior("Gamma", alpha=3, beta=1, dims="channel")
            self.model_priors["saturation_beta"] = Prior("HalfNormal", sigma=2, dims="channel")

        elif saturation == SaturationType.MICHAELIS_MENTEN:
            logger.info("Using Michaelis-Menten Saturation.")
            self.saturation = MichaelisMentenSaturation()

            self.model_variables += ["saturation_lam", "saturation_alpha"]
            self.model_priors["saturation_lam"] = Prior("HalfNormal", sigma=1, dims="channel")
            self.model_priors["saturation_alpha"] = Prior("Gamma", mu=2, sigma=1, dims="channel")

        elif saturation == SaturationType.HILL:
            logger.info("Using Hill Saturation.")
            self.saturation = HillSaturation()

            self.model_variables += ["saturation_slope", "saturation_kappa", "saturation_beta"]
            self.model_priors["saturation_slope"] = Prior("Normal", mu=1, sigma=0.001, dims="channel")
            self.model_priors["saturation_kappa"] = Prior("HalfNormal", sigma=1.5, dims="channel")
            self.model_priors["saturation_beta"] = Prior("HalfNormal", sigma=1.5, dims="channel")

        else:
            raise ValueError(f"Unsupported SaturationType: {saturation}")

    def _set_adstock(self, adstock: Union[str, AdstockType]):
        if isinstance(adstock, str):
            try:
                adstock = AdstockType(adstock.lower())
            except ValueError:
                logger.error(f"Unsupported adstock type string received: '{adstock}'")
                raise ValueError(f"Unsupported adstock type string: '{adstock}'")

        self.adstock_type = adstock

        if adstock == AdstockType.GEOMETRIC:
            logger.info("Using Geometric Adstock.")
            self.adstock = GeometricAdstock(l_max=self.lag_max)

            self.model_variables.append("adstock_alpha")
            self.model_priors["adstock_alpha"] = Prior("Beta", alpha=1, beta=3, dims="channel")

        elif adstock == AdstockType.WEIBULL_PDF:
            logger.info("Using Weibull PDF Adstock.")
            self.adstock = WeibullPDFAdstock(l_max=self.lag_max)

            self.model_variables += ["adstock_lam", "adstock_k"]
            self.model_priors["adstock_k"] = Prior("Gamma", mu=3, sigma=1, dims="channel")
            self.model_priors["adstock_lam"] = Prior("Gamma", mu=2, sigma=1, dims="channel")

        else:
            raise ValueError(f"Unsupported AdstockType: {adstock}")

    def _set_time_varying_model(self):
        if self.time_varying_media:
            logger.info("Time-varying Media MMM.")
            self.model_priors["media_tvp_config"] = HSGPKwargs(
                m=self.number_of_basis,
                eta_lam=10,
                cov_func="Matern52",
                L=1.5 * self.client_configuration.client_data.shape[0],
            )
            self.model_variables += [
                "media_temporal_latent_multiplier_raw_eta",
                "media_temporal_latent_multiplier_raw_ls",
            ]

        if self.time_varying_intercept:
            logger.info("Time-varying Intercept MMM.")
            self.model_priors["intercept_tvp_config"] = HSGPKwargs(
                m=self.number_of_basis,
                eta_lam=1,
                cov_func="Matern52",
                L=1.5 * self.client_configuration.client_data.shape[0],
            )
            if "intercept" in self.model_variables:
                self.model_variables.remove("intercept")
            self.model_variables += [
                "intercept_baseline",
                "intercept_temporal_latent_multiplier_raw_eta",
                "intercept_temporal_latent_multiplier_raw_ls",
            ]

    # =============================================================================
    # Data helpers
    # =============================================================================

    def _validate_prediction_input(self, X_new: pd.DataFrame) -> None:
        if not isinstance(X_new, pd.DataFrame):
            raise ValueError("Prediction input must be a pandas DataFrame.")
        if self.date_name not in X_new.columns:
            raise ValueError(f"Date column '{self.date_name}' not found in prediction data.")
        missing_channels = [ch for ch in self.channel_columns if ch not in X_new.columns]
        missing_controls = []
        if self.control_columns:
            missing_controls = [c for c in self.control_columns if c not in X_new.columns]
        missing_all = missing_channels + missing_controls
        if missing_all:
            raise ValueError(f"Missing required columns in prediction DataFrame: {missing_all}")

    def _get_inverse_scaler(self, variable: str):
        if self.client_configuration.scale_data:
            return self.client_configuration.get_inverse_scaler(variable)
        return lambda x: x

    def _rescale_target(self, target: np.ndarray) -> np.ndarray:
        inverse_scaler = self._get_inverse_scaler(variable="target")
        return inverse_scaler(target) if inverse_scaler is not None else target

    def summarize_variable(self, var_name: str):
        if self.model_fit is None:
            raise ValueError("The model hasn't been trained")
        if var_name not in self.model_variables:
            raise ValueError("The specified variable is not available")

        mmm = self.model_fit
        variable = mmm.posterior[var_name].stack(samples=["chain", "draw"]).transpose()
        dims = variable.dims

        if var_name in ["gamma_control"]:
            scaler = self._get_inverse_scaler(variable="controls")
            variable = xr.DataArray(scaler(variable), dims=dims)
            var_tag = self.control_columns

        elif var_name in ["saturation_beta", "saturation_alpha"]:
            scaler = self._get_inverse_scaler(variable="channels")
            variable = xr.DataArray(scaler(variable), dims=dims)
            var_tag = self.channel_columns

        elif var_name in ["intercept", "y_sigma"]:
            scaler = self._get_inverse_scaler(variable="target")
            variable = xr.DataArray(scaler(variable), dims=dims)
            var_tag = var_name

        else:
            var_tag = [f + "-" + var_name for f in self.channel_columns]

        coef = variable.mean(dim="samples").values.tolist()
        ci_up = variable.quantile(q=0.95, dim="samples").values.tolist()
        ci_low = variable.quantile(q=0.05, dim="samples").values.tolist()

        return {"variable": var_tag, "coef": coef, "ci_up_cassandra": ci_up, "ci_low_cassandra": ci_low}

    def get_target(self, original_scale: bool = False) -> np.ndarray:
        return self.client_configuration._get_target(original_scale=original_scale)

    def get_covariates(self, original_scale: bool = False) -> pd.DataFrame:
        return self.client_configuration._get_covariates(original_scale=original_scale)

    def get_data(self, original_scale: bool = False):
        target = self.get_target(original_scale)
        covariates = self.get_covariates(original_scale)
        return target, covariates

    # =============================================================================
    # MMM builder / fit / predict
    # =============================================================================

    def build_pymc_mmm(self) -> MMM:
        default_sampling_config = {
            "progressbar": True,
            "chains": 4,
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "init": "adapt_diag",
            "target_accept": 0.95,
        }

        return MMM(
            date_column=self.date_name,
            channel_columns=self.channel_columns,
            control_columns=self.control_columns if self.control_columns else None,
            yearly_seasonality=None,
            adstock=self.adstock,
            saturation=self.saturation,
            time_varying_media=self.time_varying_media,
            time_varying_intercept=self.time_varying_intercept,
            model_config=self.model_priors,
            sampler_config=default_sampling_config,
        )

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None,
    ):
        self.model_fit = self.build_pymc_mmm()

        if self.has_lift_tests:
            logging.info("Incorporate lift-tests data.frame to MMMM")
            self.model_fit.add_lift_test_measurements(self.lift_tests)

        X = self.get_covariates()
        y = self.get_target(original_scale=True)

        logger.info(
            f"Sampling {self.client_name}'s MMM using {chains} chains, "
            f"{draws} draws, and {tune} tuning steps."
        )

        self.model_fit.fit(
            y=y,
            X=X,
            tune=tune,
            draws=draws,
            cores=cores,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
        )

        logging.info("Compute y_fit using the model's train data")
        self.model_fit.sample_posterior_predictive(
            X=X,
            combined=True,
            var_names=["y"],
            original_scale=True,
            extend_idata=True,
        )
        logger.info("Sampling completed.")

    def predict(self, X_new: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.model_fit is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        if X_new is not None:
            self._validate_prediction_input(X_new)
            X = X_new
        elif hasattr(self.model_fit, "X"):
            X = self.model_fit.X
        else:
            raise ValueError("No covariates provided and none stored in fitted model.")

        y_pred = self.model_fit.sample_posterior_predictive(
            X=X,
            combined=True,
            var_names=["y"],
            original_scale=True,
            extend_idata=False,
        )
        y_fcst = np.transpose(np.array(y_pred["y"]))

        if getattr(self.client_configuration, "scale_data", True):
            y_fcst = self._rescale_target(y_fcst)

        return y_fcst

    # =============================================================================
    # Fleet methods (original behavior)
    # =============================================================================

    def standard_fleet(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
        real_parameters: Optional[Dict[str, Union[float, list]]] = None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None,
    ) -> FleetResult:

        if X_train.shape[0] != len(y_train):
            raise ValueError("X_train and y_train must have the same number of rows.")
        if X_test is not None and (y_test is None or X_test.shape[0] != len(y_test)):
            raise ValueError("X_test and y_test must both be provided and have the same number of rows.")

        temp_mmm = self.build_pymc_mmm()

        if self.has_lift_tests:
            logging.info("Incorporate lift-tests data.frame to MMMM")
            temp_mmm.build_model(X=X_train, y=y_train)
            temp_mmm.add_lift_test_measurements(self.lift_tests)

        logger.info(f"Sampling MMM using {chains} chains, {draws} draws, and {tune} tuning steps.")
        temp_mmm.fit(
            y=y_train,
            X=X_train,
            tune=tune,
            draws=draws,
            cores=cores,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
        )
        logger.info("Sampling completed.")

        if X_test is not None:
            logger.info(f"Predict {len(y_test)}-steps out of sample.")
            X_pred = X_test
        else:
            logger.info(f"Predict {len(y_train)}-steps in sample.")
            X_pred = X_train

        y_fcst = temp_mmm.sample_posterior_predictive(
            X=X_pred,
            combined=True,
            var_names=["y"],
            extend_idata=False,
        )
        y_pred = np.transpose(np.array(y_fcst["y"]))

        return FleetResult(
            mmm=temp_mmm,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            real_parameters=real_parameters,
        )

    def production_fleet(
        self,
        n_test: int = 0,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None,
    ) -> FleetResult:

        logger.info(f"{n_test} steps out of sample production fleet.")
        X = self.get_covariates()
        y = self.get_target(original_scale=True)

        [(X_train, y_train, X_test, y_test)] = rolling_split(X=X, y=y, n_test=n_test, n_splits=1)

        return self.standard_fleet(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            progressbar=progressbar,
            seed=seed,
        )

    def recovery_fleet(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None,
    ) -> List[FleetResult]:

        logger.info("Creating temporary MMM for data simulation.")

        prior_values = {}
        X = self.get_covariates()
        fake_y = np.zeros(X.shape[0])
        params = self.model_variables

        temp_mmm = self.build_pymc_mmm()
        temp_mmm.build_model(X=X, y=fake_y)

        with temp_mmm.model:
            for name in params:
                rv = temp_mmm.model[name]
                sampled_var = pm.draw(rv, draws=1, random_seed=seed)
                sampled_var_np = np.array(sampled_var)
                if rv.shape == ():
                    prior_values[name] = sampled_var_np.item()
                else:
                    prior_values[name] = sampled_var_np

        logger.info("Success: Simulated parameters.")
        true_model = pm.do(model=temp_mmm.model, vars_to_interventions=prior_values)
        simulated_target = pm.draw(true_model.y, draws=1, random_seed=seed)
        logger.info("Success: Simulated data.")

        return self.standard_fleet(
            X_train=X,
            y_train=simulated_target,
            X_test=None,
            y_test=None,
            real_parameters=prior_values,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            progressbar=progressbar,
            seed=seed,
        )
