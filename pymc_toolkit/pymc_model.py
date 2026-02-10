import logging
import pymc as pm
import numpy as np
import pandas as pd
import xarray as xr

from enum import Enum

from numba.core.transforms import with_lifting
from pymc_extras.prior import Prior
from typing import Optional, Tuple, Dict, Union, List
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_toolkit.fleet_result import FleetResult
from pymc_toolkit.client_config import ClientConfig
from pymc_marketing.mmm import MichaelisMentenSaturation, LogisticSaturation
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation, WeibullPDFAdstock

from pymc_toolkit.utils import (rolling_split,
                                    recovery_summary)

logger = logging.getLogger(__name__)

class SaturationType(Enum):
    LOGISTIC = "logistic"
    HILL = "hill"
    MICHAELIS_MENTEN = "michaelis_menten"

class AdstockType(Enum):
    GEOMETRIC = "geometric"
    WEIBULL_PDF = "weibull_pdf"

class EventBumpsBasis(Enum):
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

    Parameters
    ----------
    client_data : pd.DataFrame
       Client input data, including channel, control, and target variables.
    channel_names : list[str]
       List of column names corresponding to the marketing channels.
    date_column : str, default="ds"
       Name of the column containing the dates.
    control_names : list[str] or None, default=None
       List of control variable column names. Can be None if not used.
    target_name : str, default="y"
       Name of the target (dependent) variable column for the model.
    client_name : str or None, default=None
       Optional name to identify the client.
    lag_max : int, default=1
       Maximum number of lags to consider in the adstock function.
    scale_data : bool, default=False
       Whether to scale variables before modeling.
    priors : dict[str, Prior] or None, default=None
       Dictionary of custom priors. If None, default values are used.
    saturation : str or SaturationType, default=SaturationType.HILL
       Type of saturation function to use (hill, logistic, or michaelis_menten).
    adstock : str or AdstockType, default=AdstockType.GEOMETRIC
       Type of adstock function to use (geometric or weibull_pdf).
    number_of_basis: int, default=50,
          The number of basis functions (m) used when applying Hilbert-space Gaussian
          process priors in time-varying media and time-varying intercept.
    time_varying_media : bool, default=False
       If True, allows media coefficients to vary over time.
    time_varying_intercept : bool, default=False
      If True, allows the intercept to vary over time.
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
      time_varying_intercept: bool = False):
      # --- EVENTS (optional) ---
      df_events: Optional[pd.DataFrame] = None
      events_basis: Union[str, EventBumpsBasis] = EventBumpsBasis.GAUSSIAN,
      events_sigma_days: float = 7.0,
      events_half_mode: Union[str, HalfGaussianMode] = HalfGaussianMode.AFTER,
      events_sigma_before_days: float = 7.0,
      events_sigma_after_days: float = 14.0,
      events_prefix: str = "event",
      events_reference_date: Optional[str] = None,
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
          client_name=client_name)

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

      # define the model's variable
      logger.info("Set up PyMCModel's basic configuration.")
      self.number_of_basis = number_of_basis
      self.time_varying_media = time_varying_media
      self.lag_max = self.client_configuration.lag_max
      self.time_varying_intercept = time_varying_intercept
      self.date_name = self.client_configuration.date_name
      self.client_name = self.client_configuration.client_name
      self.channel_columns = self.client_configuration.channel_names
      self.control_columns = self.client_configuration.control_names
      self.lift_tests = self.client_configuration.calibration_inputs

      # Store the sampled MMM
      self.model_fit = None
      self.has_lift_tests = False if self.lift_tests is None else True

      # define model variables and priors
      logger.info("Define PymcModel's default priors.")
      self.model_variables = ['y_sigma','intercept']

      logger.info("Creating default priors for intercept and model's scale (y_sigma).")
      self.model_priors = {
        "intercept":Prior("HalfNormal",sigma=1),
        'likelihood':Prior("Normal",sigma=Prior("HalfNormal",sigma=2))
      }

      if self.control_columns:
        logger.info("Creating default priors for gamma control.")
        self.model_variables.append('gamma_control')
        self.model_priors["gamma_control"]=Prior("HalfNormal",sigma=1,dims="control")

      self._set_adstock(adstock)
      self._set_saturation(saturation)
      self._set_time_varying_model()

      ### update priors
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
        f"Saturation={self.saturation_type.value})")

      # =============================================================================
      # Events
      # =============================================================================

      @staticmethod
      def _coerce_enum(value, enum_cls):
          if isinstance(value, enum_cls):
              return value
          elif isinstance(value, str):
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
            basis: Union[str, EventBumpsBasis],
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
          try:
              missing = required.difference(df_events.columns)
              if missing:
                  logger.error(f"df_events is missing columns {missing}. Required: {required}")
                  return pd.DataFrame(), []
          except Exception as e:
              logger.exception("Unexpected error while validating df_events")
              return  pd.DataFrame(), []
          basis = self._coerce_enum(basis, EventBumpsBasis)
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

              if basis == EventBumpsBasis.GAUSSIAN:
                  values = _gaussian_bump(x, sigma_days)
              elif basis == EventBumpsBasis.HALF_GAUSSIAN:
                  values = _half_gaussian_bump(x, sigma_days, mode=half_mode)
              elif basis == EventBumpsBasis.ASYMMETRIC_GAUSSIAN:
                  values = _asymmetric_gaussian_bump(x, sigma_before_days, sigma_after_days)
              else:
                  raise ValueError("Unsupported EventBumpsBasis")

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
      """Internal helper to set saturation function and related priors."""
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

        logger.info("Creating default priors for saturation lambda and beta.")  
        self.model_variables.append('saturation_lam')
        self.model_variables.append('saturation_beta')
        self.model_priors['saturation_lam']=Prior("Gamma",alpha=3,beta=1,dims="channel")
        self.model_priors['saturation_beta']=Prior("HalfNormal",sigma=2,dims="channel")

      elif saturation == SaturationType.MICHAELIS_MENTEN:
        logger.info("Using Michaelis-Menten Saturation.")
        self.saturation = MichaelisMentenSaturation()

        logger.info("Creating default priors for saturation lambda and alpha.")  
        self.model_variables.append('saturation_lam')
        self.model_variables.append('saturation_alpha')
        self.model_priors['saturation_lam']=Prior("HalfNormal",sigma=1,dims="channel")
        self.model_priors['saturation_alpha']=Prior("Gamma",mu=2,sigma=1,dims="channel")
        
      elif saturation == SaturationType.HILL:
        logger.info("Using Hill Saturation.")
        self.saturation = HillSaturation()

        logger.info("Creating default priors for saturation slope, kappa and betaa.") 
        self.model_variables.append('saturation_slope')
        self.model_variables.append('saturation_kappa')
        self.model_variables.append('saturation_beta')
        self.model_priors["saturation_slope"]=Prior("Normal",mu=1,sigma=0.001,dims='channel')
        self.model_priors['saturation_kappa']=Prior("HalfNormal",sigma=1.5,dims="channel")
        self.model_priors['saturation_beta']=Prior("HalfNormal",sigma=1.5,dims="channel")
        
      else:
        logger.error(f"Unsupported SaturationType received: {saturation}")
        raise ValueError(f"Unsupported SaturationType: {saturation}")
    
    def _set_adstock(self, adstock: Union[str, AdstockType]):
      """Internal helper to set adstock function."""
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
        self.model_variables.append('adstock_alpha')
        self.model_priors['adstock_alpha']=Prior("Beta",alpha=1,beta=3,dims="channel")
        logger.info("Creating default priors for adstock alpha.") 

      elif adstock == AdstockType.WEIBULL_PDF:
        logger.info("Using Weibull PDF Adstock.") 
        self.adstock = WeibullPDFAdstock(l_max=self.lag_max)
        self.model_variables.append('adstock_lam')
        self.model_variables.append('adstock_k')
        self.model_priors['adstock_k']=Prior("Gamma",mu=3,sigma=1,dims="channel")
        self.model_priors['adstock_lam']=Prior("Gamma",mu=2,sigma=1,dims="channel")
        logger.info("Creating default priors for adstock k and lambda.") 

      else:
          logger.error(f"Unsupported AdstockType received: {adstock}")
          raise ValueError(f"Unsupported AdstockType: {adstock}")
    
    def _set_time_varying_model(self):
      """Internal helper to set time varying media models."""
      if self.time_varying_media:
        logger.info("Time-varying Media MMM.") 
        self.model_priors['media_tvp_config'] = HSGPKwargs(
                      m=self.number_of_basis,
                      eta_lam=10, 
                      cov_func="Matern52",
                      L=1.5 * self.client_configuration.client_data.shape[0]
                    )
        self.model_variables.append("media_temporal_latent_multiplier_raw_eta")
        self.model_variables.append("media_temporal_latent_multiplier_raw_ls")
        logger.info("Creating default priors for Kernel's eta and length-scale.") 
      
      if self.time_varying_intercept:
        logger.info("Time-varying Intercept MMM.") 
        self.model_priors['intercept_tvp_config'] = HSGPKwargs(
                      m=self.number_of_basis,
                      eta_lam=1, 
                      cov_func="Matern52",
                      L=1.5 * self.client_configuration.client_data.shape[0]
                    )
        self.model_variables.remove('intercept')
        self.model_variables.append('intercept_baseline')
        self.model_variables.append('intercept_temporal_latent_multiplier_raw_eta')
        self.model_variables.append('intercept_temporal_latent_multiplier_raw_ls')
        logger.info("Creating default priors for Kernel's eta and length-scale.")


    def _validate_prediction_input(self, X_new: pd.DataFrame) -> None:
      """
      Validate that the prediction input DataFrame contains all required columns,
      including date, channels, and controls.
      
      Parameters
      ----------
      X_new : pd.DataFrame
        New covariate data for prediction.
      
      Raises
      ------
      ValueError
        If the DataFrame is not valid or missing required columns.
      """      
      if not isinstance(X_new, pd.DataFrame):
        logger.error("Prediction input is not a pandas DataFrame.")
        raise ValueError("Prediction input must be a pandas DataFrame.")

      if self.date_name not in X_new.columns:
        logger.error(f"Date column '{self.date_name}' not found in prediction data.")
        raise ValueError(f"Date column '{self.date_name}' not found in prediction data.")

      missing_controls = []
      missing_channels = [ch for ch in self.channel_columns if ch not in X_new.columns]
      
      if self.control_columns:
        missing_controls = [c for c in self.control_columns if c not in X_new.columns]
      
      missing_all = missing_channels + missing_controls
        
      if missing_all:
        logger.error(f"Missing required columns for prediction: {missing_all}")
        raise ValueError(f"Missing required columns in prediction DataFrame: {missing_all}")

    def _rescale_target(self, target: np.ndarray) -> np.ndarray:
      """
      Rescale target values from the scaled space back to the original scale.
      Parameters
      ----------
      target : np.ndarray
        Target values in scaled form (e.g., after standardization).
        
      Returns
      -------
      np.ndarray
        Target values rescaled to their original scale. If no inverse scaler
        is available, returns the input unchanged.
      """
      inverse_scaler = self._get_inverse_scaler(variable="target")
      
      if inverse_scaler is None:
        logger.warning("No inverse scaler found for target; returning values unchanged.")
        return target
        
      logger.info("Applying inverse scaling transformation to target predictions.")
      return inverse_scaler(target)
    
    def _get_inverse_scaler(self, variable:str):
      """
      Returns a callable that applies the inverse transformation of the control scaler.
      Compatible with both 1D and 2D arrays.
      If scaling is not applied, returns the identity function.
      Parameters
      ----------
      variable : str
          One of ["target", "channels", "controls"].
      Returns
      -------
      callable
      A function that takes a numpy array and returns the inverse-scaled array.
      """      
      if self.client_configuration.scale_data:
        return self.client_configuration.get_inverse_scaler(variable)
      else:
        return lambda x: x

    def summarize_variable(self, var_name: str):
      """
      Summarize a posterior variable from the trained PyMC model.
      
      This function computes the mean and credible interval (5th and 95th percentiles)
      for a given model variable. If the variable is subject to scaling (e.g., control, channel, or target),
      it applies the corresponding inverse scaler before summarization.
      
      Parameters
      ----------
      var_name : str
          The name of the variable to summarize. Must be one of the model's available variables.
      
      Returns
      -------
      dict
        A dictionary containing:
         - "variable": The name(s) of the variable(s) or columns.
         - "coef": Posterior mean(s) of the variable(s).
         - "ci_up_cassandra": 95th percentile(s) of the posterior.
         - "ci_low_cassandra": 5th percentile(s) of the posterior.
      
      Raises
      ------
      ValueError
         If the model has not been trained or if the specified variable does not exist.
      
      Notes
       -----
        - The function stacks 'chain' and 'draw' dimensions into a single 'samples' dimension
          to compute summary statistics.
        - For control or channel variables, the corresponding inverse scaler is applied
          before computing mean and quantiles.
        - For non-scaled variables like 'intercept' or 'y_sigma', the target inverse scaler is used.
       """
      if self.model_fit is None:
        logger.error("The model hasn't been trained")
        raise ValueError("The model hasn't been trained")

      mmm = self.model_fit
      
      if var_name not in self.model_variables:
        logger.error("The specified variable is not available")
        raise ValueError("The specified variable is not available")
      
      variable = mmm.posterior[var_name].stack(samples=['chain','draw']).transpose()
      dims = variable.dims

      if var_name in ['gamma_control']:
        scaler = self._get_inverse_scaler(variable="controls")
        variable = xr.DataArray(scaler(variable),dims=dims)
        var_tag = self.control_columns

      elif var_name in ['saturation_beta','saturation_alpha']:
        scaler = self._get_inverse_scaler(variable="channels")
        variable = xr.DataArray(scaler(variable),dims=dims)
        var_tag = self.channel_columns

      elif var_name in ['intercept','y_sigma']:
        scaler = self._get_inverse_scaler(variable="target")
        variable = xr.DataArray(scaler(variable),dims=dims)
        var_tag = var_name

      else:
        var_tag = [f + "-" + var_name for f in self.channel_columns]
    
      coef = variable.mean(dim ='samples').values.tolist()
      ci_up =  variable.quantile(q=0.95,dim='samples').values.tolist()
      ci_low = variable.quantile(q=0.05,dim='samples').values.tolist()
      
      return {
        "variable":var_tag,
        "coef": coef,
        "ci_up_cassandra": ci_up,
        "ci_low_cassandra": ci_low}

    def get_target(self, original_scale: bool = False) -> np.ndarray:
      """
      Returns the target variable as a NumPy array.
       
      Parameters
      ----------
      original_scale : bool
        If True and the target was scaled, return the original (unscaled) values.
        
      Returns
      -------
      np.ndarray
        Target variable as a NumPy array (1D). 
        
      Raises
      ------
        Warning
        If no scaler was used adn original_scale is True.
      """
      return self.client_configuration._get_target(original_scale=original_scale)

    def get_covariates(self, original_scale: bool = False) -> pd.DataFrame:
      """
      Returns a DataFrame containing the date, channel, and control variables. 
      Parameters
      ----------
      original_scale : bool, optional
          If True and data were scaled, return the original (unscaled) values.
        
      Returns
      -------
      pd.DataFrame
          DataFrame with columns: date_column, channel columns, and control columns.
      """        
      return self.client_configuration._get_covariates(original_scale=original_scale)
        
    def get_data(self, original_scale: bool = False):
      """
      Returns the target variable as a NumPy array, and the covariates 
      table as a data.frame.
      
      Parameters
      ----------
      original_scale : bool
        If True and the target was scaled, return the original (unscaled) values.
        
      Returns
      -------
      np.ndarray
        Target variable as a NumPy array (1D). 
      pd.DataFrame
          DataFrame with columns: date_column, channel columns, and control columns.
      """
      target = self.get_target(original_scale)
      covariates = self.get_covariates(original_scale)
      return target, covariates

    def build_pymc_mmm(self) -> MMM:
      """
      Create a Media Mix Model (MMM) configured for this client.        
      Returns
      -------
      MMM
        Configured a pymc-marketing MMM instance ready to be 
        fitted or used for simulation.
        
      Raises
      ------
      ValueError
        If the saturation type is not supported.
      """
      default_sampling_config = {
            "progressbar": True,
            "chains": 4,
            "draws": 1000,
            "tune": 1000,
            "cores": 4,
            "init": "adapt_diag",
            "target_accept": 0.95}
        
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
          sampler_config=default_sampling_config)

    def fit(self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None):
      """
      Fit the PyMC MMM model with the provided sampling configuration.
       
      Parameters
      ----------
      draws : int, default=1000
          Number of posterior samples to draw after tuning.
      tune : int, default=1000
          Number of tuning (burn-in) steps before collecting samples.
      chains : int, default=4
          Number of MCMC chains to run.
      cores : int, default=4
          Number of CPU cores to use for parallel sampling.
      progressbar: bool, default=True
          Shows the MCMC iteration process when its True.
      seed : int or None, default=None
          Random seed for reproducibility.
      """
      self.model_fit = self.build_pymc_mmm()

      if self.has_lift_tests:
        logging.info("Incorporate lift-tests data.frame to MMMM")
        self.model_fit.add_lift_test_measurements(self.lift_tests)

      X = self.get_covariates()
      y = self.get_target(original_scale=True)
        
      logger.info(f"Sampling {self.client_name}'s MMM using {chains} chains, "
          f"{draws} draws, and {tune} tuning steps.")

      self.model_fit.fit(y=y,
                      X=X,
                      tune=tune,
                      draws=draws,
                      cores=cores,
                      chains=chains,
                      random_seed=seed,
                      progressbar=progressbar)

      logging.info("Compute y_fit using the model's train data")
      self.model_fit.sample_posterior_predictive(X=X,
                                                combined=True,
                                                var_names=["y"],
                                                original_scale=True,
                                                extend_idata=True)
      logger.info("Sampling completed.")
    
    def predict(self, X_new: Optional[pd.DataFrame] = None) -> np.ndarray:
      """
      Generate predictions from the fitted PyMC MMM model.
      
      Parameters
      ----------
      X_new : 
        pd.DataFrame or None, default=None. New covariate 
        data to predict on. If None, uses the covariates from 
        the training set.
      
      Returns
      -------
        np.ndarray: Posterior predictive samples for the target variable.
      """
      if self.model_fit is None:
        logger.error("Prediction attempted before fitting the model.")
        raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

      if X_new is not None:
        logger.info("Validating prediction input DataFrame.")
        self._validate_prediction_input(X_new)
        X = X_new
      elif hasattr(self.model_fit, "X"): 
        X = self.model_fit.X
      else:
        logger.error("No covariates found for prediction.")
        raise ValueError("No covariates provided and none stored in fitted model.")

      y_pred = self.model_fit.sample_posterior_predictive(X=X,
                                                    combined=True,
                                                    var_names=["y"],
                                                    original_scale=True,
                                                    extend_idata=False)      
      y_fcst = np.transpose(np.array(y_pred["y"]))

      if getattr(self.client_configuration, "scale_data", True):
        logger.info("Rescaling predictions to original target scale.")
        y_fcst = self._rescale_target(y_fcst)
      
      return y_fcst    
          
############################
# Fleet methods
############################

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
        seed: Optional[int] = None) -> FleetResult:
        """
        Fit an MMM model and generate posterior predictive samples for one train/test split.

        Parameters
        ----------
        X_train : pd.DataFrame
          Training features.
        y_train : np.ndarray
          Training target values.
        X_test : pd.DataFrame, optional
          Test features (out-of-sample).
        y_test : np.ndarray, optional
          Test target values.
        draws : int, default=1000
          Number of MCMC draws.
        tune : int, default=1000
          Number of tuning steps.
        chains : int, default=4
          Number of MCMC chains.
        cores : int, default=4
          Number of CPU cores for parallel sampling.
        progressbar : bool, default=True
          Whether to display a progress bar during sampling.
        seed : int, optional
         Random seed for reproducibility.

        Returns
        -------
        FleetResult
          An object containing the fitted model, predictions, and data used.

        Raises
        ------
        ValueError
          If X_train and y_train row counts do not match.
          If X_test and y_test are mismatched or one is missing.
        """
        if X_train.shape[0] != len(y_train):
            logger.error("X_train and y_train must have the same number of rows.")
            raise ValueError("X_train and y_train must have the same number of rows.")
        if X_test is not None and (y_test is None or X_test.shape[0] != len(y_test)):
            logger.error("X_test and y_test must both be provided and have the same number of rows.")
            raise ValueError("X_test and y_test must both be provided and have the same number of rows.")
        
        temp_mmm = self.build_pymc_mmm()
        if self.has_lift_tests:
          logging.info("Incorporate lift-tests data.frame to MMMM")
          temp_mmm.build_model(X=X_train,y=y_train)
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
            progressbar=progressbar)
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
            extend_idata=False)
        
        y_pred = np.transpose(np.array(y_fcst["y"]))
        
        return FleetResult(
            mmm=temp_mmm,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            real_parameters=real_parameters)
        
    def production_fleet(
        self,
        n_test: int = 0,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None) -> FleetResult:
        """
        Run a production fleet evaluation for a single train/test split.
        
        Parameters
        ----------
        n_test : int, default=0
          Number of observations to include in the test set.
        draws, tune, chains, cores, progressbar, seed
          Same as in `standard_fleet`.
        
        Returns
        -------
        FleetResult
          The result of the production fleet run.
        """
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
            seed=seed)
  
    def recovery_fleet(
        self, 
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 4,
        progressbar: bool = True,
        seed: Optional[int] = None) -> List[FleetResult]:
        """
        Simulates target data using a Media Mix Model (MMM) with prior parameters,
        then fits a new MMM on the simulated data to test parameter recovery.
        
        Parameters
        ----------
        draws, tune, chains, cores, progressbar, seed
          Same as in `standard_fleet`.
        
        Returns
        -------
        FleetResult
          The result of the standard fleet run with no test data.
        """
        logger.info("Creating temporary MMM for data simulation.")
        
        prior_values = {}
        X = self.get_covariates()
        fake_y = np.zeros(X.shape[0])
        params = self.model_variables
        temp_mmm = self.build_pymc_mmm() 
        
        # build pymc model from MMM
        temp_mmm.build_model(X=X,y = fake_y)
        
        ## Sample parameters
        with temp_mmm.model:
          for name in params:
            rv = temp_mmm.model[name]
            sampled_var = pm.draw(rv, draws=1,random_seed=seed)
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
            real_parameters = prior_values,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            progressbar=progressbar,
            seed=seed)
