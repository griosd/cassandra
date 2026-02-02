import logging
import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from pymc_toolkit.utils import get_all_zero_columns, get_media_with_negatives

# Configure logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ClientConfig:
    """
    Class to generate and manage data configuration for a Media Mix 
    Modeling (MMM) analysis. This class stores, cleans, and splits data 
    for the pymc_model_config class.
    """

    def __init__(self, 
                 client_data: pd.DataFrame,
                 channel_names: list,
                 date_column: str = "ds",
                 control_names: list = None, 
                 target_name: str = 'y',
                 client_name: str = None,
                 calibrations: dict = None,
                 lag_max: int = 1,
                 scale_data: bool = False):
        """
        Initializes a ClientConfig instance.

        Parameters
        ----------
        client_data : pd.DataFrame
            Raw client data containing the date column. (Required)
        channel_names : list of str
            Names of media channels.
        date_column : str, optional
            Name of the date column in `client_data`.
        control_names : list of str, optional
            Names of control variables. Defaults to an empty list.
        target_name : str, optional
            Name of the dependent variable. Defaults to "y".
        client_name : str, optional
            Name of the client. Defaults to "client0".
        lag_max : int, optional
            Maximum lag for the geometric adstock transformation. 
            Must be positive. Default is 7.

        Raises
        ------
        ValueError:
            If required arguments are missing or inconsistent.
        """   
        self.client_name = client_name or "client0"
        self.target_name = target_name or "y"
        self.date_name = date_column
        self.channel_columns = channel_names
        self.control_columns = control_names or []
        
        self.invalid_controls = []
        self.invalid_channels = []

        self.lag_max = lag_max
        self.scale_data = scale_data
        
         ## Validate Client data
        self._validate_columns(client_data, date_column)
        self.client_data = client_data.copy()

        ## Validate media and controls
        self._validate_media()
        self._validate_controls()
        self._get_unused_columns()

        # Empty data scaleres
        self.controls_scaler = None
        self.channels_scaler = None
        self.target_scaler = None

        self._fit_scalers()
        self.channel_scales_dict = self._extract_channel_scalers_dict()
        self._extract_target_scale()

        # clean calibration inputs
        if calibrations is not None: 
            self.calibration_inputs = self._get_calibration_input_for_pymc(calibration_tests=calibrations,
                                                            channel_scales=self.channel_scales_dict)
        else:
            self.calibration_inputs = None

        logger.info(f"Client data loaded for '{self.client_name}'.")
    
    def _validate_columns(self, df:pd.DataFrame, date_column:str):
        "Evaluate that the specified media and controls are columns within the client data"
        logger.info("Validate Client's data")
        if not isinstance(df, pd.DataFrame):
            logger.error("Provided data is not a pandas DataFrame.")
            raise ValueError("Provided data is not a pandas DataFrame.")

        if date_column not in df.columns:
            logger.error(f"Date column '{date_column}' not found in data.")
            raise ValueError(f"Date column '{date_column}' not found in data.")

        missing_channels = [ch for ch in self.channel_columns if ch not in df.columns]
        missing_controls = [c for c in self.control_columns if c not in df.columns]
        missing_target = [] if self.target_name in df.columns else [self.target_name]
        missing_date = [] if self.date_name in df.columns else [self.date_name]
        missing_all = missing_channels + missing_controls + missing_target + missing_date

        if missing_all:
            logger.error(f"Missing required columns: {missing_all}")
            raise ValueError(f"Missing required columns in DataFrame: {missing_all}")
    
    def _validate_media(self):
        """
        Identifies media channels with zero values or negative spends, and remove them
        from the analysis
        """
        logger.info("Validate Media channels")
        self.invalid_channels = (
            get_all_zero_columns(self.client_data[self.channel_columns]) +
            get_media_with_negatives(self.client_data, self.channel_columns))

        # remove duplicates                                                
        self.invalid_channels = list(set(self.invalid_channels))

        if self.invalid_channels: 
            logger.warning(
              f"Invalid media columns: {self.invalid_channels},"+ 
               "may have zero spend channels or channels with"+
               "negative spend and will be discarded from the analysis")
        
        self.channel_names=[x for x in self.channel_columns if x not in self.invalid_channels]
        logger.info(f"available channels:{self.channel_names}")

    def _validate_controls(self):
        """
        Identifies controls with zero values, and remove them from the analysis.
        """
        logger.info("Validate control channels")
        if not self.control_columns:
            self.control_names = []
            logger.info("No controls declared from the client")
            return
        
        self.invalid_controls += get_all_zero_columns(self.client_data,
                                                    columns=self.control_columns)
        if self.invalid_controls: 
            logger.warning(
                f"Invalid control: {self.invalid_controls},"+ 
                "are columns with zero values and will be discarded from the analysis")
            
        self.control_names=[x for x in self.control_columns if x not in self.invalid_controls]
        logger.info(f"available controls:{self.control_names}")
    
    def _get_unused_columns(self):
        "Identify unused columns from Client data"
        logger.info("Identify unused columns from client's data")
        used_data = self.channel_names + self.control_names + [self.date_name]+[self.target_name]
        self.unused_columns = [col for col in self.client_data.columns if col not in used_data]
        if self.unused_columns: 
            logger.info(f"Unused columns: {self.unused_columns}")

    def _fit_scalers(self):  
        """Extracts and stores the target vector and covariate DataFrame."""
        
        if not self.scale_data:
            return
        
        self.target_scaler = MaxAbsScaler()
        target = self.client_data[self.target_name].copy().values.flatten()
        self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        logger.info("Target variable scaled using MaxAbsScaler.")
            
        # Channels
        channels = self.client_data[self.channel_names].copy()
        self.channels_scaler = MaxAbsScaler()
        self.channels_scaler.fit_transform(channels)
        logger.info("Channel variables scaled using MaxAbsScaler.")
        
        # Controls
        if self.control_names:
            controls = self.client_data[self.control_names].copy()
            self.controls_scaler = MaxAbsScaler()
            self.controls_scaler.fit_transform(controls)
            logger.info("Control variables scaled using MaxAbsScaler.")
        else:
            logger.info("No control variables available to scale.")
        
        logger.debug("Target, channels, and controls scaled.")
    
    def __repr__(self):
        lines = [
            f"ClientConfig(client_name='{self.client_name}', target='{self.target_name}', date_column='{self.date_name}')",
            f"Channels: {self.channel_names or 'None'}",
            f"Invalid channels: {self.invalid_channels or 'None'}",
            f"Controls: {self.control_names or 'None'}",
            f"Invalid controls: {self.invalid_controls or 'None'}",
            f"Unused columns: {self.unused_columns or 'None'}"
        ]
     
        return "\n".join(lines)

    def _get_target(self, original_scale: bool = False) -> np.ndarray:
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
        target = self.client_data[self.target_name].copy().values.flatten()

        if original_scale is False and self.scale_data:
            logger.info("Returning scaled target variable.")
            scaler = self.get_scaler(variable="target")
            return scaler(target)
        else:
            logger.info("Returning target in original scale.")
            return target
    
    def _get_controls(self, original_scale: bool = False) -> pd.DataFrame:
        """
        Returns the control data as a DataFrame. 
        
        Parameters
        ----------
        original_scale: bool
          If True, returns controls in original (unscaled) scale if possible.
          Otherwise returns scaled controls.
        
        Returns
        -------
        pd.DataFrame
          Control variables in scaled or original scale.
        
        Raises
        ------
        Warning
          If original_scale=True but no scaler is available, warns and returns data as stored.
        """
        if not self.control_names:
            logger.info("No control variables available; returning empty DataFrame.")
            return pd.DataFrame(index=self.client_data.index)
        
        controls = self.client_data[self.control_names]

        if original_scale is False and self.scale_data:
            logger.info("Returning controls in scaled form.")
            return pd.DataFrame(
                    self.controls_scaler.transform(controls),
                    columns=self.control_names,
                    index=controls.index)
        else:
            logger.info("Returning controls in original scale.")
            return pd.DataFrame(
                controls,
                columns=self.control_names,
                index=controls.index)
    
    def _get_channels(self, original_scale:bool = False) -> pd.DataFrame:
        """
        Returns the channel data in a data.frame format.
        Parameters
        ----------
        original_scale: bool
            Indicates if channels return at model or original scale.
        Returns
        -------
        pd.DataFrame
           DataFrame with the channel values.
        Raises
        ------
          Warning
          If no scaler was used adn original_scale is True.
        """
        channels = self.client_data[self.channel_names]

        if original_scale is False and self.scale_data:
            logger.info("Returning channels in scaled form.")
            return pd.DataFrame(self.channels_scaler.transform(channels),
                        columns=self.channel_names,
                        index=channels.index)
        else:
            logger.info("Returning channels in original (unscaled) scale.")
            return pd.DataFrame(channels,
                    columns=self.channel_names,
                    index=channels.index)

    def _get_covariates(self, original_scale: bool = False) -> pd.DataFrame:
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
        logger.debug("Building covariates DataFrame with valid media and valid controls")
        dates = pd.to_datetime(self.client_data[self.date_name])
        date_df = dates.to_frame(name=self.date_name)
        channels_df = self._get_channels(original_scale=original_scale)
        controls_df = self._get_controls(original_scale=original_scale)
        covariates_df = pd.concat([date_df, channels_df, controls_df], axis=1)
        logger.info(f"Covariates DataFrame constructed with shape {covariates_df.shape}.")
        
        return covariates_df
    
    def _get_data(self, original_scale: bool = False):
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
        target = self._get_target(original_scale)
        covariates = self._get_covariates(original_scale)
        return target, covariates
    
    def _extract_channel_scalers_dict(self):
        """Internal function that build a dictionary with the channel scales."""
        if self.channels_scaler is None:
            channel_scales = [1] * len(self.channel_columns)
        else:
            channel_scales = self.channels_scaler.max_abs_.tolist()
        
        return dict(zip(self.channel_columns, channel_scales))
    
    def _extract_target_scale(self):
        """Internal function that returns the Target's scale value."""
        if self.target_scaler is None: 
            self.target_scale_value = 1.0
        else:
            self.target_scale_value = float(self.target_scaler.max_abs_)

    def _get_calibration_input_for_pymc(self, calibration_tests, channel_scales):
        """
        Builds a pymc_marketing-compatible DataFrame from calibration data and channel scales.
        Parameters:
        ----------
        calibration (list of dict): 
           Calibration events, each with required fields:
            - 'Channel'
            - 'Pre_Spend_Expected'
            - 'Delta_Spend'
            - 'Delta_Revenue_Scaled'
            - 'mIROAS_std'
        channel_scales (dict): 
           Mapping of channel names to MaxAbsScaler scale values.
        
        Returns:
        --------
          pd.DataFrame: DataFrame with columns ['channel', 'x', 'delta_x', 'delta_y', 'sigma']
        
        Raises:
        ------
          ValueError: If required fields are missing in any calibration entry,
                    or if any channel is missing in the scale dictionary.
        """
        required_fields = {'Channel','Pre_Spend_Expected','Delta_Spend',
                    'Delta_Revenue_Scaled', 'mIROAS_std'}
        
        # --- Validate all calibration entries BEFORE processing ---
        for idx, entry in enumerate(calibration_tests):
            missing = required_fields - entry.keys()
            if missing:
                logger.warning(f"Missing fields {missing} in calibration entry at index {idx}: {entry}")
                return None
            if entry['Channel'] not in channel_scales:
                logger.warning(f"Missing scale for channel '{entry['Channel']}' in channel_scales.")
                return None
        
        # --- Proceed with transformation ---
        rows = []
        
        for entry in calibration_tests:
            channel = entry['Channel']
            scale = channel_scales[channel]
            
            x = entry['Pre_Spend_Expected'] / scale
            delta_x = entry['Delta_Spend'] / scale
            delta_y = entry['Delta_Revenue_Scaled']
            sigma = entry['mIROAS_std'] * delta_x
            
            rows.append({
                'channel': channel,
                'x': x,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'sigma': sigma
            })
        
        return pd.DataFrame(rows)

    @property
    def coords(self):
        return {
            "lag_max": self.lag_max,
            "client_name": self.client_name,
            "target": self.target_name,
            "n_timesteps": self.client_data.shape[0],
            "n_channels": len(self.channel_names),
            "channels": self.channel_names,
            "invalid_channels": self.invalid_channels,
            "n_controls": len(self.control_names),
            "controls": self.control_names,
            "invalid_controls": self.invalid_controls,    
            "ds": pd.to_datetime(self.client_data[self.date_name]),
            "unused_columns": self.unused_columns,
            "scale_data": self.scale_data,
        }

    def to_dict(self):
        """
        Returns the configuration as a dictionary.

        Returns
        -------
        dict
            Configuration dictionary for MMM simulation.
        """
        logger.debug("Exporting client configuration to dictionary.")
        return self.coords

    def add_new_data(self, new_data:pd.DataFrame):
        """
        Updates the client configuration with new data and validates required columns.

        Parameters
        ----------
        new_data : pd.DataFrame
            A new dataset containing media, control, date, and target variables.

        Raises
        ------
        ValueError:
            If the date column, target, channel, or control columns are missing 
            from the DataFrame.
        """
        required_cols = [self.target_name] + self.channel_columns + self.control_columns + [self.date_name]
        
        missing_cols = [c for c in required_cols if c not in new_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns in new_data: {missing_cols}")
            raise ValueError(f"Missing required columns in new_data: {missing_cols}")
        
        new_data_aligned = new_data.reindex(columns=self.client_data.columns)

        updated_data = pd.concat([self.client_data, new_data_aligned], ignore_index=True)
        self._validate_columns(updated_data, self.date_name)
        self.client_data = updated_data.copy()
        self._fit_scalers()

        logger.info("New client data updated successfully.")

    def get_scaler(self, variable: str):
        """
        Returns a function to scale the requested variable using the corresponding scaler.
        
        Parameters
        ----------
        variable : str
          One of ["target", "channels", "controls"].
        
        Returns
        -------
        function
          A function that takes input and returns scaled output.
        
        Raises
        ------
        ValueError
           If the variable name is invalid or no scaler is available.
        """
        valid_vars = ["target", "channels", "controls"]
        
        if variable not in valid_vars:
            logger.error(f"Invalid variable name '{variable}'. Must be one of {valid_vars}.")
            raise ValueError(f"Variable must be one of {valid_vars}")
        
        scaler = getattr(self, f"{variable}_scaler", None)    
        if scaler is None:
            logger.warning(f"No scaler available for '{variable}'. Data may not have been scaled.")
            return lambda x: x
        
        logger.info(f"Scaler function for '{variable}' returned.")
        def forward(x):
          x = np.asarray(x)
          if x.ndim == 1:
            x = x.reshape(-1, 1)
            return scaler.transform(x).flatten()
          else:
            return scaler.transform(x)
        
        return forward        
    
    def get_inverse_scaler(self, variable: str):
        """
        Returns a function to inverse scale the requested variable using the corresponding scaler.
        
        Parameters
        ----------
        variable : str
          One of ["target", "channels", "controls"].
        
        Returns
        -------      
        function
          A function that takes input and returns unscaled output.
        
        Raises
        ------
        ValueError
          If the variable name is invalid or no scaler is available.
        """
        valid_vars = ["target", "channels", "controls"]
        
        if variable not in valid_vars:
            logger.error(f"Invalid variable name '{variable}'. Must be one of {valid_vars}.")
            raise ValueError(f"Variable must be one of {valid_vars}")
        
        scaler = getattr(self, f"{variable}_scaler", None)
        if scaler is None:
            logger.warning(f"No scaler available for '{variable}'. Cannot inverse transform.")
            return lambda x: x
        
        logger.info(f"Inverse scaler function for '{variable}' returned.")
        def inverse(x):
          x = np.asarray(x)
          if x.ndim == 1:
            x = x.reshape(-1, 1)
            return scaler.inverse_transform(x).flatten()
          else:
            return scaler.inverse_transform(x)
        
        return inverse
    