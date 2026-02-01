import logging
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pymc_marketing.metrics import crps
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def simulate_spend_channels(dates:pd.DatetimeIndex,
                            channel_names:list, 
                            seed:int = None) -> pd.DataFrame:
    """
    Generates random spending patterns with on/off behavior across channels.

    Args:
        dates (list or pd.DatetimeIndex): List of time steps.
        channel_names (list of str): Names of media channels.
        seed int or None, optional
        Random seed for reproducibility. If None, no seed is passed (non-deterministic).

    Returns:
        pd.DataFrame: Simulated spend data indexed by date and channel.
    """
    n_rows = len(dates)
    n_cols = len(channel_names)

    # Generate full matrix of random values
    rng = np.random.default_rng(seed)
    data_matrix = np.abs(rng.normal(loc=0, scale=1, size=(n_rows, n_cols)))

    # Build DataFrame directly
    spends = pd.DataFrame(data_matrix, columns=channel_names)
    spends["ds"] = pd.to_datetime(dates)
    
    return spends

def crps_accuracy(target, predict):
    """
    Compute the accumulated of the average continuous ranked probability score.
    The lower the better.
    
    Args:
      target (array-like): The ground truth values.
      predict (array-like): The predicted values. It is expected that y_pred has one extra sample
        dimension on the left.
    
     Returns
    -------
    float
        The CRPS value as a (possibly weighted) average of the per-observation CRPS values.

    References
    ----------
    - This implementation is a minimal adaptation from the one in the Pyro project: 
    https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html#crps_empirical
    """
    crps_error = crps(target, predict) / target.mean()
    
    return float(crps_error)

def compute_coverage(posterior, true_value, hdi_prob=0.9):
    hdi_bounds = az.hdi(posterior, hdi_prob=hdi_prob, skipna=True)  # shape (n_dims, 2)
    lower = hdi_bounds[:, 0]
    upper = hdi_bounds[:, 1]

    # Verificamos si el valor real cae dentro del HDI
    in_interval = (true_value >= lower) & (true_value <= upper)
    coverage = np.mean(in_interval) 
    return float(coverage)

def recovery_percentage(mmm, var_name, real_parameters, original_scale=False, averaged=True):
    """
    Compute the percentage CRPS accuracy for a set of sampled parameters.

    The CRPS accuracy is defined as:  
        100 * |1 - CRPS(posterior, true_value) / true_value|

    Args:
        mmm: Fitted MMM model with `.idata` (e.g., PyMC InferenceData).
        var_name (str): Name of the variable to evaluate (e.g. "saturation_beta").
        real_parameters (dict): Dictionary of true parameter values. Must include `var_name`.

    Returns:
        np.ndarray: Array of CRPS accuracy percentages for each parameter (1 per dimension).

    References:
        Adapted from Pyro: 
        https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html#crps_empirical
    """
    true_value = np.atleast_1d(np.array(real_parameters[var_name]))  # Shape (n_dims,)
    posterior = get_posterior_array(mmm = mmm, var_name = var_name)

    if original_scale: 
        scaler = mmm.channel_transformer if var_name=="saturation_beta" else mmm.target_transformer
        posterior = scaler.inverse_transform(posterior)

    # Now safe to call crps
    crps_error = crps(true_value, posterior)
    perccentage_crps_error = np.abs(1 - crps_error / true_value)

    return np.median(perccentage_crps_error).tolist() if averaged else perccentage_crps_error.tolist()

def coverage_percentage(mmm, var_name, real_parameters, original_scale=False, hdi_prob=0.9):
    """
    Compute the percentage CRPS accuracy for a set of sampled parameters.

    The CRPS accuracy is defined as:  
        100 * |1 - CRPS(posterior, true_value) / true_value|

    Args:
        mmm: Fitted MMM model with `.idata` (e.g., PyMC InferenceData).
        var_name (str): Name of the variable to evaluate (e.g. "saturation_beta").
        real_parameters (dict): Dictionary of true parameter values. Must include `var_name`.
        original_scale (Bool): a Bool value, computes coverage using parameters real scale.
        hd_prob (float): probability in the credible intervals.

    Returns:
        np.ndarray: Array of CRPS accuracy percentages for each parameter (1 per dimension).

    References:
        Adapted from Pyro: 
        https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html#crps_empirical
    """
    true_value = np.atleast_1d(np.array(real_parameters[var_name]))
    posterior = get_posterior_array(mmm = mmm, var_name = var_name)

    if original_scale: 
        scaler = mmm.channel_transformer if var_name=="saturation_beta" else mmm.target_transformer
        posterior = scaler.inverse_transform(posterior)

    # Now safe to call crps
    return compute_coverage(posterior, true_value, hdi_prob)

def get_posterior_array(mmm, var_name, ensure_chain_draw=True):
    """
    Extracts posterior samples from a model and ensures correct shape for ArviZ compatibility.

    Parameters
    ----------
    mmm : object
        An object containing the InferenceData (typically with `.idata` attribute).
    var_name : str
        Name of the variable to extract from the posterior.
    ensure_chain_draw : bool, optional (default=True)
        If True, reshapes the posterior to have at least 3 dimensions
        in the format (chain, draw, variable_dim), which is the expected input for ArviZ.

    Returns
    -------
    posterior : np.ndarray
        Numpy array of posterior samples with shape suitable for use in ArviZ functions like `az.hdi`.
    """
    posterior = az.extract(mmm.idata, var_names=var_name).values
    posterior = np.atleast_2d(posterior)  # Ensure shape (n_draws, n_dims)
    
    if posterior.shape[0] < posterior.shape[1]:
        posterior = posterior.T  # Ensure shape is (n_draws, n_dims)

    return posterior

def recovery_summary(mmm, real_parameters, hdi_prob=0.9):
    """
    Generate a recovery report comparing true parameters to their posterior estimates.

    This function computes:
    - CRPS-based recovery accuracy (as percentage)
    - HDI-based coverage accuracy (as percentage)

    The results are returned as a dictionary mapping each parameter to a list:
    [recovery, coverage]

    Parameters:
        mmm (MMM): A fitted PyMC-Marketing MMM model.
        real_parameters (dict): Dictionary of true parameter values used in simulation.
        hdi_prob (float): HDI probability for coverage (default=0.9).

    Returns:
        dict: Dictionary with structure:
              {
                  "saturation_beta": [recovery%, coverage%],
                  "y_sigma": [recovery%, coverage%],
                  ...
                  "y_fit_train": <crps accuracy>
              }
    """
    var_config = get_variables_to_scale(real_parameters)

    if mmm.control_columns is not None:
        var_config["gamma_control"] = True

    report = {}

    for var_name, scale in var_config.items():
        key = var_name if var_name != "gamma_control" else "controls"

        recovery = recovery_percentage(
            mmm=mmm,
            var_name=var_name,
            real_parameters=real_parameters,
            original_scale=scale,
            averaged=True
        )

        coverage = coverage_percentage(
            mmm=mmm,
            var_name=var_name,
            real_parameters=real_parameters,
            original_scale=scale,
            hdi_prob=hdi_prob
        )

        report[key] = [recovery, coverage]

    # Add CRPS accuracy from in-sample predictions
    y_fcst = mmm.sample_posterior_predictive(
          X = mmm.X,
          extend_idata=False,
          var_names=["y"]
    )
    y_pred = np.transpose(np.array(y_fcst["y"]))
    crps_score = crps_accuracy(mmm.y, y_pred).tolist()
    y_coverage = compute_coverage(posterior=y_pred,
                                  true_value=mmm.y,
                                  hdi_prob=hdi_prob)
    
    report["y_fit_train"] = [crps_score/100, y_coverage]

    return report

def get_variables_to_scale(real_parameters):
    """
    Determine which variables in a parameter dictionary should be scaled.

    This function checks each key in the input dictionary and returns 
    a new dictionary indicating whether each variable should be scaled
    based on a predefined list.

    Parameters
    ----------
    real_parameters : dict
        Dictionary of parameter names (as keys) and their corresponding values.

    Returns
    -------
    dict
        Dictionary where each key from `real_parameters` is mapped to a boolean:
        - True if the variable should be scaled
        - False otherwise
    """
    scaled_variables = ["intercept", "gamma_control", "saturation_alpha", "y_sigma", "saturation_beta"]
    
    var_config = {
        key: key in scaled_variables
        for key in real_parameters.keys()
    }

    return var_config

def rolling_split(
        X: pd.DataFrame,
        y: np.ndarray,
        n_test: int = 4,
        n_splits: int = 1) -> List[Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame], Optional[np.ndarray]]]:
    """
    Generate rolling or expanding train-test splits for time series/sequential data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    y : np.ndarray
        Target array.
    n_test : int, default=4
        Number of observations to include in each test set.
        If 0, returns a single split with full training data and no test set.
    n_splits : int, default=1
        Number of rolling splits to generate.

    Returns
    -------
    splits : list of tuples
        Each tuple contains:
        (X_train, y_train, X_test, y_test)
        where X_test and y_test may be None if n_test == 0.

    Raises
    ------
    ValueError
        If n_test < 0 or n_splits <= 0.
        If n_test is too large for the dataset size.
        If n_splits and n_test combination results in no training samples.
    """
    if n_test < 0:
        logger.error("n_test is lower than 0 split can not be performed.")
        raise ValueError("n_test is lower than 0 split can not be performed.")

    if n_splits <= 0:
        logger.error("n_splits must be positive.")
        raise ValueError("n_splits must be positive.")

    n_total = X.shape[0]

    if n_test == 0:
        logger.info("n_test = 0, returning single split with full data as train and None for test.")
        return [(X.copy(), y.copy(), None, None)]

    n_init = n_total - n_test * n_splits
    
    if n_init <= 0:
        logger.error("Invalid parameters: n_init calculated <= 0. Reduce n_splits or n_test.")
        raise ValueError("Invalid parameters: n_init calculated <= 0. Reduce n_splits or n_test.")
    
    if n_splits == 1:
        logger.info(f"Total samples: {n_total}")
        logger.info(f"Single split: train size = {n_init}, test size = {n_test}")
        return [(
                X.iloc[:n_init],
                y[:n_init],
                X.iloc[n_init:],
                y[n_init:],
            )]

    logger.info(f"Total samples: {n_total}")
    logger.info(f"Initial train size: {n_init}, test size: {n_test}, splits: {n_splits}")

    splits: List[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]] = []
    for i in range(n_splits):
        start_train = i * n_test
        end_train = n_init + i * n_test
        start_test = end_train
        end_test = end_train + n_test

        logger.info(f"Split {i + 1}")
        logger.info(f"{end_train - start_train} train samples")
        logger.info(f"{end_test - start_test} test samples")

        X_train = X.iloc[start_train:end_train]
        X_test = X.iloc[start_test:end_test]
        y_train = y[start_train:end_train]
        y_test = y[start_test:end_test]

        splits.append((X_train, y_train, X_test, y_test))

    return splits

def get_all_zero_columns(data,columns:Optional[List[str]]=None):
    """
    Identify columns where all values are zero.
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to check.
    columns : list of str, optional
        Subset of columns to check. If None, all columns are checked.
    
    Returns
    -------
    list
        List of column names where all values are zero.
    """
    # If no subset provided, check all columns
    if columns is None:
        columns = data.columns

    zero_cols = data[columns].columns[(data[columns] == 0).all()]
    return list(zero_cols)

def get_media_with_negatives(data:pd.DataFrame, media:Optional[List[str]]=None):
    """
    Identify columns that contain at least one negative value.
    
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to check.
    media : list of str, optional
        Subset of columns to check. If None, all columns are checked.
    
    Returns
    -------
    list
        List of column names containing negative values.
    """
    # If no subset provided, check all columns
    if media is None:
        media = data.columns
    
    negative_media = data[media].columns[(data[media] < 0).any()]
    return list(negative_media)

def filter_by_dates(data:pd.DataFrame, date_name:str, start_date:str=None, end_date:str=None):
    """
    Filter a DataFrame between two dates (inclusive).
    
    Rules:
      - If start_date or end_date is None, it defaults to the min/max date in the column.
      - If start_date < min(date_col), it is clipped to min(date_col) with a warning.
      - If end_date > max(date_col), it is clipped to max(date_col) with a warning.
      - If start_date >= end_date after adjustment, a warning is logged
        and the original DataFrame is returned.
    
    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame.
    date_col : str
        Column name containing dates.
    start_date : str | pd.Timestamp | None
        Start date (inclusive). If None, defaults to min(date_col).
    end_date : str | pd.Timestamp | None
        End date (inclusive). If None, defaults to max(date_col).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame or original DataFrame if dates are invalid.
    """
    df = data.copy()
    df[date_name] = pd.to_datetime(df[date_name])

    min_date, max_date = df[date_name].min(), df[date_name].max()

    # Handle None values
    start_date = pd.to_datetime(start_date) if start_date is not None else min_date
    end_date = pd.to_datetime(end_date) if end_date is not None else max_date

    # Clip out-of-range values
    if start_date < min_date:
        logging.warning(f"start_date {start_date} is before min({min_date}); clipping to min_date.")
        start_date = min_date
    if end_date > max_date:
        logging.warning(f"end_date {end_date} is after max({max_date}); clipping to max_date.")
        end_date = max_date

    # Validate
    if start_date >= end_date:
        logging.warning("start_date must be earlier than end_date. Returning original DataFrame.")
        return df

    # Apply filter
    mask = df[date_name].between(start_date, end_date)
    return df.loc[mask]

###############################################################################################
#              Plot utilities 
###############################################################################################

def plot_roas(mmm):
    """
    Computes ROAS per channel from a Media Mix Model (MMM) object,
    generates ArviZ diagnostic plots (forest and posterior), 
    and creates a horizontal bar plot of the average ROAS per channel.

    Parameters
    ----------
    mmm : object
        pymc_marketing Media Mix Model 

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the horizontal bar plot of average ROAS per channel.
    """
    
    # --- Compute ROAS samples ---
    channel_contribution_original_scale = mmm.compute_channel_contribution_original_scale()
    spend_sum = mmm.X.loc[:, mmm.channel_columns].sum().to_numpy()
    
    roas_samples = channel_contribution_original_scale.sum(dim="date") / spend_sum[np.newaxis, np.newaxis, :]
        
    # --- Horizontal bar plot of average ROAS ---
    roas_mean = roas_samples.mean(dim=['chain', 'draw'])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.arange(len(roas_mean['channel'])))
    ax.barh(roas_mean['channel'].values, roas_mean.values, color=colors)
    ax.set_xlabel("ROAS")
    ax.set_ylabel("Channels")
    ax.set_title("Average ROAS per Channel")
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_posterior_vs_true(mmm, var_name, real_parameters, hdi_prob = 0.95, original_scale = True, ax = None):
    """
    Plot boxplots of posterior samples for each parameter with the true value overlaid.

    Parameters:
    - mmm: MMM object with `.fit_result` (InferenceData).
    - var_name: str, name of the variable to compare.
    - real_parameters: list or array of true parameter values (same length as num parameters).
    - hdi_prob: Plots highest posterior density interval for chosen percentage of density.
    - ax: Optional matplotlib Axes object for subplotting.
    """
    if ax is None:
        ax = plt.gca()
    
    parameter = real_parameters[var_name]
    n_params = parameter.size
    
    true_value = np.atleast_1d(np.array(real_parameters[var_name]))  # Shape (n_dims,)
    posterior = get_posterior_array(mmm = mmm, var_name = var_name)

    if original_scale: 
        scaler = mmm.channel_transformer if var_name=="saturation_beta" else mmm.target_transformer
        posterior = scaler.inverse_transform(posterior)
    
    prob = (1 - hdi_prob)/2
    
    lower1 = np.quantile(posterior, prob, axis=0)
    upper1 = np.quantile(posterior, 1 - prob, axis=0)
    
    lower2 = np.quantile(posterior, 0.25, axis=0)
    upper2 = np.quantile(posterior, 0.75, axis=0)
    medians = np.median(posterior, axis=0)

    y_pos = np.arange(n_params)

    # HDI intervals
    ax.hlines(y=y_pos, xmin=lower1, xmax=upper1, color="C0", alpha = 1.0, lw=1, label="90% HDI")
    ax.hlines(y=y_pos, xmin=lower2, xmax=upper2, color="C0", alpha = 1.0, lw=4, label="75% HDI")
    ax.plot(medians, y_pos, "o", color="C0", label="Posterior median")
    ax.plot(true_value, y_pos, "o", mfc="none", mec="black", mew=1.5, label="True value")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{var_name}[{i}]" for i in range(n_params)])
    ax.set_title(f"{hdi_prob*100}% True value recovery plot: {var_name}")
    ax.invert_yaxis()

    return ax