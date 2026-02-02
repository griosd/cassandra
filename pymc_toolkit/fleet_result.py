import logging
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM
from typing import Dict, Union, Optional
from pymc_marketing.mmm.evaluation import calculate_metric_distributions

from pymc_toolkit.utils import (plot_roas,
                                    crps_accuracy,
                                    recovery_percentage,
                                    coverage_percentage,
                                    get_variables_to_scale,
                                    plot_posterior_vs_true)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FleetResult:
    """
    Container for storing and analyzing the results of a Media Mix Modeling (MMM)
    run in different contexts: training, production, or recovery.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM model.
    X_train : pd.DataFrame
        Predictor variables used for training.
    y_train : np.ndarray
        Target variable used for training.
    y_pred : np.ndarray
        Posterior predictive samples, shape (n_draws, n_obs).
    X_test : pd.DataFrame, optional
        Predictor variables for out-of-sample prediction.
    y_test : np.ndarray, optional
        Target variable for out-of-sample prediction.
    real_parameters : dict, optional
        Ground-truth parameters used in simulations (for recovery).

    Raises
    ------
    ValueError
        If y_pred dimensions do not match y_train or y_test.
        If X_test is provided without y_test.
    """

    def __init__(
        self,
        mmm: MMM,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        y_pred: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
        real_parameters: Optional[Dict[str, Union[float, list]]] = None):

        logger.info("Initializing FleetResult...")

        self.mmm = mmm
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred = y_pred
        self.X_test = X_test
        self.y_test = y_test
        self.real_parameters = real_parameters
        self.model_variables = self._get_model_variables()

        if self.y_pred.ndim != 2:
            logger.error("y_pred must have shape (n_draws, n_obs).")
            raise ValueError("y_pred must have shape (n_draws, n_obs)")

        if X_test is None:
            self._validate_train_shapes()
            self.y_fit = self.y_pred
            logger.info("No test set detected — using in-sample predictions.")
        else:
            if y_test is None:
                logger.error("X_test was provided but y_test is missing.")
                raise ValueError("When providing X_test, you must also provide y_test.")
            self._validate_test_shapes()
            self._get_y_fit()
            logger.info("Test set detected — in-sample fit computed.")

        self.type = (
            "recovery"
            if self.real_parameters is not None
            else "train"
            if self.y_test is None
            else "production"
        )

    def __repr__(self):
        return f"FleetResult(type='{self.type}')"

    def _validate_train_shapes(self):
        if self.y_pred.shape[1] != len(self.y_train):
            logger.error("y_train length does not match y_pred shape.")
            raise ValueError(
                f"y_train length ({len(self.y_train)}) does not match "
                f"y_pred second dimension ({self.y_pred.shape[1]}).")

    def _validate_test_shapes(self):
        if self.y_pred.shape[1] != len(self.y_test):
            logger.error("y_test length does not match y_pred shape.")
            raise ValueError(
                f"y_test length ({len(self.y_test)}) does not match "
                f"y_pred second dimension ({self.y_pred.shape[1]}).")

    def _get_y_fit(self):
        logger.info("Computing in-sample predictions (y_fit)...")
        y_fcst = self.mmm.sample_posterior_predictive(
            X=self.X_train,
            combined=True,
            progressbar=False,
            var_names=["y"],
            extend_idata=False,
        )
        self.y_fit = np.transpose(np.array(y_fcst["y"]))
        logger.info("In-sample predictions computed.")

    def _plot_predict(self,
          train_window: int = 10,
          hdi_prob: float = 0.9,
          figsize: tuple = (14, 6),
          cumulative: bool = False,
          ax=None):
        """
        Plots time series for test data with posterior predictive samples,
        optionally including the last N points from training.
        Can also display cumulative predictions and observations.
        
        Parameters
        ----------
        train_window (int or None): 
           Number of most recent training points to display. 
           If None, training is not shown. Ignored if cumulative=True.
        hdi_prob (float): 
          Credible interval level (default 0.9).
        figsize (tuple): 
          Figure size (used only if ax is None)
        cumulative (bool):
           If True, plot cumulative sums and ignore train_window.
        ax (matplotlib.axes.Axes, optional):
           Axes object to plot on. If None, a new figure and axes are created.
        
        Returns
        -------
        matplotlib.figure.Figure
        The figure object if a new one was created; otherwise None.
        """
        if cumulative:
            train_window = None
            y_pred = self.y_pred.cumsum(axis=1)
            y_test = np.cumsum(self.y_test)
        else:
            y_pred = self.y_pred
            y_test = self.y_test
        
        y_mean = np.median(y_pred, axis=0)
        lower = np.percentile(y_pred, (1 - hdi_prob) / 2 * 100, axis=0)
        upper = np.percentile(y_pred, (1 + hdi_prob) / 2 * 100, axis=0)
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if train_window is not None:
            y_train = self.y_train[-train_window:]
            start_idx = max(0, len(self.y_train) - train_window)
            
            ax.plot(range(start_idx, start_idx + len(y_train)), 
                y_train, label="Train", color="black", linewidth=2)
            
            ax.axvline(start_idx + len(y_train) - 0.5, color="gray", linestyle="--", alpha=0.7)
            
            test_idx = np.arange(start_idx + len(y_train),
                             start_idx + len(y_train) + len(y_test))
        else:
            test_idx = np.arange(len(y_test))
        
        ax.plot(test_idx, y_test, label="Test (Observed)", color="green", marker="o", linestyle="-")
        ax.plot(test_idx, y_mean, label="Prediction mean", color="royalblue", linewidth=2)
        ax.fill_between(test_idx, lower, upper, color="royalblue", alpha=0.25, label=f"{int(hdi_prob * 100)}% CI")
        
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Target (Cumulative)" if cumulative else "Target")
        ax.set_title("Posterior Predict" + (" (Cumulative)" if cumulative else ""), fontsize=16)
        ax.legend()
        plt.tight_layout()
        
        return fig

    def _get_model_variables(self):
        """
        Retrieve the list of model variables to summarize from the fitted MMM.
        
        Returns
        -------
        list
          List of variable names for posterior summary and plotting.
        """
        mmm = self.mmm
        model_variables = list(mmm.model_config.keys())
        
        for drop in ["likelihood", "gamma_fourier","media_tvp_config","intercept_tvp_config"]:
            if drop in model_variables:
                model_variables.remove(drop)
                model_variables.append("y_sigma")
        
        if mmm.time_varying_media:
            model_variables.extend(["media_temporal_latent_multiplier_raw_eta",
                                "media_temporal_latent_multiplier_raw_ls"])
        
        if mmm.time_varying_intercept:
            model_variables.remove("intercept")
            model_variables.extend(["intercept_temporal_latent_multiplier_raw_eta",
                                "intercept_temporal_latent_multiplier_raw_ls","intercept_baseline"])
        
        return model_variables
        
    def _get_basic_plots(self):
        """
        Generate basic plots shared across recovery and production reports:
             - Traceplot
             - Waterfall decomposition
             - Saturation curves
             - ROAS plot
             - Posterior predictive fit for training data
        
        Returns
        -------
        dict 
            Dictionary with matplotlib figure objects.
        """
        plots = {}
        mmm = self.mmm
        model_variables = self._get_model_variables()
        
        # --- Traceplot ---
        az.plot_trace(mmm.fit_result, var_names=model_variables, compact=True)
        fig = plt.gcf()
        plots["traceplot"] = fig
        plt.close(fig)
        
        # --- Waterfall ---
        fig = mmm.plot_waterfall_components_decomposition()
        plots["waterfall"] = fig
        plt.close(fig)
        
        # --- Saturation curves ---
        fig = mmm.plot_direct_contribution_curves(
            show_fit=True, same_axes=False,
            channels=mmm.channel_columns,
            quantile_lower=0.45, quantile_upper=0.55)
        plots["saturation_curves"] = fig
        plt.close(fig)
        
        # --- ROAS plot ---
        fig = plot_roas(mmm)
        plots["roas"] = fig
        plt.close(fig)
        
        # --- Posterior predictive y_fit ---
        mmm.sample_posterior_predictive(X=self.X_train, progressbar=False, extend_idata=True, var_names=["y"])
        fig = mmm.plot_posterior_predictive(original_scale=True)
        plots["train_fit"] = fig
        plt.close(fig)
        
        return plots
    
    def _get_recovery_plots(self):
        """
        Generate plots comparing posterior distributions to true parameter values
        for recovery studies.
        
        Returns
        -------
        dict
          Dictionary with matplotlib figure objects for recovery parameters.
        """
        plots = {}
        mmm = self.mmm
        real_values = self.real_parameters
        
        # Adstock alpha
        ax = plot_posterior_vs_true(mmm, "adstock_alpha", real_values, hdi_prob=0.95, original_scale=False)
        plots["adstock_alpha"] = ax.get_figure()
        plt.close(ax.get_figure())
        
        # Saturation scale
        vn = "saturation_lam" if mmm.saturation.lookup_name in ["michaelis_menten", "logistic"] else "saturation_kappa"
        ax = plot_posterior_vs_true(mmm, vn, real_values, hdi_prob=0.95, original_scale=False)
        plots["saturation_scale"] = ax.get_figure()
        plt.close(ax.get_figure())
        
        # Saturation ROI beta
        vn = "saturation_beta" if mmm.saturation.lookup_name in ["hill", "logistic"] else "saturation_alpha"
        ax = plot_posterior_vs_true(mmm, vn, real_values, hdi_prob=0.95, original_scale=True)
        plots["saturation_beta"] = ax.get_figure()
        plt.close(ax.get_figure())
        
        # Control variables
        if mmm.control_columns is not None:
            ax = plot_posterior_vs_true(mmm, "gamma_control", real_values, hdi_prob=0.95, original_scale=True)
            plots["controls"] = ax.get_figure()
            plt.close(ax.get_figure())
        
        # Intercept and y_sigma side-by-side
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        plot_posterior_vs_true(mmm, "intercept", real_values, ax=axs[0])
        plot_posterior_vs_true(mmm, "y_sigma", real_values, ax=axs[1])
        plots["intercept_sigma"] = fig
        plt.close(fig)
        
        return plots
    
    def _get_prediction_plots(self):
        """
        Generate prediction plots for production use:
          - Test fit
          - Cumulative test fit
        
        Returns
        -------
        dict
          Dictionary with matplotlib figure objects for production predictions.
        """
        plots = {}

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        self._plot_predict(train_window=20, ax=axs[0])
        self._plot_predict(cumulative=True, ax=axs[1])
        plots["test_predictions"] = fig
        plt.close(fig)
        
        return plots
    
    def _get_fleet_elements(self):
        """
        Compute all elements needed for the production or recovery report.
        
        Returns
        -------
        dict
        Dictionary containing:
            - "df_train": pd.DataFrame with training metrics
            - "df_mcmc": pd.DataFrame with MCMC diagnostics
            - "param_summary_df": pd.DataFrame with posterior parameter summaries
            - "plots": dict of matplotlib figure objects
            - "df_test": pd.DataFrame with test metrics (if type='production')
            - "df_rec": pd.DataFrame with recovery diagnostics (if type='recovery')
        """
        mmm = self.mmm
        plots = self._get_basic_plots()
        
        # Conditional plots
        if self.type == "recovery":
            plots.update(self._get_recovery_plots())
        elif self.type == "production":
            plots.update(self._get_prediction_plots())
        # DataFrames
        df_train = pd.DataFrame(self.get_model_accuracy().items(), columns=["Metric", "Value"])
        
        df_mcmc = pd.DataFrame(self.get_mcmc_diagnostics())
        df_mcmc.insert(0, 'chain', range(1, len(df_mcmc) + 1))
        
        model_variables = self._get_model_variables()
        param_summary_df = az.summary(data=mmm.fit_result, hdi_prob=0.95, var_names=model_variables)
        
        results = {
            "df_train": df_train,
            "df_mcmc": df_mcmc,
            "param_summary_df": param_summary_df,
            "plots": plots
        }
        
        if self.type == "production":
            results['df_test'] = pd.DataFrame(
                self.get_model_accuracy(train=False).items(),
                columns=["Metric", "Value"])
        if self.type == "recovery":
            results['df_rec'] = pd.DataFrame(
                self.get_recovery_diagnostics().items(),
                columns=["Parameter", "Value"])
        
        return results

    def get_model_accuracy(self, train: bool = True, cumulative: bool = False) -> pd.Series:
        """
        Compute accuracy metrics (MAPE, NRMSE, R², CRPS) for training or test data.

        Parameters
        ----------
        train : bool, default True
            If True, use training data; otherwise use test data.
        cumulative (bool):
          If True, plot cumulative sums and ignore train_window.

        Returns
        -------
        pd.Series
            Computed metrics.
        """
        dataset_type = "train" if train else "test"
        logger.info(f"Calculating {dataset_type} metrics...")
        y_true = self.y_train if train else self.y_test
        y_fcst = self.y_fit if train else self.y_pred

        y_true = np.cumsum(y_true) if cumulative else y_true
        y_fcst = np.cumsum(y_fcst,axis=1) if cumulative else y_fcst

        metrics_dict = calculate_metric_distributions(
            y_true=y_true,
            y_pred=y_fcst.T,
            metrics_to_calculate=["mape", "nrmse", "r_squared"],
        )
        metrics = pd.DataFrame.from_dict(metrics_dict).median(axis=0)
        metrics["crps_error"] = crps_accuracy(target=y_true, predict=y_fcst)

        if cumulative:
            metrics.index = metrics.index.map(lambda i: f"cumulative_{i}")

        logger.info(f"{dataset_type.capitalize()} metrics calculated.")
        return metrics.to_dict()

    def get_mcmc_diagnostics(self) -> dict:
        """
        Return MCMC diagnostics: divergences count and max tree depth.

        Returns
        -------
        dict
            MCMC diagnostics.
        """
        logger.info("Retrieving MCMC diagnostics...")
        diagnostics = {
            "divergences": self.mmm.idata["sample_stats"]["diverging"]
            .sum(axis=1)
            .to_numpy().tolist(),
            "Max_treedepth": self.mmm.idata["sample_stats"]["tree_depth"]
            .max(axis=1)
            .to_numpy().tolist(),
        }
        logger.info("MCMC diagnostics retrieved.")
        return diagnostics

    def get_recovery_diagnostics(self, use_recovery: bool = True, hdi_prob: float = 0.9) -> dict:
        """
        Generate a recovery report comparing true parameters to posterior estimates.

        Parameters
        ----------
        use_recovery : bool, default True
            If True, compute recovery percentage; if False, compute HDI coverage.
        hdi_prob : float, default 0.9
            Probability for the HDI interval.

        Returns
        -------
        dict
            Recovery report.
        """
        if self.type != "recovery":
            logger.error("FleetResult is not of type 'recovery'.")
            raise ValueError(f"{self.type}-FleetResult is not a recovery fleet.")

        logger.info("Calculating recovery diagnostics...")
        var_config = get_variables_to_scale(self.real_parameters)
        if self.mmm.control_columns is not None:
            var_config["gamma_control"] = True

        report = {}
        for var_name, scale in var_config.items():
            key = var_name if var_name != "gamma_control" else "controls"
            if use_recovery:
                value = recovery_percentage(
                    mmm=self.mmm,
                    var_name=var_name,
                    real_parameters=self.real_parameters,
                    original_scale=scale,
                    averaged=True,
                )
            else:
                value = coverage_percentage(
                    mmm=self.mmm,
                    var_name=var_name,
                    real_parameters=self.real_parameters,
                    original_scale=scale,
                    hdi_prob=hdi_prob,
                )
            report[key] = value

        logger.info("Recovery diagnostics calculated.")
        return report

    def summary(self, cumulative: bool = False) -> dict:
        """
        Return a summary dictionary with key metrics and diagnostics.

        Returns
        -------
        dict
            Summary metrics and diagnostics.
        """
        logger.info("Generating diagnostics summary...")
        mcmc = self.get_mcmc_diagnostics()
    
        diagnostics = {
            "divergences": int(np.sum(mcmc["divergences"])),
            "max_tree_depth": int(np.max(mcmc["Max_treedepth"])),
        }
        diagnostics["train"] = self.get_model_accuracy(train=True)

        if self.type == "production":
            diagnostics["test"] = self.get_model_accuracy(train=False,
                                                    cumulative=cumulative)

        if self.type == "recovery":
            diagnostics["recovery"] = self.get_recovery_diagnostics()

        logger.info("Diagnostics summary generated.")
        return diagnostics
    
    def generate_report(self, output_html="fleet_report.html", output_dir=None):
        """
        Generates an HTML report for the fleet, embedding matplotlib figures as base64 images.
        Calls `compute_report_elements` internally to gather all necessary data.
        
        Parameters
        ----------
        output_html : str
           Name of the HTML file to generate.
        output_dir : str or None
          Directory where the HTML file will be saved. Defaults to current working directory.
        
        Returns
        -------
        str
          Absolute path to the generated HTML file.
        """
        import os
        import base64
        from io import BytesIO
        
        # Gather all elements
        elements = self._get_fleet_elements()
        
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
        
        # Convert plots to base64
        plots_base64 = {k: fig_to_base64(fig) for k, fig in elements.get("plots", {}).items()}
        
        # Report title depends on fleet type
        report_title = "Production Fleet Report" if self.type == "production" else "Recovery Fleet Report"
        html_sections = [f"<h1>{report_title}</h1>"]
        
        # Include relevant DataFrames
        if "df_test" in elements:
            html_sections.append("<h2>Test Metrics</h2>")
            html_sections.append(elements["df_test"].to_html(index=False, float_format="%.4f"))
        
        if "df_rec" in elements:
            html_sections.append("<h2>Recovery Diagnostics</h2>")
            html_sections.append(elements["df_rec"].to_html(index=False, float_format="%.4f"))
        
        ## Train summary
        html_sections.append("<h2>Training Metrics</h2>")
        html_sections.append(elements["df_train"].to_html(index=False, float_format="%.4f"))
        
        # mcmc diagnostic
        html_sections.append("<h2>MCMC Diagnostics</h2>")
        html_sections.append(elements["df_mcmc"].to_html(index=False))
        
        # posterior summary
        html_sections.append("<h2>Parameter Summary</h2>")
        html_sections.append(elements["param_summary_df"].to_html())
        
        # Titles for plots
        plot_titles = {
            "traceplot": "Traceplots",
            "adstock_alpha": "Adstock Parameter Recovery",
            "saturation_scale": "Saturation Scale Recovery",
            "saturation_beta": "Saturation Beta Recovery",
            "controls": "Control Parameter Recovery",
            "intercept_sigma": "Intercept & Sigma Recovery",
            "waterfall": "Marginal Contribution Plots",
            "saturation_curves": "Direct Contribution Curves",
            "roas": "Average ROAs per channel",
            "train_fit": "Posterior Predictive Fit (Train)",
            "test_predictions": "Posterior Predictive Fit (Test)",
        }
        # Insert plots that exist
        for key, title in plot_titles.items():
            if key in plots_base64:
                html_sections.append(f"<h3>{title}</h3>")
                html_sections.append(f'<img src="{plots_base64[key]}" width="900">')
        
        # Save HTML
        output_path = os.path.join(output_dir, output_html)
        with open(output_path, "w") as f:
            f.write("\n".join(html_sections))
        
        return os.path.abspath(output_path)