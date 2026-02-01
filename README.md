# PyMC experiments

A repository for track MMM experiments using [PyMC-marketing](https://www.pymc-marketing.io/en/latest/index.html) toolbox. The repository is divided in four main folders:

-   data: Stores the necessary data to run experiments.

-   discoveries: high impact scripts ready for production. These scripts follow [developers structure guide](https://www.notion.so/cassandra-mmm/MMM-Discovery-Documentation-Workflow-for-Data-Scientists-1e37dc4be0fc80498618ea0dbce0c0c8?source=copy_link)

-   `pymc_toolkit`: the folder has a package structure and is an API to configure Clients, data, PymMC models, and run fleets (`production`, `forecast`, `recovery`, and `stability`),

-   test: unit tests for `pymc_toolkit`.

Additional files such as `.gitignore`, and `requirements.txt` are necessary files for a correct product delivery and authorization.

-   `requirements.txt`: has the necessary package dependencies for running `pymc-experiments`.

-   `requirements-development.txt`: dependencies for development `pymc-toolkit`.

## Install `pymc_toolkit`

`pymc_toolkit` behaves as a standard Python package, and its dependencies must be installed for it to work correctly. Follow these steps:

1.  Install a suitable Python version (3.11 or higher). We recommend [pyenv](https://github.com/pyenv/pyenv) to manage different Python versions across your applications. For macOS, follow [these steps](https://github.com/pyenv/pyenv?tab=readme-ov-file#macos). We also recommend to use python version `3.11.5`

2.  Create a virtual environment named `.env` using `venv`:

``` bash
python -m venv .env
```

3.  Activate your virtual environment:

``` bash
source .env/bin/activate
```

4.  Install the requirements file:

``` bash
python -m pip install -r requirements.txt
```

🎉 Now `pymc_toolkit` is ready to use!

## User Guide

Using `pymc_toolkit` is as simple as using `pymc_marketing` as it is a simple wrapper of the original package plus some extra push to make it more spicy. The following example analyses the `MS.csv` data set to explain the repo's functionality.

First let's call some dependencies to load the data and create a model

``` python
import pandas as pd
from pymc_toolkit.pymc_model import PymcModel
```

Now let's load the data, which is a weekly dataset with 161 weeks, with 13 media channels, 2 control variables, and the target variable (`ecommerce_revenue`).

``` python
data = pd.read_csv('data/ms.csv')
>>> data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 161 entries, 0 to 160
Data columns (total 17 columns):
 #   Column                             Non-Null Count  Dtype  
---  ------                             --------------  -----  
 0   date_week                          161 non-null    object 
 1   ecommerce_revenue                  161 non-null    float64
 2   snapchat_cost                      161 non-null    float64
 3   emails_delivered                   161 non-null    int64  
 4   facebook_ads_conversions_cost      161 non-null    float64
 5   facebook_ads_brand_awareness_cost  161 non-null    float64
 6   facebook_ads_engagement_cost       161 non-null    float64
 7   google_ads_video_cost              161 non-null    float64
 8   google_ads_shopping_cost           161 non-null    float64
 9   google_ads_search_men_cost         161 non-null    float64
 10  google_ads_search_kids_cost        161 non-null    float64
 11  google_ads_display_only_cost       161 non-null    float64
 12  google_ads_search_brand_cost       161 non-null    float64
 13  google_ads_search_women_cost       161 non-null    float64
 14  google_ads_search_others_cost      161 non-null    float64
 15  youtube_organic_views              161 non-null    int64  
 16  events                             161 non-null    int64  
dtypes: float64(13), int64(3), object(1)
memory usage: 21.5+ KB
```

Let's separate the columns into control, media, and target categories.

``` python
target_name = 'ecommerce_revenue'

channel_names = ['snapchat_cost',
       'facebook_ads_conversions_cost', 'facebook_ads_brand_awareness_cost',
       'facebook_ads_engagement_cost', 'google_ads_video_cost',
       'google_ads_shopping_cost', 'google_ads_search_men_cost',
       'google_ads_search_kids_cost', 'google_ads_display_only_cost',
       'google_ads_search_brand_cost', 'google_ads_search_women_cost',
       'google_ads_search_others_cost', 'youtube_organic_views']

control_names = ['events', 'emails_delivered']
```

Let's create a model using a Time-varying media with Cassandras custom priors

``` python
>>>model = PymcModel(client_data=data,
             target_name=target_name,
             date_column='date_week',
             channel_names=channel_names,
             control_names=control_names,
             lag_max=4,
             scale_data=True,
             saturation='michaelis_menten',
             time_varying_media=True)
INFO:pymc_toolkit.pymc_model:Creating the client's data configuration.
INFO:pymc_toolkit.client_config:Target variable scaled using MaxAbsScaler.
INFO:pymc_toolkit.client_config:Channel variables scaled using MaxAbsScaler.
INFO:pymc_toolkit.client_config:Control variables scaled using MaxAbsScaler.
INFO:pymc_toolkit.client_config:Client data loaded for 'client0' with 161 rows.
INFO:pymc_toolkit.pymc_model:Set up PyMCModel's basic configuration.
INFO:pymc_toolkit.pymc_model:Define PymcModel's default priors.
INFO:pymc_toolkit.pymc_model:Creating default priors for intercept and model's scale (y_sigma).
INFO:pymc_toolkit.pymc_model:Creating default priors for gamma control.
INFO:pymc_toolkit.pymc_model:Using Geometric Adstock.
INFO:pymc_toolkit.pymc_model:Creating default priors for adstock alpha.
INFO:pymc_toolkit.pymc_model:Using Michaelis-Menten Saturation.
INFO:pymc_toolkit.pymc_model:Creating default priors for saturation lambda and alpha.
INFO:pymc_toolkit.pymc_model:Time-varying Media MMM.
INFO:pymc_toolkit.pymc_model:Creating default priors for Kernel's eta and length-scale.
```

This will create a simple object that will store all the neccesary arguments, to deploy this model in production plus adding complete logging for a faster model debugging and process tracking.

An important step in Bayesian modeling is prior configuration, we can access to cassandra's default priors, by calling the argument `model_priors`.

``` python
model.model_priors
{'intercept': Prior("HalfNormal", sigma=1),
 'likelihood': Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
 'gamma_control': Prior("HalfNormal", sigma=1, dims="control"),
 'media_tvp_config': HSGPKwargs(m=50, L=241.5, eta_lam=10.0, ls_mu=5, ls_sigma=5, cov_func='Matern52'),
 'saturation_lam': Prior("HalfNormal", sigma=1, dims="channel"),
 'saturation_alpha': Prior("Gamma", mu=2, sigma=1, dims="channel"),
 'adstock_alpha': Prior("Beta", alpha=1, beta=3, dims="channel")
 }
```

Fitting the model is as simple as in `pymc`, by calling the `fit` method. This will create a temporary `pymc_marketing` object, extract the important data from our PymcModel, train the model using the temporary object, and stores the fit in our model class. Additionally, will produce predictions for our training dataset for a correct model evaluation.

``` python
>>>  model.fit()
INFO:pymc_toolkit.client_config:Returning channels in scaled form.
INFO:pymc_toolkit.client_config:Returning controls in scaled form.
INFO:pymc_toolkit.client_config:Covariates DataFrame constructed with shape (161, 16).
INFO:pymc_toolkit.client_config:Returning target variable in original scale.
INFO:pymc_toolkit.client_config:Inverse scaler function for 'target' returned.
INFO:pymc_toolkit.pymc_model:Sampling client0's MMM using 4 chains, 1000 draws, and 1000 tuning steps.
Initializing NUTS using adapt_diag...
INFO:pymc.sampling.mcmc:Initializing NUTS using adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
INFO:pymc.sampling.mcmc:Multiprocess sampling (4 chains in 4 jobs)
NUTS: [intercept, adstock_alpha, saturation_alpha, saturation_lam, media_temporal_latent_multiplier_raw_eta, media_temporal_latent_multiplier_raw_ls, media_temporal_latent_multiplier_raw_hsgp_coefs_offset, gamma_control, y_sigma]
INFO:pymc.sampling.mcmc:NUTS: [intercept, adstock_alpha, saturation_alpha, saturation_lam, media_temporal_latent_multiplier_raw_eta, media_temporal_latent_multiplier_raw_ls, media_temporal_latent_multiplier_raw_hsgp_coefs_offset, gamma_control, y_sigma]
                                                                                                                                  
  Progress                                   Draws   Divergences   Step size   Grad evals   Sampling Speed   Elapsed   Remaining  
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2000    3             0.01        255          20.13 draws/s    0:01:39   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2000    1             0.02        255          20.62 draws/s    0:01:36   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2000    1             0.01        255          18.73 draws/s    0:01:46   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   2000    1             0.01        255          18.96 draws/s    0:01:45   0:00:00    
                                                                                                                                  

Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 107 seconds.
INFO:pymc.sampling.mcmc:Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 107 seconds.
There were 6 divergences after tuning. Increase `target_accept` or reparameterize.
ERROR:pymc.stats.convergence:There were 6 divergences after tuning. Increase `target_accept` or reparameterize.
INFO:root:Compute y_fit using the model's train data
Sampling: [y]
INFO:pymc.sampling.forward:Sampling: [y]
Sampling ... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 / 0:00:00

INFO:pymc_toolkit.pymc_model:Sampling completed.
```

Lets get the intercept's estimate with confidence intervals. Simply use the `sumarize_variable` method.

``` python
model.summarize_variable(var_name="intercept")
INFO:pymc_toolkit.client_config:Inverse scaler function for 'target' returned.
{'variable': 'intercept',
 'coef': 3162.75538405655,
 'ci_up_cassandra': 9250.538075094692,
 'ci_low_cassandra': 193.30986164526558}
```

It notifies that this parameters needs to be inverse scaled, using the target's scale.

Similar procedures apply for `predict()`, and `refresh()`

**Important:** The model logs are going to be omitted to make the report easier to read.

## Fleets

Models in production need more complex procedures than just train (`fit`), and predict. It needs a proper model evaluation, monitoring, calibration, recovery and stability assessment. All these procedures are implemented in `pymc_toolkit`

### Recovery fleet.

This process is vital in the first steps of model building, and it tells how credible is our code to deploy the actual model using your data. The process is:

1.  Simulate the model parameters using your covariates.

2.  Create a fake target using the model and simulated parameters.

3.  train the model using the fake target and covariates.

The following code shows how to perform a recovery. Fleets have their own class to store its results for proper data visualization and summary.

``` python
>>> rleet = model.recovery_fleet()
>>> rfleet.summary()
{'divergences': 1,
 'max_tree_depth': 5,
 'train': {'mape': 1.3626885243388833,
  'nrmse': 0.20562748699002514,
  'r_squared': 0.5280365382698324,
  'crps': 0.7157267643403227},

'recovery': {
  'y_sigma': 0.9812822944861193,
  'intercept': 0.40170085644414244,
  'controls': 0.556468919816216,
  'media_temporal_latent_multiplier_raw_eta': 0.6613063909784102,
  'media_temporal_latent_multiplier_raw_ls': 0.5561216804044913,
  'saturation_lam': 0.5733667318143025,
  'saturation_alpha': 0.3830523755455959,
  'adstock_alpha': 0.7680047722035878}
}
```

### production fleet

These fleets are the deplyment's MVP, this procedure is vital for monitoring, evaluation, and product presentation, the procedure is simple but effective.

1.  Refresh data,

2.  split the data leaving out the last month.

3.  train the model

4.  show the models result for the last month.

The code is simply done by:

``` python
>>>pfleet = model.production_fleet()
>>>pfleet.summary()
{'divergences': 4,
 'max_tree_depth': 9,
 'train': {
   'mape': 0.5817914033765539,
   'nrmse': 0.13438855750805637,
   'r_squared': 0.5186832675840984,
   'crps': 0.7883064072366388
  },
 'test': {
   'mape': 0.45976233322524873,
   'nrmse': 0.6720136850022673,
   'r_squared': 0.26422523039568835,
   'crps': 0.6644610507715243}
}
```

### Holdout fleets

These processes are useful to measure the model's long-term performance, also useful to measure how often do clients need to refresh the model. The code is:

``` python
>>>hfleet = model.holdout_fleet(holdouts=[4,8,12])

## One month prediction
>>>hfleet[0]
{
'divergences': 8,
'max_tree_depth': 9,
'train': {
  'mape': 0.583884525082264,
  'nrmse': 0.13489347548862585,
  'r_squared': 0.5176506945820203,
  'crps': 0.7881281722463331
},
'test': {
  'mape': 0.4675951760449169,
  'nrmse': 0.6866038884371821,
  'r_squared': 0.25191199829654987,
  'crps': 0.6540177705042755}
}
```

### Buld reports

To fully evaluate a fleet just generate a report:

``` python
>>> pfleet.generate_report(output_html="discoveries/data/toy_prod_fleet.html")
```