import pandas as pd
from cassandra.pymc_toolkit.pymc_model import PymcModel

data = pd.read_csv('data/Poseidon.csv')

target_name = 'total_revenue'

channel_names = ['dsp_spend_us_amazonadsspend', 'dsp_spend_others_amazonadsspend', 'dsp_spend_us_amazondspspend',
       'dsp_spend_others_amazondspspend', 'dsp_spend_us_totalspend', 'dsp_spend_others_totalspend','mntn_tv_total_spend', 
       'prime_variable_total_cost_usd', 'prime_variable_branded_searches', 'pr_coverage_others_uvm',
       'snapchat_ads_us_spend','google_ads_search_brand_costs', 'google_ads_search_no_brand_costs', 
       'google_ads_video_costs', 'google_ads_discovery_costs', 'google_ads_shopping_costs',
       'google_ads_display_costs', 'google_ads_performance_max_costs', 'google_ads_others_costs', 'influencer_spend',
       'amazon_ads_impressions', 'amazon_ads_spend', 'fb_tof_spend', 'fb_bof_spend', 'fb_mof_spend', 'fb_ret_spend',
       'bing_spend','pr_uvm']

control_names = ['discounts', 'emails_us_total_emails_sent', 'influencers_us_total_reach']

model = PymcModel(client_data=data,
             target_name=target_name,
             date_column='date_week',
             channel_names=channel_names,
             control_names=control_names,
             lag_max=4,
             scale_data=True,
             saturation='michaelis_menten',
             time_varying_media=True)

pfleet = model.production_fleet(n_test=12, chains=4, draws=1000,tune=500,cores=4)
pfleet.summary()

pfleet.generate_report(output_html="toy_prod_fleet.html")
