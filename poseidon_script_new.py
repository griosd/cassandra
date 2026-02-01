import pandas as pd
from cassandra.pymc_toolkit.pymc_model import PymcModel

data = pd.read_csv('data/Poseidon.csv')

target_name = 'total_revenue'

channel_amazon = ['amazon_ads_spend','google_ads_search_brand_costs', 'google_ads_others_costs']
channel_shopify = ['fb_bof_spend', 'google_ads_search_brand_costs', 'fb_tof_spend', 'google_ads_others_costs']

control_names = ['discounts', 'emails_us_total_emails_sent', 'influencers_us_total_reach']
channel_names = list(set(channel_amazon + channel_shopify))
data = data[['total_revenue', 'date_week'] + control_names + channel_names].copy()

model = PymcModel(client_data=data,
             target_name=target_name,
             date_column='date_week',
             channel_names=channel_names,
             control_names=control_names,
             lag_max=1,
             scale_data=True,
             saturation='michaelis_menten',
             time_varying_media=True)

pfleet = model.production_fleet(n_test=12, chains=4, draws=1000,tune=500,cores=4)
pfleet.summary()

pfleet.generate_report(output_html="toy_prod_fleet_new.html")
