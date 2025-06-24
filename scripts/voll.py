"""
VOLL analysis.

Ed Oughton

June 2025

"""
# import sys
import os
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__),'..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, '..', '..', 'data_raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
VIS = os.path.join(BASE_PATH, '..', 'vis', 'figures')

mpl.rcParams['font.family'] = 'Times New Roman'


def calc_employment_energy_intensity_broad_categories():
    """
    
    """
    filename = 'employment_lut.csv'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    if not os.path.exists(path_in):
        print(f'First need to run preprocess.py to generate:{path_in}')
    data = pd.read_csv(path_in)
    data = data[['Target Code', 'ec_count']]

    filename = "nz_industry_broad_category_mapping.csv"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    lut = lut[['industry_groupings','Broad_Category']]

    merged_df = pd.merge(data, lut, left_on='Target Code', right_on='industry_groupings', how='left')
    merged_df = merged_df[['Broad_Category', 'industry_groupings', 'ec_count']]
    merged_df = merged_df.groupby('Broad_Category', as_index=False)['ec_count'].sum()

    filename = "electricity-2025-q1.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    data = pd.read_excel(path_in, sheet_name='2 - Annual GWh', header=8)
    data = data[['Calendar year', 2024]]
    data = data[19:31]
    data = data[data['Calendar year'] != "Industrial:"]
    data['Broad_Category'] = data['Calendar year']
    data['elec_consumption_gwh'] = data[2024]
    data = data[['Broad_Category','elec_consumption_gwh']]
    merged_df = pd.merge(merged_df, data, left_on='Broad_Category', right_on='Broad_Category', how='left')
    merged_df['GWh_per_employee'] = (merged_df['elec_consumption_gwh'] / merged_df['ec_count'])

    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)
    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA_df = VA_row.reset_index()
    VA_df.columns = ['Industry_Code', 'Total_Value_Added']
    filename = "nz_industry_broad_category_mapping.csv"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    VA_df = pd.merge(VA_df, lut, left_on='Industry_Code', right_on='sector_name', how='left')
    VA_df = VA_df[['Industry_Code', 'Total_Value_Added', 'Broad_Category']]
    VA_df = VA_df.drop_duplicates()
    VA_df = VA_df.groupby('Broad_Category', as_index=False)['Total_Value_Added'].sum()

    merged_df = pd.merge(merged_df, VA_df, left_on='Broad_Category', right_on='Broad_Category', how='left')
    # 1.188 represents an 18% increase in inflation of the CPI from 2018 to 2025
    merged_df['VoLL_usd_MWh'] = merged_df['Total_Value_Added']*1e6 / (merged_df['elec_consumption_gwh']*1e3) * 1.188

    merged_df.to_csv(os.path.join(DATA_PROCESSED, 'NZL', 'electricity_intensity_per_employee_broad_categories.csv'), index=False)


def calc_employment_energy_intensity_all_sectors():
    """
    
    
    """
    filename = 'employment_lut.csv'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    if not os.path.exists(path_in):
        print(f'First need to run preprocess.py to generate:{path_in}')
    data = pd.read_csv(path_in)
    data = data[['Target Code', 'ec_count']]

    filename = "nz_industry_broad_category_mapping.csv"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    lut = lut[['industry_groupings','Broad_Category']]

    merged_df = pd.merge(data, lut, left_on='Target Code', right_on='industry_groupings', how='left')
    merged_df = merged_df[['Broad_Category', 'industry_groupings', 'ec_count']]
    merged_df = merged_df.groupby('Broad_Category', as_index=False)['ec_count'].sum()

    filename = "electricity-2025-q1.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    data = pd.read_excel(path_in, sheet_name='2 - Annual GWh', header=8)
    data = data[['Calendar year', 2024]]
    data = data[19:31]
    data = data[data['Calendar year'] != "Industrial:"]
    data['Broad_Category'] = data['Calendar year']
    data['elec_consumption_gwh'] = data[2024]
    data = data[['Broad_Category','elec_consumption_gwh']]
    merged_df = pd.merge(merged_df, data, left_on='Broad_Category', right_on='Broad_Category', how='left')
    merged_df['GWh_per_employee'] = merged_df['elec_consumption_gwh'] / merged_df['ec_count'] 

    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)
    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA_df = VA_row.reset_index()
    VA_df.columns = ['Industry_Code', 'Total_Value_Added']
    filename = "nz_industry_broad_category_mapping.csv"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    VA_df = pd.merge(VA_df, lut, left_on='Industry_Code', right_on='sector_name', how='left')
    VA_df = VA_df[['Industry_Code', 'Total_Value_Added', 'Broad_Category']]
    VA_df = VA_df.drop_duplicates()
    VA_df = VA_df.groupby('Broad_Category', as_index=False)['Total_Value_Added'].sum()

    merged_df = pd.merge(merged_df, VA_df, left_on='Broad_Category', right_on='Broad_Category', how='left')
    # 1.188 represents an 18% increase in inflation of the CPI from 2018 to 2025
    merged_df['VoLL_usd_MWh'] = merged_df['Total_Value_Added']*1e6 / (merged_df['elec_consumption_gwh']*1e3) * 1.188

    filename = 'employment_lut.csv'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    if not os.path.exists(path_in):
        print(f'First need to run preprocess.py to generate:{path_in}')
    data = pd.read_csv(path_in)
    data = data[['sector_name', 'ec_count']]

    filename = "nz_industry_broad_category_mapping.csv"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    lut = lut[['industry_groupings','sector_name', 'Broad_Category']]

    data = pd.merge(data, lut, left_on='sector_name', right_on='sector_name', how='left')
    data = data[['Broad_Category', 'sector_name', 'ec_count']]
    data = data.groupby(['sector_name','Broad_Category'], as_index=False)['ec_count'].sum()

    subset = merged_df[['Broad_Category', 'GWh_per_employee','VoLL_usd_MWh']]
    all_sectors = pd.merge(data, subset, left_on='Broad_Category', right_on='Broad_Category', how='left')
    all_sectors['GWh_per_sector'] = all_sectors['ec_count'] * all_sectors['GWh_per_employee']

    path_out = os.path.join(DATA_PROCESSED, 'NZL', 'electricity_intensity_per_employee_all_sectors.csv')
    all_sectors.to_csv(path_out,index=False)



if __name__ == "__main__":

    calc_employment_energy_intensity_broad_categories()

    calc_employment_energy_intensity_all_sectors()
