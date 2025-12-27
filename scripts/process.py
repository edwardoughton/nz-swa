"""
Estimate GDP losses using a basic input-output model. 

Ed Oughton

May 2025

"""
import os
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__),'..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
RESULTS = os.path.join(BASE_PATH, '..', 'results')


def get_supply_side_scenario_shocks_employment():
    """
    Computes percentage employment disruption per sector (comparative approach).
    Used to simulate proportional supply-side shock to value-added (VA) vector.
    """
    output = {}
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')

    for i in range(1, 8):
        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))

        # Sector columns
        start_col = 'Accommodation'
        end_col = 'Wood product manufacturing'
        sector_cols = data.loc[:, start_col:end_col].columns

        sector_shocks = {}
        for sector in sector_cols:
            # Sum disrupted employment across all days
            disrupted = sum(data[sector] * data[f'd{j}'] for j in range(1, 7))
            total_sector_shock = disrupted.sum()

            # Annual employment baseline (365 days)
            baseline = data[sector].sum() * 365

            # Compute percentage shock
            shock_percent = (total_sector_shock / baseline) * 100
            sector_shocks[sector] = shock_percent

        output[f"{filename[:-4]}"] = sector_shocks

    return output


def process_supply_shocks_employment(iso3, scenario_name, supply_shocks):
    """
    Estimates GDP loss using Ghosh model, where input supply_shocks are actual direct
    economic losses (in NZD) per sector — not percentage reductions.

    Parameters
    ----------
    iso3 : str
        Country code (e.g. 'NZL').
    scenario_name : str
        Name of the scenario.
    supply_shocks : dict
        Dictionary of direct losses per sector (in NZD), e.g., {'Commercial': 10_000_000, ...}.
    """
    print(f"Running Ghosh model for {scenario_name}...")

    # Load input file
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)

    # Extract inter-industry transaction matrix
    transactions = df.iloc[0:109, 1:110]
    transactions.index = df.iloc[0:109, 0]
    transactions.columns = transactions.index
    transactions = transactions.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extract value-added vector (VA)
    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA = pd.Series(VA_row.values, index=transactions.columns).apply(pd.to_numeric, errors='coerce')

    # Compute Ghosh B matrix
    total_output = transactions.sum(axis=0) + VA
    B_matrix = transactions.div(total_output, axis=1).fillna(0)
    B_np = B_matrix.values
    I = np.eye(B_np.shape[0])
    G_inv = np.linalg.inv(I - B_np.T)
    G_inv_df = pd.DataFrame(G_inv, index=B_matrix.index, columns=B_matrix.columns)

    # Recalculate internally consistent baseline output
    VA_aligned = VA.reindex(G_inv_df.columns).fillna(0)
    baseline_output = G_inv_df @ VA_aligned
    baseline_output = pd.Series(baseline_output, index=G_inv_df.index)

    # Prepare supply shock factors (as retained output shares)
    shock_factors = pd.Series({k: max(0.0, 1.0 - v / 100.0) for k, v in supply_shocks.items()})
    shock_factors = shock_factors.reindex(VA.index).fillna(1.0)
    shock_factors.to_csv(os.path.join(RESULTS, f'shock_factors_{scenario_name}.csv'))

    # Apply shocks to VA
    shocked_VA = VA * shock_factors
    shocked_VA.to_csv(os.path.join(RESULTS, f'shocked_value_added_{scenario_name}.csv'))

    # Direct GDP loss
    direct_loss = (VA - shocked_VA).clip(lower=0)
    direct_loss_total = direct_loss.sum()

    # Compute baseline and shocked output
    baseline_output = G_inv_df @ VA
    shocked_output = G_inv_df @ shocked_VA

    # Loss computation
    combined = pd.DataFrame({
        'Original Output': baseline_output,
        'Shocked Output': shocked_output,
    })
    combined['Loss'] = (combined['Original Output'] - combined['Shocked Output']).clip(lower=0)
    combined['Description'] = combined.index

    # Merge with sector codes
    lut = pd.read_csv(os.path.join(DATA_PROCESSED, iso3, 'nzsioc_lut.csv'))
    combined = combined.reset_index().merge(lut, left_on='Description', right_on='Description', how='left')
    combined = combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Loss']]

    # Add Direct and Indirect Loss
    direct_loss = direct_loss.reindex(combined['Description']).reset_index(drop=True)
    combined['Direct Loss'] = direct_loss
    combined['Indirect Loss'] = (combined['Loss'] - combined['Direct Loss']).clip(lower=0)

    # Save outputs
    combined.to_csv(os.path.join(RESULTS, f'gdp_loss_by_sector_{scenario_name}.csv'), index=False)
    combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Loss']].to_csv(
        os.path.join(RESULTS, f'supply_side_losses_{scenario_name}.csv'), index=False
    )

    # Write summary
    with open(os.path.join(RESULTS, f'gdp_loss_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million NZD\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million NZD\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million NZD\n")


def get_direct_losses_with_voll():
    """
    Uses employment disruptions, electricity intensity per employee,
    and VoLL to estimate direct economic losses per sector.
    """
    output = {}

    # Load employment shocks
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')

    # Load updated electricity intensity and VoLL per sector
    intensity_data = pd.read_csv(os.path.join(DATA_PROCESSED, 'NZL', 'electricity_intensity_per_employee_all_sectors.csv'))
    intensity_data = intensity_data.set_index('sector_name')

    for i in range(1, 8):

        # if not i == 1:
        #     continue

        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))

        # Determine which columns represent sectors
        start_col = 'Accommodation'
        end_col = 'Wood product manufacturing'
        sectors = data.loc[:, start_col:end_col].columns

        sector_losses = {}

        for sector in sectors:
            if sector not in intensity_data.index:
                print(f"Warning: {sector} not found in intensity data, skipping.")
                continue

            # Retrieve intensity and VoLL
            intensity = intensity_data.at[sector, 'GWh_per_employee']
            voll = intensity_data.at[sector, 'VoLL_usd_MWh']

            # Sum disrupted employment across all six disruption periods
            disrupted_total = sum(data[sector] * data[f'd{j}'] for j in range(1, 7)).sum()

            # Step 1: Estimate lost load in GWh
            lost_gwh = disrupted_total * intensity

            # Step 2: Estimate economic loss ($)
            direct_loss = lost_gwh * voll  # Convert GWh * $/MWh to $

            sector_losses[sector] = direct_loss / 1e6

        output[f"{filename[:-4]}"] = sector_losses

    return output


def process_supply_shocks_with_voll(iso3, scenario_name, supply_shocks):
    """
    Estimates GDP loss using Ghosh model, where input supply_shocks are actual direct
    economic losses (in millions NZD) per sector — NOT percentage reductions.

    Parameters
    ----------
    iso3 : str
        Country code (e.g. 'NZL').
    scenario_name : str
        Name of the scenario.
    supply_shocks : dict
        Dictionary of direct losses per sector (in millions NZD), 
        e.g., {'Commercial': 10.5, ...}
    """
    print(f"Running Ghosh model for {scenario_name}...")

    # Load IO table
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)

    # Extract transactions and VA
    transactions = df.iloc[0:109, 1:110]
    transactions.index = df.iloc[0:109, 0]
    transactions.columns = transactions.index
    transactions = transactions.apply(pd.to_numeric, errors='coerce').fillna(0)

    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA = pd.Series(VA_row.values, index=transactions.columns).apply(pd.to_numeric, errors='coerce')

    # Compute Ghosh inverse
    total_output = transactions.sum(axis=0) + VA
    B_matrix = transactions.div(total_output, axis=1).fillna(0)
    B_np = B_matrix.values
    I = np.eye(B_np.shape[0])
    G_inv = np.linalg.inv(I - B_np.T)
    G_inv_df = pd.DataFrame(G_inv, index=B_matrix.index, columns=B_matrix.columns)

    # Baseline output
    VA_aligned = VA.reindex(G_inv_df.columns).fillna(0)
    baseline_output = G_inv_df @ VA_aligned
    baseline_output = pd.Series(baseline_output, index=G_inv_df.index)

    # Convert supply_shocks (in millions NZD) to series
    direct_loss_series = pd.Series(supply_shocks).reindex(VA.index).fillna(0)

    # Compute shock factors
    shock_factors = (1.0 - (direct_loss_series / VA)).clip(lower=0)
    shock_factors.to_csv(os.path.join(RESULTS, f'shock_factors_{scenario_name}.csv'))

    # Apply shocks
    shocked_VA = VA * shock_factors
    shocked_VA.to_csv(os.path.join(RESULTS, f'shocked_value_added_{scenario_name}.csv'))

    # Compute shocked output
    shocked_output = G_inv_df @ shocked_VA

    # Loss summary
    combined = pd.DataFrame({
        'Original Output': baseline_output,
        'Shocked Output': shocked_output,
    })
    combined['Loss'] = (combined['Original Output'] - combined['Shocked Output']).clip(lower=0)
    combined['Description'] = combined.index

    # Merge with sector LUT
    lut = pd.read_csv(os.path.join(DATA_PROCESSED, iso3, 'nzsioc_lut.csv'))
    combined = combined.reset_index().merge(lut, left_on='Description', right_on='Description', how='left')
    combined = combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Loss']]

    # Add direct/indirect losses
    direct_loss_aligned = direct_loss_series.reindex(combined['Description']).reset_index(drop=True)
    combined['Direct Loss'] = direct_loss_aligned
    combined['Indirect Loss'] = (combined['Loss'] - combined['Direct Loss']).clip(lower=0)

    # Save sector-level results
    combined.to_csv(os.path.join(RESULTS, f'gdp_loss_by_sector_{scenario_name}.csv'), index=False)
    combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Loss']].to_csv(
        os.path.join(RESULTS, f'supply_side_losses_{scenario_name}.csv'), index=False
    )

    # Write summary file
    with open(os.path.join(RESULTS, f'gdp_loss_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million NZD\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million NZD\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million NZD\n")


if __name__ == "__main__":

    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)

    iso3 = 'NZL'

    shocks_supply = get_supply_side_scenario_shocks_employment()
    for scenario_name, shocks in shocks_supply.items():
        # if not scenario_name == 'scenario1':
        #     continue
        process_supply_shocks_employment(iso3, scenario_name+'_employment_approach', shocks)

    shocks_supply = get_direct_losses_with_voll()
    for scenario_name, shocks in shocks_supply.items():
        # if not scenario_name == 'scenario1':
        #     continue
        process_supply_shocks_with_voll(iso3, scenario_name+'_survey_approach', shocks)