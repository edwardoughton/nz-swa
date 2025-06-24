"""
Estimate GDP losses using a basic input-output model. 

Ed Oughton

February 2022

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


def get_demand_side_scenario_shocks():
    """
    
    """
    output = {}

    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')
    
    for i in range(1,6):

        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))
        
        # total_hours = 24 * 365
        
        d1 = data['population'] * data['d1']
        d2 = data['population'] * data['d2']
        d3 = data['population'] * data['d3']
        d4 = data['population'] * data['d4']
        d5 = data['population'] * data['d5']
        d6 = data['population'] * data['d6']

        # Get numerator
        total_without_power = d1.sum() + d2.sum() + d3.sum() + d4.sum() + d5.sum() + d6.sum()

        # Get denominator 
        denominator = data['population'].sum() * 365

        # Percentage of lost demand
        shock = (total_without_power / denominator) * 100
        output[f"{filename[:-4]}"] = float(shock)

    return output


def process_demand_shocks(iso3, scenario_name, shock):
    """
    Processes New Zealand Input-Output data and estimates GDP loss due to a shock.

    This function reads national accounts input-output tables from an Excel file, 
    applies a specified shock to final demand, and estimates the resulting direct 
    and indirect economic impacts using Input-Output analysis. It performs the following steps:

    1. Extracts the final demand vector and applies a shock.
    2. Constructs the inter-industry transaction matrix.
    3. Computes the technical coefficient matrix (A) from transactions, using total output.
    4. Calculates the Leontief inverse (L = (I - A)^-1).
    5. Estimates impacts on total output (X = L * Y) accounting for indirect effects.
    6. Calculates and exports direct, indirect, and total economic losses.

    Outputs are written to the 'processed' and 'RESULTS' directories.

    Parameters
    ----------
    iso3 : str
        The ISO3 country code (currently not used in this function, reserved for future use).
    scenario_name : str
        The identifier for the scenario, used in naming output files.
    shock : float
        A scalar multiplier applied to reduce the final demand (e.g., 0.9 for a 10% reduction).
    """
    print(f"for scenario {scenario_name}, shock: {shock}")
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    df = pd.read_excel(path_in, sheet_name='4 Transactions', header=5)

    # Extract the total output row (used for computing technical coefficients)
    total_output = df.set_index("Unnamed: 0")
    bottom_row_column_sums = total_output.loc[['Total output']]
    bottom_row_column_sums = bottom_row_column_sums.apply(pd.to_numeric, errors='coerce')

    # Identify the start and end columns for transaction data
    col_start = df.columns.get_loc('Unnamed: 0')
    col_end = df.columns.get_loc('Religious services; civil, professional, and other interest groups')
    bottom_row_column_sums = bottom_row_column_sums.iloc[:, col_start:col_end]
    bottom_row_column_sums.to_csv(os.path.join(BASE_PATH, 'processed', 'bottom_row_column_sums.csv'))

    # Extract and apply shock to the final demand vector
    Y = df.set_index("Unnamed: 0")
    Y = Y[['Sub-total final consumption expenditure']]
    Y = Y[0:109]  # Select only the relevant industries
    reduced_demand = (Y * (shock/100))
    # print(Y.sum(), reduced_demand.sum())
    Y = reduced_demand
    # Y.to_csv(os.path.join(RESULTS, f'direct_loss_{scenario_name}.csv'))

    # Extract inter-industry transaction matrix
    transactions = df.iloc[:, col_start:col_end + 1]
    transactions = transactions.set_index("Unnamed: 0")
    transactions = transactions[0:111]
    transactions = transactions.apply(pd.to_numeric, errors='coerce')

    # Drop total and balancing rows; fill any missing values
    transactions = transactions.iloc[:-2]
    transactions = transactions.fillna(0)
    transactions.to_csv(os.path.join(BASE_PATH, 'processed', 'transactions.csv'))

    # Compute technical coefficient matrix A by dividing each column by total output
    column_sums = bottom_row_column_sums.iloc[0]
    A_matrix = transactions.div(column_sums, axis=1)
    A_matrix.to_csv(os.path.join(BASE_PATH, 'processed', 'A_matrix.csv'))

    # Convert A to NumPy for matrix operations
    A_matrix_np = A_matrix.values

    # Create identity matrix I
    I = np.eye(len(A_matrix_np))
    I_df = pd.DataFrame(I, index=A_matrix.index, columns=A_matrix.columns)
    I_df.to_csv(os.path.join(BASE_PATH, 'processed', 'I_matrix.csv'))

    # Compute the Leontief inverse: L = (I - A)^(-1)
    L = np.linalg.inv(I - A_matrix_np)
    leontief_inverse = pd.DataFrame(L, index=A_matrix.index, columns=A_matrix.columns)
    leontief_inverse.to_csv(os.path.join(BASE_PATH, 'processed', 'L_matrix.csv'))

    # Compute total output X from shocked final demand
    X = leontief_inverse @ Y
    X.to_csv(os.path.join(BASE_PATH, 'processed', 'X.csv'))

    # Calculate indirect losses: difference between total and direct impact
    indirect = X - reduced_demand

    # Combine direct and indirect impacts
    combined_losses = pd.DataFrame({
        'Direct Loss': reduced_demand.squeeze(),  # Remove single-dimension entries
        'Indirect Loss': indirect.squeeze()
    })

    # Calculate total loss and export results
    combined_losses['Total Loss'] = combined_losses['Direct Loss'] + combined_losses['Indirect Loss']
    
    # Add SIOC code
    filename = "nzsioc_lut.csv"
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)

    combined_losses = combined_losses.reset_index()
    combined_losses = combined_losses.merge(lut, left_on='Unnamed: 0', right_on='Description')

    combined_losses = combined_losses[['Description', 'NZSIOC', 'Direct Loss', 'Indirect Loss', 'Total Loss']]
    combined_losses.to_csv(os.path.join(RESULTS, f'combined_losses_{scenario_name}.csv'))

    # print(round(combined_losses['Direct Loss'].sum()), round(combined_losses['Indirect Loss'].sum()), round(combined_losses['Total Loss'].sum()))


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

    # shocks_demand = get_demand_side_scenario_shocks()
    # for scenario_name, shock in shocks_demand.items():
    #     process_demand_shocks(iso3, scenario_name, shock)

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