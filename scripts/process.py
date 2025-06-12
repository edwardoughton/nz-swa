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


def get_supply_side_scenario_shocks():
    """
    
    """
    output = {}

    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')
    
    for i in range(1,8):

        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))
        
        # total_hours = 24 * 365
        start_col = 'Accommodation'
        end_col = 'Wood product manufacturing'  # Assuming you want from 'A' to 'S' inclusive
        sector_cols = data.loc[:, start_col:end_col].columns

        sector_shocks = {}
        for sector in sector_cols:

            d1 = data[sector] * data['d1']
            d2 = data[sector] * data['d2']
            d3 = data[sector] * data['d3']
            d4 = data[sector] * data['d4']
            d5 = data[sector] * data['d5']
            d6 = data[sector] * data['d6']

            # Get numerator
            total_sector_shock = d1.sum() + d2.sum() + d3.sum() + d4.sum() + d5.sum() + d6.sum()

            # Get denominator 
            denominator = data[sector].sum() * 365

            # Percentage of lost demand
            sector_shock = (total_sector_shock / denominator) * 100
            sector_shocks[sector] = float(sector_shock)
        
        output[f"{filename[:-4]}"] = sector_shocks

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


def process_supply_shocks(iso3, scenario_name, supply_shocks):
    """
    Processes Input-Output data and estimates GDP loss due to a supply-side shock 
    using the Ghosh model.

    Parameters
    ----------
    iso3 : str
        Country code (e.g. 'NZL').
    scenario_name : str
        Name of the scenario.
    supply_shocks : dict
        Dictionary of percent shocks (0–100) by sector (e.g., {'A': 10.5, ...}).
    """
    print(f"Running Ghosh model for {scenario_name}...")

    # Load input file
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)

    # Extract inter-industry transaction matrix (first 109 rows and columns)
    transactions = df.iloc[0:109, 1:110]
    transactions.index = df.iloc[0:109, 0]
    transactions.columns = transactions.index
    transactions = transactions.apply(pd.to_numeric, errors='coerce').fillna(0)
    transactions.to_csv(os.path.join(BASE_PATH, 'processed', 'transactions.csv'))

    # Extract value-added vector (VA)
    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA = pd.Series(VA_row.values, index=transactions.columns).apply(pd.to_numeric, errors='coerce')
    VA.to_csv(os.path.join(BASE_PATH, 'processed', 'value_added.csv'))

    # Compute total output from the model
    total_output = transactions.sum(axis=0) + VA

    # Compute Ghosh B matrix
    B_matrix = transactions.div(total_output, axis=1).fillna(0)
    B_matrix.to_csv(os.path.join(BASE_PATH, 'processed', 'B_matrix.csv'))

    # Ghosh inverse
    B_np = B_matrix.values
    I = np.eye(B_np.shape[0])
    G_inv = np.linalg.inv(I - B_np.T)
    G_inv_df = pd.DataFrame(G_inv, index=B_matrix.index, columns=B_matrix.columns)
    G_inv_df.to_csv(os.path.join(BASE_PATH, 'processed', 'Ghosh_inverse.csv'))

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

    # Compute shocked output
    shocked_VA_aligned = shocked_VA.reindex(G_inv_df.columns).fillna(0)
    shocked_output = G_inv_df @ shocked_VA_aligned
    shocked_output_series = pd.Series(shocked_output, index=G_inv_df.index)
    shocked_output_series.to_csv(os.path.join(RESULTS, f'gross_output_{scenario_name}.csv'))

    # Calculate output loss
    combined = pd.DataFrame({
        'Original Output': baseline_output,
        'Shocked Output': shocked_output_series
    })
    combined['Loss'] = (combined['Original Output'] - combined['Shocked Output']).clip(lower=0)
    combined['Description'] = combined.index

    # Merge with sector codes
    lut = pd.read_csv(os.path.join(DATA_PROCESSED, iso3, 'nzsioc_lut.csv'))
    combined = combined.reset_index().merge(lut, left_on='Description', right_on='Description', how='left')
    combined = combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Loss']]
    combined.to_csv(os.path.join(RESULTS, f'supply_side_losses_{scenario_name}.csv'), index=False)

    # Merge sector-level direct and indirect losses
    direct_loss = direct_loss.reindex(combined['Description']).reset_index(drop=True)
    combined['Direct Loss'] = direct_loss
    combined['Indirect Loss'] = (combined['Loss'] - combined['Direct Loss']).clip(lower=0)

    # Save sector-level losses
    combined[['Description', 'NZSIOC', 'Direct Loss', 'Indirect Loss', 'Loss']].to_csv(
        os.path.join(RESULTS, f'gdp_loss_by_sector_{scenario_name}.csv'), index=False
    )

    # Also write summary totals
    with open(os.path.join(RESULTS, f'gdp_loss_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million NZD\\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million NZD\\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million NZD\\n")


if __name__ == "__main__":

    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)

    iso3 = 'NZL'

    # shocks_demand = get_demand_side_scenario_shocks()

    # for scenario_name, shock in shocks_demand.items():
    #     process_demand_shocks(iso3, scenario_name, shock)

    shocks_supply = get_supply_side_scenario_shocks()

    for scenario_name, shocks in shocks_supply.items():
        # if not scenario_name == 'scenario1':
        #     continue
        process_supply_shocks(iso3, scenario_name, shocks)
