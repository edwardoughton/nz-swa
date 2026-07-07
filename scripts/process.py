"""
Estimate GDP losses using a basic input-output model. 

Ed Oughton

May 2025

"""
import os
import configparser
import numpy as np
import pandas as pd

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__),'..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
RESULTS = os.path.join(BASE_PATH, '..', 'results')

GDP_DEFLATOR_2020_TO_2026 = 1.255
PRICE_YEAR_LABEL = '2026 NZD'


def _load_io_table_components():
    """
    Load the national input-output table and return the core model inputs.
    """
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    df = pd.read_excel(os.path.join(folder, filename), sheet_name='4 Transactions', header=5)

    transactions = df.iloc[0:109, 1:110]
    transactions.index = df.iloc[0:109, 0]
    transactions.columns = transactions.index
    transactions = transactions.apply(pd.to_numeric, errors='coerce').fillna(0)

    VA_row = df[df['Unnamed: 0'] == 'Total value added'].iloc[0, 1:110]
    VA = pd.Series(VA_row.values, index=transactions.columns).apply(pd.to_numeric, errors='coerce')

    total_output_row = df[df['Unnamed: 0'] == 'Total output'].iloc[0, 1:110]
    total_output = pd.Series(total_output_row.values, index=transactions.columns).apply(pd.to_numeric, errors='coerce')

    final_demand_columns = [
        'Exports',
        'Final Consumption Expenditure - households',
        'Final Consumption Expenditure - NPISH',
        'Final Consumption Expenditure - central government',
        'Final Consumption Expenditure - local government',
        'Gross fixed capital formation',
        'Change in inventories',
    ]
    final_demand = (
        df.loc[0:108, final_demand_columns]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )
    final_demand.index = transactions.index

    return transactions, VA, total_output, final_demand


def _get_value_added_to_output_ratio(VA, total_output):
    """
    Build sector value-added to output ratios for converting output losses to GDP.
    """
    ratio = VA.div(total_output.replace(0, np.nan))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    return ratio.clip(lower=0)


def _get_diagonal_model_gdp_loss(model_inverse, direct_shock, VA, total_output):
    """
    Estimate direct GDP loss as the same-sector contribution from the IO model.

    direct_shock is the exogenous model shock vector: value-added loss for Ghosh
    models, or final-demand loss for Leontief models.
    """
    direct_shock = pd.Series(direct_shock).reindex(model_inverse.columns).fillna(0).clip(lower=0)
    diagonal_output_loss = pd.Series(
        np.diag(model_inverse.values) * direct_shock.values,
        index=model_inverse.index,
    )
    va_output_ratio = _get_value_added_to_output_ratio(VA, total_output).reindex(model_inverse.index).fillna(0)
    return diagonal_output_loss * va_output_ratio

def _build_gdp_loss_table(iso3, baseline_output, shocked_output, direct_gdp_loss, VA, total_output):
    """
    Convert output losses to GDP losses and assemble the sector result table.
    """
    baseline_output = pd.Series(baseline_output).astype(float)
    shocked_output = pd.Series(shocked_output).reindex(baseline_output.index).fillna(0).astype(float)
    direct_gdp_loss = pd.Series(direct_gdp_loss).astype(float)

    output_loss = (baseline_output - shocked_output).clip(lower=0)
    va_output_ratio = _get_value_added_to_output_ratio(VA, total_output).reindex(baseline_output.index).fillna(0)

    combined = pd.DataFrame({
        'Description': baseline_output.index,
        'Original Output': baseline_output.values,
        'Shocked Output': shocked_output.values,
        'Output Loss': output_loss.values,
        'Value Added to Output Ratio': va_output_ratio.values,
    })
    combined['Original GDP Estimate'] = combined['Original Output'] * combined['Value Added to Output Ratio']
    combined['Shocked GDP Estimate'] = combined['Shocked Output'] * combined['Value Added to Output Ratio']
    combined['Loss'] = combined['Output Loss'] * combined['Value Added to Output Ratio']

    lut = pd.read_csv(os.path.join(DATA_PROCESSED, iso3, 'nzsioc_lut.csv'))
    combined = combined.merge(lut, on='Description', how='left')

    direct_loss_aligned = direct_gdp_loss.reindex(combined['Description']).fillna(0).reset_index(drop=True)
    combined['Direct Loss'] = direct_loss_aligned
    combined['Indirect Loss'] = combined['Loss'] - combined['Direct Loss']

    monetary_columns = [
        'Original Output',
        'Shocked Output',
        'Output Loss',
        'Original GDP Estimate',
        'Shocked GDP Estimate',
        'Loss',
        'Direct Loss',
        'Indirect Loss',
    ]
    combined[monetary_columns] = combined[monetary_columns] * GDP_DEFLATOR_2020_TO_2026
    return combined[
        [
            'Description',
            'NZSIOC',
            'Original Output',
            'Shocked Output',
            'Output Loss',
            'Value Added to Output Ratio',
            'Original GDP Estimate',
            'Shocked GDP Estimate',
            'Loss',
            'Direct Loss',
            'Indirect Loss',
        ]
    ]


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


def get_demand_side_scenario_shocks_population():
    """
    Compute the share of population disrupted in each scenario.

    The returned shock is a single percentage applied to household final demand
    in the Leontief model.
    """
    output = {}
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')

    for i in range(1, 8):
        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))

        day_columns = [f'd{j}' for j in range(1, 7) if f'd{j}' in data.columns]
        disrupted = sum(data['population'] * data[day] for day in day_columns)
        total_population_shock = disrupted.sum()

        baseline_population = data['population'].sum() * 365
        shock_percent = (total_population_shock / baseline_population) * 100 if baseline_population else 0

        output[f"{filename[:-4]}"] = shock_percent

    return output


def get_demand_side_scenario_shocks_survey_voll():
    """
    Compute a demand-side household shock using survey-based residential VoLL.

    This applies a population-weighted disruption metric, where each location is
    weighted by its implied residential VoLL from the Transpower survey.
    """
    output = {}
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')

    lut_path = os.path.join(DATA_PROCESSED, 'NZL', 'residential_voll_lut.csv')
    if not os.path.exists(lut_path):
        raise FileNotFoundError(
            f'Residential VoLL lookup not found at {lut_path}. Run scripts/voll.py first.'
        )

    residential_voll = pd.read_csv(lut_path)
    residential_voll = residential_voll[['location3', 'residential_voll_nzd_mwh']]

    for i in range(1, 8):
        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))

        if 'location3' not in data.columns:
            if 'location' in data.columns:
                data['location3'] = data['location'].astype(str).str[:3]
            else:
                raise ValueError(f'No location key found in {filename} for residential VoLL merge')

        merged = data.merge(residential_voll, on='location3', how='left')
        merged['residential_voll_nzd_mwh'] = merged['residential_voll_nzd_mwh'].fillna(
            residential_voll['residential_voll_nzd_mwh'].mean()
        )

        day_columns = [f'd{j}' for j in range(1, 7) if f'd{j}' in merged.columns]
        weighted_population = merged['population'] * merged['residential_voll_nzd_mwh']

        disrupted = sum(weighted_population * merged[day] for day in day_columns)
        total_weighted_shock = disrupted.sum()

        baseline_weighted = weighted_population.sum() * 365
        shock_percent = (total_weighted_shock / baseline_weighted) * 100 if baseline_weighted else 0

        output[f"{filename[:-4]}"] = shock_percent

    return output


def process_supply_shocks_employment(iso3, scenario_name, supply_shocks):
    """
    Estimates GDP loss using a Ghosh model with proportional supply shocks.

    Parameters
    ----------
    iso3 : str
        Country code (e.g. 'NZL').
    scenario_name : str
        Name of the scenario.
    supply_shocks : dict
        Dictionary of proportional value-added shocks per sector, in percent.
    """
    print(f"Running Ghosh model for {scenario_name}...")

    # Load input file
    transactions, VA, total_output, _ = _load_io_table_components()

    # Compute Ghosh B matrix
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
    shock_factors = shock_factors.reindex(VA_aligned.index).fillna(1.0)
    shock_factors.to_csv(os.path.join(RESULTS, f'shock_factors_{scenario_name}.csv'))

    # Apply shocks to VA
    shocked_VA = VA_aligned * shock_factors
    shocked_VA.to_csv(os.path.join(RESULTS, f'shocked_value_added_{scenario_name}.csv'))

    direct_va_loss = (VA_aligned - shocked_VA).clip(lower=0)

    # Compute shocked output
    shocked_output = G_inv_df @ shocked_VA
    direct_gdp_loss = _get_diagonal_model_gdp_loss(
        G_inv_df,
        direct_va_loss,
        VA_aligned,
        total_output,
    )

    combined = _build_gdp_loss_table(
        iso3,
        baseline_output,
        shocked_output,
        direct_gdp_loss,
        VA_aligned,
        total_output,
    )

    # Save outputs
    combined.to_csv(os.path.join(RESULTS, f'gdp_loss_by_sector_{scenario_name}.csv'), index=False)
    combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Output Loss']].to_csv(
        os.path.join(RESULTS, f'supply_side_losses_{scenario_name}.csv'), index=False
    )

    # Write summary
    with open(os.path.join(RESULTS, f'gdp_loss_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")


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
            voll = intensity_data.at[sector, 'VoLL_nzd_MWh']

            # Sum disrupted employment across all six disruption periods
            disrupted_total = sum(data[sector] * data[f'd{j}'] for j in range(1, 7)).sum()

            # Step 1: Estimate lost load in MWh from employee-days and annual GWh/employee.
            lost_mwh = disrupted_total * (intensity / 365) * 1e3

            # Step 2: Estimate economic loss in NZD.
            direct_loss = lost_mwh * voll

            sector_losses[sector] = direct_loss / 1e6

        output[f"{filename[:-4]}"] = sector_losses

    voll_folder = os.path.join(DATA_PROCESSED, 'NZL', 'VoLL')
    os.makedirs(voll_folder, exist_ok=True)

    records = []
    for scenario_name, sector_losses in output.items():
        for sector_name, direct_loss in sector_losses.items():
            records.append({
                'scenario': scenario_name,
                'sector_name': sector_name,
                'direct_voll_loss_million_nzd_2020': direct_loss,
            })

    pd.DataFrame(records).sort_values(['scenario', 'sector_name']).to_csv(
        os.path.join(voll_folder, 'direct_losses_with_voll_by_sector.csv'),
        index=False,
    )

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
    transactions, VA, total_output, _ = _load_io_table_components()

    # Compute Ghosh inverse
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
    direct_loss_series = pd.Series(supply_shocks).reindex(VA_aligned.index).fillna(0)

    # Compute shock factors
    shock_ratio = direct_loss_series.div(VA_aligned.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    shock_factors = (1.0 - shock_ratio).clip(lower=0)
    shock_factors.to_csv(os.path.join(RESULTS, f'shock_factors_{scenario_name}.csv'))

    # Apply shocks
    shocked_VA = VA_aligned * shock_factors
    shocked_VA.to_csv(os.path.join(RESULTS, f'shocked_value_added_{scenario_name}.csv'))

    # Compute shocked output
    shocked_output = G_inv_df @ shocked_VA
    direct_va_loss = (VA_aligned - shocked_VA).clip(lower=0)
    direct_gdp_loss = _get_diagonal_model_gdp_loss(
        G_inv_df,
        direct_va_loss,
        VA_aligned,
        total_output,
    )

    combined = _build_gdp_loss_table(
        iso3,
        baseline_output,
        shocked_output,
        direct_gdp_loss,
        VA_aligned,
        total_output,
    )

    direct_voll_loss = direct_loss_series.reindex(combined['Description']).fillna(0).reset_index(drop=True)
    combined['Direct VoLL Loss'] = direct_voll_loss * GDP_DEFLATOR_2020_TO_2026
    combined['GDP Propagation Difference'] = combined['Loss'] - combined['Direct VoLL Loss']

    # Save sector-level results
    combined.to_csv(os.path.join(RESULTS, f'gdp_loss_by_sector_{scenario_name}.csv'), index=False)
    combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Output Loss']].to_csv(
        os.path.join(RESULTS, f'supply_side_losses_{scenario_name}.csv'), index=False
    )

    # Write summary file
    with open(os.path.join(RESULTS, f'gdp_loss_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")

        f.write(f"Direct VoLL loss used as shock input: {round(combined['Direct VoLL Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"GDP propagation difference: {round(combined['GDP Propagation Difference'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")


def process_demand_shocks_population(iso3, scenario_name, population_shock_percent):
    """
    Estimate output losses using a Leontief demand model.

    Population disruption is used as a proportional reduction in household final
    demand, while all other final-demand components are held constant.
    """
    print(f"Running Leontief model for {scenario_name}...")

    transactions, VA, total_output, final_demand = _load_io_table_components()

    total_final_demand = final_demand.sum(axis=1).reindex(transactions.index).fillna(0)

    leontief_output = transactions.sum(axis=1) + total_final_demand

    A_matrix = transactions.div(leontief_output, axis=1).fillna(0)
    A_np = A_matrix.values
    I = np.eye(A_np.shape[0])
    L_inv = np.linalg.inv(I - A_np)
    L_inv_df = pd.DataFrame(L_inv, index=A_matrix.index, columns=A_matrix.columns)

    household_final_demand = final_demand['Final Consumption Expenditure - households'].reindex(A_matrix.index).fillna(0)

    retained_share = max(0.0, 1.0 - population_shock_percent / 100.0)
    shock_factors = pd.Series(retained_share, index=A_matrix.index, name='household_demand_retained_share')
    shock_factors.to_csv(os.path.join(RESULTS, f'demand_shock_factors_{scenario_name}.csv'))

    shocked_household_final_demand = household_final_demand * retained_share
    shocked_household_final_demand.to_csv(
        os.path.join(RESULTS, f'shocked_household_final_demand_{scenario_name}.csv')
    )

    shocked_total_final_demand = total_final_demand - household_final_demand + shocked_household_final_demand

    baseline_output = L_inv_df @ total_final_demand
    shocked_output = L_inv_df @ shocked_total_final_demand

    direct_final_demand_loss = (household_final_demand - shocked_household_final_demand).clip(lower=0)
    direct_gdp_loss = _get_diagonal_model_gdp_loss(
        L_inv_df,
        direct_final_demand_loss,
        VA,
        total_output,
    )

    combined = _build_gdp_loss_table(
        iso3,
        baseline_output,
        shocked_output,
        direct_gdp_loss,
        VA,
        total_output,
    )

    combined.to_csv(os.path.join(RESULTS, f'demand_side_gdp_loss_by_sector_{scenario_name}.csv'), index=False)
    combined[['Description', 'NZSIOC', 'Original Output', 'Shocked Output', 'Output Loss']].to_csv(
        os.path.join(RESULTS, f'demand_side_losses_{scenario_name}.csv'), index=False
    )

    with open(os.path.join(RESULTS, f'demand_side_summary_{scenario_name}.csv'), 'w') as f:
        f.write(f"Direct GDP loss: {round(combined['Direct Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Indirect GDP loss: {round(combined['Indirect Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
        f.write(f"Total GDP loss: {round(combined['Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")


if __name__ == "__main__":

    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)

    iso3 = 'NZL'

    shocks_supply = get_supply_side_scenario_shocks_employment()
    for scenario_name, shocks in shocks_supply.items():
        process_supply_shocks_employment(iso3, scenario_name+'_employment_approach', shocks)

    shocks_supply_voll = get_direct_losses_with_voll()
    for scenario_name, shocks in shocks_supply_voll.items():
        process_supply_shocks_with_voll(iso3, scenario_name+'_survey_approach', shocks)

    shocks_demand = get_demand_side_scenario_shocks_population()
    for scenario_name, shock in shocks_demand.items():
        process_demand_shocks_population(iso3, scenario_name+'_population_approach', shock)

    shocks_demand_survey = get_demand_side_scenario_shocks_survey_voll()
    for scenario_name, shock in shocks_demand_survey.items():
        process_demand_shocks_population(iso3, scenario_name+'_survey_voll_approach', shock)
