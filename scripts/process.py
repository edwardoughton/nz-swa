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

MODEL_RESULT_DEFINITIONS = [
    {
        'method_id': 'demand_population',
        'method_label': 'Demand-Side Leontief (Population-Weighted Shock)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_population_approach.csv',
    },
    {
        'method_id': 'demand_survey_voll',
        'method_label': 'Demand-Side Leontief (Survey-Based VoLL Shock)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_survey_voll_approach.csv',
    },
    {
        'method_id': 'supply_percent_shock',
        'method_label': 'Supply-Side Ghosh (Employment-Weighted Shock)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_employment_approach.csv',
    },
    {
        'method_id': 'supply_customer_class_voll',
        'method_label': 'Supply-Side Ghosh (Survey-Based VoLL Shock)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_survey_approach.csv',
    },
]

MITIGATION_COSTS_MILLION_NZD = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.25,
    5: 0.50,
    6: 24.75,
    7: 68.75,
}

MITIGATION_DESCRIPTIONS = {
    1: 'No mitigation investment',
    2: 'No mitigation investment',
    3: 'No mitigation investment; BCR baseline',
    4: 'Switching sequence',
    5: 'Switching sequence and islanding',
    6: 'Switching sequence, islanding, and 12 GIC blockers',
    7: 'Switching sequence, islanding, and 34 GIC blockers',
}

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
    B_matrix = transactions.div(total_output, axis=0).fillna(0)
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


def get_supply_side_scenario_shocks_survey_voll():
    """
    Estimate direct sector losses using employment disruption, sector electricity
    intensity, and Transpower customer-class VoLL by affected location.

    The resulting sector losses are in millions of NZD and are used as the
    exogenous value-added shock in the supply-side Ghosh model.
    """
    output = {}

    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')

    intensity_path = os.path.join(DATA_PROCESSED, 'NZL', 'electricity_intensity_per_employee_all_sectors.csv')
    class_voll_path = os.path.join(DATA_PROCESSED, 'NZL', 'customer_class_voll_lut.csv')
    sector_class_path = os.path.join(DATA_PROCESSED, 'NZL', 'sector_customer_class_lut.csv')

    missing_paths = [path for path in [intensity_path, class_voll_path, sector_class_path] if not os.path.exists(path)]
    if missing_paths:
        missing = ', '.join(missing_paths)
        raise FileNotFoundError(f'Missing VoLL input lookup(s): {missing}. Run scripts/voll.py first.')

    intensity_data = pd.read_csv(intensity_path).set_index('sector_name')
    sector_class_lut = pd.read_csv(sector_class_path).set_index('sector_name')
    customer_voll = pd.read_csv(class_voll_path)

    class_share_columns = {
        'agricultural': 'agricultural_share',
        'commercial': 'commercial_share',
        'industrial': 'industrial_share',
    }
    class_voll_columns = {
        'agricultural': 'agricultural_voll_nzd_mwh',
        'commercial': 'commercial_voll_nzd_mwh',
        'industrial': 'industrial_voll_nzd_mwh',
    }
    mean_share = {
        customer_class: customer_voll[share_column].mean()
        for customer_class, share_column in class_share_columns.items()
    }
    mean_voll = {
        customer_class: customer_voll[voll_column].mean()
        for customer_class, voll_column in class_voll_columns.items()
    }

    output_records = []

    for i in range(1, 8):
        filename = f"scenario{i}.csv"
        data = pd.read_csv(os.path.join(folder, filename))

        if 'location3' not in data.columns:
            raise ValueError(f'{filename} does not contain location3 for Transpower VoLL merge')

        start_col = 'Accommodation'
        end_col = 'Wood product manufacturing'
        sectors = data.loc[:, start_col:end_col].columns

        sector_losses = {}

        for sector in sectors:
            if sector not in intensity_data.index:
                print(f"Warning: {sector} not found in intensity data, skipping.")
                continue
            if sector not in sector_class_lut.index:
                print(f"Warning: {sector} not found in customer class LUT, using commercial VoLL.")
                customer_class = 'commercial'
            else:
                customer_class = sector_class_lut.at[sector, 'customer_class']

            voll_column = class_voll_columns.get(customer_class, 'commercial_voll_nzd_mwh')
            share = mean_share.get(customer_class, mean_share['commercial'])
            default_voll = mean_voll.get(customer_class, mean_voll['commercial'])

            intensity = intensity_data.at[sector, 'GWh_per_employee']
            location_voll = customer_voll[['location3', voll_column]].copy()
            location_voll = location_voll.rename(columns={voll_column: 'class_voll_nzd_mwh'})
            merged = data.merge(location_voll, on='location3', how='left')

            total_voll = pd.to_numeric(merged.get('VoLL'), errors='coerce') if 'VoLL' in merged else pd.Series(np.nan, index=merged.index)
            class_voll = pd.to_numeric(merged['class_voll_nzd_mwh'], errors='coerce')
            class_voll = class_voll.fillna(total_voll * share).fillna(default_voll)

            employee_days = sum(
                pd.to_numeric(merged[sector], errors='coerce').fillna(0)
                * pd.to_numeric(merged[f'd{j}'], errors='coerce').fillna(0)
                for j in range(1, 7)
            )
            lost_mwh = employee_days * (intensity / 365) * 1e3
            direct_loss_nzd = lost_mwh * class_voll

            sector_losses[sector] = direct_loss_nzd.sum() / 1e6

            total_lost_mwh = lost_mwh.sum()
            weighted_voll = direct_loss_nzd.sum() / total_lost_mwh if total_lost_mwh else class_voll.mean()
            output_records.append({
                'scenario': filename[:-4],
                'sector_name': sector,
                'customer_class': customer_class,
                'lost_mwh': total_lost_mwh,
                'weighted_voll_nzd_mwh': weighted_voll,
                'direct_voll_loss_million_nzd': sector_losses[sector],
            })

        output[f"{filename[:-4]}"] = sector_losses

    voll_folder = os.path.join(DATA_PROCESSED, 'NZL', 'VoLL')
    os.makedirs(voll_folder, exist_ok=True)

    pd.DataFrame(output_records).sort_values(['scenario', 'sector_name']).to_csv(
        os.path.join(voll_folder, 'direct_losses_with_customer_class_voll_by_sector.csv'),
        index=False,
    )

    return output

def process_supply_shocks_with_voll(iso3, scenario_name, supply_shocks):
    """
    Estimates GDP loss using Ghosh model, where input supply_shocks are actual direct
    economic losses (in millions NZD) per sector â€” NOT percentage reductions.

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
    B_matrix = transactions.div(total_output, axis=0).fillna(0)
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
        f.write(f"Direct Transpower VoLL loss used as shock input: {round(combined['Direct VoLL Loss'].sum(), 2)} million {PRICE_YEAR_LABEL}\n")
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


def _read_total_loss_million_nzd(filename):
    """
    Read a result file and return total GDP loss in million 2026 NZD.
    """
    path_in = os.path.join(RESULTS, filename)
    if not os.path.exists(path_in):
        return np.nan
    data = pd.read_csv(path_in)
    if 'Loss' not in data.columns:
        return np.nan
    return data['Loss'].sum()


def export_benefit_cost_ratios(baseline_scenario=3):
    """
    Export benefit-cost ratios using avoided GDP loss relative to Scenario 3.

    Benefits and costs are both expressed in million 2026 NZD. Scenarios with no
    mitigation investment are retained in the output but have blank BCR values.
    """
    rows = []

    for method_order, method in enumerate(MODEL_RESULT_DEFINITIONS):
        baseline_filename = method['filename_template'].format(scenario=baseline_scenario)
        baseline_loss = _read_total_loss_million_nzd(baseline_filename)

        for scenario in range(1, 8):
            filename = method['filename_template'].format(scenario=scenario)
            scenario_loss = _read_total_loss_million_nzd(filename)
            cost = MITIGATION_COSTS_MILLION_NZD[scenario]
            avoided_loss = baseline_loss - scenario_loss if pd.notna(baseline_loss) and pd.notna(scenario_loss) else np.nan
            bcr = avoided_loss / cost if cost > 0 and pd.notna(avoided_loss) else np.nan

            rows.append({
                'scenario': f'Scenario {scenario}',
                'scenario_number': scenario,
                'method_order': method_order,
                'method_id': method['method_id'],
                'method': method['method_label'],
                'baseline_scenario': f'Scenario {baseline_scenario}',
                'mitigation_description': MITIGATION_DESCRIPTIONS[scenario],
                'mitigation_cost_million_2026_nzd': cost,
                'baseline_loss_million_2026_nzd': baseline_loss,
                'scenario_loss_million_2026_nzd': scenario_loss,
                'avoided_loss_million_2026_nzd': avoided_loss,
                'benefit_cost_ratio': bcr,
            })

    bcr = pd.DataFrame(rows)
    bcr = bcr.sort_values(['scenario_number', 'method_order']).reset_index(drop=True)

    long_out = os.path.join(RESULTS, 'benefit_cost_ratios_scenario3_baseline.csv')
    bcr.to_csv(long_out, index=False)

    wide = bcr.pivot(index='scenario', columns='method', values='benefit_cost_ratio').reset_index()
    wide['scenario_number'] = wide['scenario'].str.extract(r'(\d+)').astype(int)
    wide = wide.sort_values('scenario_number').drop(columns=['scenario_number'])
    wide_out = os.path.join(RESULTS, 'benefit_cost_ratios_scenario3_baseline_wide.csv')
    wide.to_csv(wide_out, index=False)

    return bcr
if __name__ == "__main__":

    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)

    iso3 = 'NZL'

    shocks_supply = get_supply_side_scenario_shocks_employment()
    for scenario_name, shocks in shocks_supply.items():
        process_supply_shocks_employment(iso3, scenario_name+'_employment_approach', shocks)

    shocks_supply_voll = get_supply_side_scenario_shocks_survey_voll()
    for scenario_name, shocks in shocks_supply_voll.items():
        process_supply_shocks_with_voll(iso3, scenario_name+'_survey_approach', shocks)
        #break

    shocks_demand = get_demand_side_scenario_shocks_population()
    for scenario_name, shock in shocks_demand.items():
        process_demand_shocks_population(iso3, scenario_name+'_population_approach', shock)

    shocks_demand_survey = get_demand_side_scenario_shocks_survey_voll()
    for scenario_name, shock in shocks_demand_survey.items():
        process_demand_shocks_population(iso3, scenario_name+'_survey_voll_approach', shock)

    export_benefit_cost_ratios()
