import argparse
import configparser
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
RESULTS = os.path.join(BASE_PATH, '..', 'results')

SCENARIOS = range(1, 8)
TOLERANCE_MILLION_NZD = 0.05

METHOD_DEFINITIONS = [
    {
        'method_id': 'demand_population',
        'method_label': 'Demand-Side Leontief (Population Shock)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_population_approach.csv',
        'summary_template': 'demand_side_summary_scenario{scenario}_population_approach.csv',
        'shock_template': 'demand_shock_factors_scenario{scenario}_population_approach.csv',
        'shock_type': 'demand_uniform_retained_share',
        'unit': 'million_2026_nzd',
    },
    {
        'method_id': 'demand_survey_voll',
        'method_label': 'Demand-Side Leontief (Survey-Based Residential VoLL)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_survey_voll_approach.csv',
        'summary_template': 'demand_side_summary_scenario{scenario}_survey_voll_approach.csv',
        'shock_template': 'demand_shock_factors_scenario{scenario}_survey_voll_approach.csv',
        'shock_type': 'demand_uniform_retained_share',
        'unit': 'million_2026_nzd',
    },
    {
        'method_id': 'supply_percent_shock',
        'method_label': 'Supply-Side Ghosh (% Shock)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_employment_approach.csv',
        'summary_template': 'gdp_loss_summary_scenario{scenario}_employment_approach.csv',
        'shock_template': 'shock_factors_scenario{scenario}_employment_approach.csv',
        'shock_type': 'sector_retained_value_added_share',
        'unit': 'million_2026_nzd',
    },
    {
        'method_id': 'supply_customer_class_voll',
        'method_label': 'Supply-Side Ghosh (Customer-Class Survey VoLL)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_survey_approach.csv',
        'summary_template': 'gdp_loss_summary_scenario{scenario}_survey_approach.csv',
        'shock_template': 'shock_factors_scenario{scenario}_survey_approach.csv',
        'shock_type': 'sector_retained_value_added_share',
        'unit': 'million_2026_nzd',
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_in:
        for chunk in iter(lambda: file_in.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _add_check(checks, check, status, severity='error', scenario='', method='', value='', expected='', details=''):
    checks.append({
        'scenario': scenario,
        'method': method,
        'check': check,
        'status': status,
        'severity': severity,
        'value': value,
        'expected': expected,
        'details': details,
    })


def _read_series_csv(path):
    data = pd.read_csv(path, index_col=0)
    if data.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(data.iloc[:, 0], errors='coerce')


def _read_summary_values(path):
    with open(path, 'r') as file_in:
        summary_text = file_in.read()
    return [float(v) for v in re.findall(r':\s*([-+]?\d*\.?\d+)\s+million', summary_text)]


def audit_scenario_inputs(checks):
    scenario_dir = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')
    required_columns = ['population', 'location3'] + [f'd{i}' for i in range(1, 7)]
    sector_start = 'Accommodation'
    sector_end = 'Wood product manufacturing'
    rows = []

    previous_disrupted_population_days = None
    for scenario in SCENARIOS:
        scenario_label = f'Scenario {scenario}'
        path = os.path.join(scenario_dir, f'scenario{scenario}.csv')
        if not os.path.exists(path):
            _add_check(checks, 'scenario_file_exists', 'FAIL', 'error', scenario_label, value=path)
            continue

        data = pd.read_csv(path)
        missing = [column for column in required_columns if column not in data.columns]
        _add_check(
            checks,
            'scenario_required_columns_exist',
            'PASS' if not missing else 'FAIL',
            'error',
            scenario_label,
            value=', '.join(missing),
            expected=', '.join(required_columns),
        )
        if missing:
            continue

        day_columns = [f'd{i}' for i in range(1, 7)]
        numeric_columns = ['population'] + day_columns
        negative_counts = {
            column: int((pd.to_numeric(data[column], errors='coerce') < 0).sum())
            for column in numeric_columns
        }
        negative_total = sum(negative_counts.values())
        _add_check(
            checks,
            'scenario_numeric_inputs_nonnegative',
            'PASS' if negative_total == 0 else 'FAIL',
            'error',
            scenario_label,
            value=negative_total,
            expected='0 negative values',
            details=str(negative_counts),
        )

        if sector_start in data.columns and sector_end in data.columns:
            sector_columns = data.loc[:, sector_start:sector_end].columns.tolist()
            sector_negative_total = int(
                (data[sector_columns].apply(pd.to_numeric, errors='coerce') < 0).sum().sum()
            )
            _add_check(
                checks,
                'scenario_sector_employee_inputs_nonnegative',
                'PASS' if sector_negative_total == 0 else 'FAIL',
                'error',
                scenario_label,
                value=sector_negative_total,
                expected='0 negative values',
            )
        else:
            sector_columns = []
            _add_check(
                checks,
                'scenario_sector_column_range_exists',
                'FAIL',
                'error',
                scenario_label,
                value=f'{sector_start}..{sector_end}',
                expected='sector employment columns',
            )

        population = pd.to_numeric(data['population'], errors='coerce').fillna(0)
        disrupted_population_days = sum(
            population * pd.to_numeric(data[column], errors='coerce').fillna(0)
            for column in day_columns
        ).sum()

        rows.append({
            'scenario': scenario_label,
            'input_file': path,
            'sha256': _sha256(path),
            'rows': len(data),
            'sector_columns': len(sector_columns),
            'population_sum': population.sum(),
            'disrupted_population_days': disrupted_population_days,
        })

        if previous_disrupted_population_days is not None:
            increased = disrupted_population_days > previous_disrupted_population_days + 1e-9
            _add_check(
                checks,
                'scenario_disrupted_population_days_nonincreasing',
                'WARN' if increased else 'PASS',
                'warning',
                scenario_label,
                value=round(float(disrupted_population_days), 6),
                expected=f'<= previous scenario ({previous_disrupted_population_days:.6f})',
                details='Reasonableness check only; inspect if scenario design intentionally breaks monotonicity.',
            )
        previous_disrupted_population_days = disrupted_population_days

    scenario_audit = pd.DataFrame(rows)
    scenario_audit.to_csv(os.path.join(RESULTS, 'validation_scenario_input_audit.csv'), index=False)
    return scenario_audit


def audit_shock_inputs(checks):
    rows = []

    for method in METHOD_DEFINITIONS:
        for scenario in SCENARIOS:
            scenario_label = f'Scenario {scenario}'
            filename = method['shock_template'].format(scenario=scenario)
            path = os.path.join(RESULTS, filename)
            if not os.path.exists(path):
                _add_check(checks, 'shock_factor_file_exists', 'FAIL', 'error', scenario_label, method['method_label'], value=filename)
                continue

            retained = _read_series_csv(path)
            retained_valid = retained.dropna()
            out_of_bounds = int(((retained_valid < -1e-12) | (retained_valid > 1 + 1e-12)).sum())
            _add_check(
                checks,
                'shock_factors_between_zero_and_one',
                'PASS' if out_of_bounds == 0 else 'FAIL',
                'error',
                scenario_label,
                method['method_label'],
                value=out_of_bounds,
                expected='0 out-of-bounds retained shares',
            )

            unique_count = int(retained_valid.round(12).nunique())
            if method['shock_type'] == 'demand_uniform_retained_share':
                _add_check(
                    checks,
                    'demand_shock_is_uniform_scalar',
                    'PASS' if unique_count == 1 else 'FAIL',
                    'error',
                    scenario_label,
                    method['method_label'],
                    value=unique_count,
                    expected='1 unique retained share',
                )

            rows.append({
                'scenario': scenario_label,
                'method_id': method['method_id'],
                'method': method['method_label'],
                'shock_file': filename,
                'shock_type': method['shock_type'],
                'retained_share_min': retained_valid.min(),
                'retained_share_mean': retained_valid.mean(),
                'retained_share_max': retained_valid.max(),
                'shock_percent_mean': (1 - retained_valid.mean()) * 100,
                'unique_retained_values_rounded_12dp': unique_count,
            })

    shock_audit = pd.DataFrame(rows)
    shock_audit.to_csv(os.path.join(RESULTS, 'validation_shock_input_audit.csv'), index=False)
    return shock_audit


def validate_result_files(checks, tolerance):
    totals = []
    required_columns = [
        'Original Output',
        'Shocked Output',
        'Output Loss',
        'Value Added to Output Ratio',
        'Loss',
        'Direct Loss',
        'Indirect Loss',
    ]

    for method in METHOD_DEFINITIONS:
        previous_total = None
        for scenario in SCENARIOS:
            scenario_label = f'Scenario {scenario}'
            filename = method['filename_template'].format(scenario=scenario)
            summary_filename = method['summary_template'].format(scenario=scenario)
            path = os.path.join(RESULTS, filename)
            summary_path = os.path.join(RESULTS, summary_filename)

            if not os.path.exists(path):
                _add_check(checks, 'results_file_exists', 'FAIL', 'error', scenario_label, method['method_label'], value=filename)
                continue

            data = pd.read_csv(path)
            missing = [column for column in required_columns if column not in data.columns]
            _add_check(
                checks,
                'result_required_columns_exist',
                'PASS' if not missing else 'FAIL',
                'error',
                scenario_label,
                method['method_label'],
                value=', '.join(missing),
                expected=', '.join(required_columns),
            )
            if missing:
                continue

            output_loss_error = (
                data['Output Loss'] - (data['Original Output'] - data['Shocked Output']).clip(lower=0)
            ).abs().max()
            gdp_conversion_error = (
                data['Loss'] - data['Output Loss'] * data['Value Added to Output Ratio']
            ).abs().max()
            indirect_formula_error = (
                data['Indirect Loss'] - (data['Loss'] - data['Direct Loss'])
            ).abs().max()
            per_sector_error = (data['Loss'] - (data['Direct Loss'] + data['Indirect Loss'])).abs().max()

            for check_name, error in [
                ('output_loss_is_baseline_minus_post_shock_million_nzd', output_loss_error),
                ('gdp_loss_uses_value_added_output_ratio_million_nzd', gdp_conversion_error),
                ('indirect_loss_is_total_gdp_minus_direct_gdp_million_nzd', indirect_formula_error),
                ('per_sector_total_consistency_million_nzd', per_sector_error),
            ]:
                _add_check(
                    checks,
                    check_name,
                    'PASS' if error <= tolerance else 'FAIL',
                    'error',
                    scenario_label,
                    method['method_label'],
                    value=round(float(error), 9),
                    expected=f'<= {tolerance}',
                )

            negative_loss_cells = int((data[['Loss', 'Direct Loss', 'Indirect Loss']] < -tolerance).sum().sum())
            _add_check(
                checks,
                'loss_columns_nonnegative',
                'PASS' if negative_loss_cells == 0 else 'FAIL',
                'error',
                scenario_label,
                method['method_label'],
                value=negative_loss_cells,
                expected='0 negative loss cells',
            )

            direct = float(data['Direct Loss'].sum())
            indirect = float(data['Indirect Loss'].sum())
            total = float(data['Loss'].sum())
            totals.append({
                'scenario': scenario_label,
                'scenario_number': scenario,
                'method_id': method['method_id'],
                'method': method['method_label'],
                'direct_loss_million_2026_nzd': direct,
                'indirect_loss_million_2026_nzd': indirect,
                'total_loss_million_2026_nzd': total,
            })

            if previous_total is not None:
                increased = total > previous_total + tolerance
                _add_check(
                    checks,
                    'total_loss_nonincreasing_by_scenario',
                    'WARN' if increased else 'PASS',
                    'warning',
                    scenario_label,
                    method['method_label'],
                    value=round(total, 6),
                    expected=f'<= previous scenario ({previous_total:.6f})',
                    details='Reasonableness check only; inspect if scenario design intentionally breaks monotonicity.',
                )
            previous_total = total

            if not os.path.exists(summary_path):
                _add_check(checks, 'summary_file_exists', 'FAIL', 'error', scenario_label, method['method_label'], value=summary_filename)
                continue

            values = _read_summary_values(summary_path)
            if len(values) < 3:
                _add_check(checks, 'summary_values_parseable', 'FAIL', 'error', scenario_label, method['method_label'], value=len(values), expected='3 values')
                continue

            for check_name, data_value, summary_value in [
                ('summary_direct_matches_data_million_nzd', direct, values[0]),
                ('summary_indirect_matches_data_million_nzd', indirect, values[1]),
                ('summary_total_matches_data_million_nzd', total, values[2]),
            ]:
                error = abs(data_value - summary_value)
                _add_check(
                    checks,
                    check_name,
                    'PASS' if error <= tolerance else 'FAIL',
                    'error',
                    scenario_label,
                    method['method_label'],
                    value=round(data_value, 9),
                    expected=round(summary_value, 9),
                    details=f'abs_error={error:.9f}',
                )

    totals_df = pd.DataFrame(totals)
    totals_df.to_csv(os.path.join(RESULTS, 'validation_model_totals.csv'), index=False)
    return totals_df


def validate_benefit_cost_ratios(checks, totals_df, tolerance):
    path = os.path.join(RESULTS, 'benefit_cost_ratios_scenario3_baseline.csv')
    if not os.path.exists(path):
        _add_check(checks, 'bcr_file_exists', 'WARN', 'warning', value=path)
        return

    bcr = pd.read_csv(path)
    required_columns = [
        'scenario_number',
        'method_id',
        'mitigation_cost_million_2026_nzd',
        'baseline_loss_million_2026_nzd',
        'scenario_loss_million_2026_nzd',
        'avoided_loss_million_2026_nzd',
        'benefit_cost_ratio',
    ]
    missing = [column for column in required_columns if column not in bcr.columns]
    _add_check(
        checks,
        'bcr_required_columns_exist',
        'PASS' if not missing else 'FAIL',
        'error',
        value=', '.join(missing),
        expected=', '.join(required_columns),
    )
    if missing:
        return

    totals_lookup = totals_df.set_index(['scenario_number', 'method_id'])['total_loss_million_2026_nzd'].to_dict()
    baseline_lookup = {
        method['method_id']: totals_lookup.get((3, method['method_id']), np.nan)
        for method in METHOD_DEFINITIONS
    }

    for _, row in bcr.iterrows():
        scenario = int(row['scenario_number'])
        method_id = row['method_id']
        scenario_label = f'Scenario {scenario}'
        method_label = row.get('method', method_id)
        scenario_loss = totals_lookup.get((scenario, method_id), np.nan)
        baseline_loss = baseline_lookup.get(method_id, np.nan)
        avoided = baseline_loss - scenario_loss
        cost = float(row['mitigation_cost_million_2026_nzd'])
        expected_bcr = avoided / cost if cost > 0 else np.nan

        _add_check(
            checks,
            'bcr_scenario_loss_matches_results',
            'PASS' if abs(float(row['scenario_loss_million_2026_nzd']) - scenario_loss) <= tolerance else 'FAIL',
            'error',
            scenario_label,
            method_label,
            value=round(float(row['scenario_loss_million_2026_nzd']), 9),
            expected=round(float(scenario_loss), 9),
        )
        _add_check(
            checks,
            'bcr_baseline_loss_matches_scenario3_results',
            'PASS' if abs(float(row['baseline_loss_million_2026_nzd']) - baseline_loss) <= tolerance else 'FAIL',
            'error',
            scenario_label,
            method_label,
            value=round(float(row['baseline_loss_million_2026_nzd']), 9),
            expected=round(float(baseline_loss), 9),
        )
        _add_check(
            checks,
            'bcr_avoided_loss_formula',
            'PASS' if abs(float(row['avoided_loss_million_2026_nzd']) - avoided) <= tolerance else 'FAIL',
            'error',
            scenario_label,
            method_label,
            value=round(float(row['avoided_loss_million_2026_nzd']), 9),
            expected=round(float(avoided), 9),
        )

        actual_bcr = row['benefit_cost_ratio']
        if cost == 0:
            status = 'PASS' if pd.isna(actual_bcr) else 'FAIL'
            expected = 'blank for zero mitigation cost'
            value = actual_bcr
        else:
            status = 'PASS' if abs(float(actual_bcr) - expected_bcr) <= tolerance else 'FAIL'
            expected = round(float(expected_bcr), 9)
            value = round(float(actual_bcr), 9)
        _add_check(checks, 'bcr_ratio_formula', status, 'error', scenario_label, method_label, value=value, expected=expected)


def _markdown_table(data):
    if data.empty:
        return ''

    frame = data.copy()
    frame = frame.fillna('')
    columns = list(frame.columns)
    rows = []
    rows.append('| ' + ' | '.join(columns) + ' |')
    rows.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')

    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f'{value:.6f}'
            value = str(value).replace('|', '\\|').replace('\n', ' ')
            values.append(value)
        rows.append('| ' + ' | '.join(values) + ' |')

    return '\n'.join(rows)

def write_markdown_report(checks_df, scenario_audit, shock_audit, totals_df):
    path = os.path.join(RESULTS, 'validation_report.md')
    status_counts = checks_df.groupby(['severity', 'status']).size().reset_index(name='count')
    hard_failures = checks_df[(checks_df['severity'] == 'error') & (checks_df['status'] == 'FAIL')]
    warnings = checks_df[checks_df['status'] == 'WARN']

    lines = [
        '# Validation Report',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat()}',
        '',
        '## Status Counts',
        '',
        _markdown_table(status_counts),
        '',
        '## Hard Failures',
        '',
    ]
    if hard_failures.empty:
        lines.append('No hard failures.')
    else:
        lines.append(_markdown_table(hard_failures[['scenario', 'method', 'check', 'value', 'expected', 'details']]))

    lines.extend(['', '## Warnings', ''])
    if warnings.empty:
        lines.append('No warnings.')
    else:
        lines.append(_markdown_table(warnings[['scenario', 'method', 'check', 'value', 'expected', 'details']]))

    lines.extend([
        '',
        '## Scenario Input Audit',
        '',
        _markdown_table(scenario_audit[['scenario', 'rows', 'sector_columns', 'population_sum', 'disrupted_population_days']]),
        '',
        '## Shock Input Audit',
        '',
        _markdown_table(shock_audit[['scenario', 'method_id', 'retained_share_min', 'retained_share_mean', 'retained_share_max', 'shock_percent_mean']]),
        '',
        '## Model Totals',
        '',
        _markdown_table(totals_df[['scenario', 'method_id', 'direct_loss_million_2026_nzd', 'indirect_loss_million_2026_nzd', 'total_loss_million_2026_nzd']]),
        '',
    ])

    with open(path, 'w', encoding='utf-8') as file_out:
        file_out.write('\n'.join(lines))


def run_validation(tolerance):
    os.makedirs(RESULTS, exist_ok=True)
    checks = []
    scenario_audit = audit_scenario_inputs(checks)
    shock_audit = audit_shock_inputs(checks)
    totals_df = validate_result_files(checks, tolerance)
    validate_benefit_cost_ratios(checks, totals_df, tolerance)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(os.path.join(RESULTS, 'validation_report.csv'), index=False)
    write_markdown_report(checks_df, scenario_audit, shock_audit, totals_df)

    hard_failures = checks_df[(checks_df['severity'] == 'error') & (checks_df['status'] == 'FAIL')]
    warnings = checks_df[checks_df['status'] == 'WARN']
    print(f'Validation checks: {len(checks_df)}')
    print(f'Hard failures: {len(hard_failures)}')
    print(f'Warnings: {len(warnings)}')
    print(f'Wrote {os.path.join(RESULTS, "validation_report.csv")}')
    print(f'Wrote {os.path.join(RESULTS, "validation_report.md")}')

    return 1 if not hard_failures.empty else 0


def main():
    parser = argparse.ArgumentParser(description='Validate NZ-SWA model inputs and outputs.')
    parser.add_argument(
        '--tolerance-million-nzd',
        type=float,
        default=TOLERANCE_MILLION_NZD,
        help='Allowed absolute accounting error in million 2026 NZD.',
    )
    args = parser.parse_args()
    return run_validation(args.tolerance_million_nzd)


if __name__ == '__main__':
    sys.exit(main())
