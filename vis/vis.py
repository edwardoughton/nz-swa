import os
import sys
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import contextily as ctx
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from shapely import wkt
import textwrap

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
VIS = os.path.join(BASE_PATH, '..', 'vis', 'figures')
RESULTS = os.path.join(BASE_PATH, '..', 'results')

mpl.rcParams['font.family'] = 'Times New Roman'

def plot_panel():
    """
    Create a 1x4 panel plot to show the initial geographic context.

    - Subplot A: number of earthed substations from 'unique_substations.gpkg'
    """

    # Load substations
    filename = 'unique_substations.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    substations = gpd.read_file(path_in)

    # Load national outline shapefile
    outline_filename = 'national_outline.shp'
    outline_path = os.path.join(DATA_PROCESSED, 'NZL', outline_filename)
    national_outline = gpd.read_file(outline_path)

    # Match CRS
    substations = substations.to_crs(epsg=3857)
    national_outline = national_outline.to_crs(epsg=3857)

    # Create 1x4 subplot figure
    fig, axs = plt.subplots(1, 4, figsize=(14, 8), gridspec_kw={'wspace': 0.05})
    fig.patch.set_facecolor('#f0f0f0')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11.5,
        'legend.title_fontsize': 11.5
    })

    # Subplot A
    cmap = cm.plasma
    colors = [cmap(i) for i in [0.2, 0.8]]
    status_color_map = {'E': colors[0], 'No': colors[1]}
    label_map = {'E': 'Earthed', 'No': 'Not Earthed'}
    national_outline.plot(ax=axs[0], edgecolor='black', facecolor='none', linewidth=.7)
    for status in ['E', 'No']:
        group = substations[substations['Earthed'] == status]
        group.plot(ax=axs[0], color=status_color_map[status], label=label_map[status],
                   edgecolor='grey', linewidth=0.5, markersize=15)
    ctx.add_basemap(ax=axs[0], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[0].set_title('(A) Earthed Substations', loc='left', pad=0)
    axs[0].legend(title='Earthing', loc='lower right')
    axs[0].axis('off')

    # Subplot B
    path_in = os.path.join(folder, 'substation_counts.gpkg')
    counts = gpd.read_file(path_in)
    bins = [0, 1, 2, 3, 4, float('inf')]
    labels = ['1', '2', '3', '4', '>4']
    counts['count_cat'] = pd.cut(counts['count'], bins=bins, labels=labels, right=True)
    counts = counts.to_crs(epsg=3857)
    national_outline.plot(ax=axs[1], edgecolor='black', facecolor='none', linewidth=.7)
    counts = counts.sort_values(by='count_cat')
    counts.plot(
        ax=axs[1],
        column='count_cat',
        cmap='plasma',
        legend=True,
        legend_kwds={'title': 'Substation Count'},
        edgecolor='grey',
        linewidth=0.5,
        markersize=15,
    )
    colors = plt.cm.plasma(np.linspace(0, 1, len(labels)))
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    ctx.add_basemap(ax=axs[1], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[1].legend(handles=patches, title='Count', loc='lower right', frameon=True)
    axs[1].set_title('(B) Transformer Count', loc='left', pad=0)
    axs[1].axis('off')

    # Subplot C
    filename = 'population_by_node.gpkg'
    folder_in = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder_in, filename)
    population = gpd.read_file(path_in)
    population = population.to_crs(epsg=3857)
    pop_bins = [0, 5000, 10000, 50000, 100000, float('inf')]
    pop_labels = ['<5k', '<10k', '<50k', '<100k', '≥100k']
    population['pop_cat'] = pd.cut(population['population'], bins=pop_bins, labels=pop_labels, right=False)
    population = population.sort_values(by='pop_cat')
    national_outline.plot(ax=axs[2], edgecolor='black', facecolor='none', linewidth=.7)
    population.plot(
        ax=axs[2],
        column='pop_cat',
        cmap='plasma',
        legend=True,
        legend_kwds={'title': 'Population', 'loc': 'lower right'},
        edgecolor='grey',
        linewidth=0.5,
        markersize=15,
    )
    ctx.add_basemap(ax=axs[2], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[2].set_title('(C) Substation Population', loc='left', pad=0)
    axs[2].axis('off')

    # Subplot D
    path_in = os.path.join(BASE_PATH, 'processed', 'NZL', 'transmission_lines.gpkg')
    lines = gpd.read_file(path_in)
    lines = lines.to_crs(epsg=3857)
    voltages = [66, 110, 220]
    label_map = {66: '66 kV', 110: '110 kV', 220: '220 kV'}
    voltage_colors = {66: 'blue', 110: 'red', 220: 'orange'}
    national_outline.plot(ax=axs[3], edgecolor='black', facecolor='none', linewidth=.7)
    for voltage in voltages:
        subset = lines[lines['Voltage (kV)'] == voltage]
        subset.plot(
            ax=axs[3],
            color=voltage_colors[voltage],
            linewidth=1,
            label=label_map[voltage]
        )
    ctx.add_basemap(ax=axs[3], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[3].set_title('(D) Transmission Lines', loc='left', pad=0)
    axs[3].legend(title='Voltage', loc='lower right')
    axs[3].axis('off')

    plt.tight_layout(pad=1.0)
    if not os.path.exists(VIS):
        os.makedirs(VIS)

    path_out = os.path.join(VIS, 'panel_plot.png')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


def plot_outage_areas_1_to_2():
    """
    Create a 2 (scenarios) × 4 (days) panel of choropleth maps showing spatio-temporal 
    impacts of a blackout across SA2 regions in NZ.

    """
    # --- Base data ---
    # Load service areas
    filename = 'service_areas.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    service_areas = gpd.read_file(path_in)[['geometry']].to_crs(3857)
    service_areas = service_areas.reset_index().rename(columns={'index': 'service_area_id'})

    # Load SA2 polygons
    filename = 'statistical-area-2-2023-generalised.shp'
    path_in = os.path.join(DATA_RAW, filename)
    sa2 = gpd.read_file(path_in)
    exclude_list = ['258200', '259000', '259600']
    sa2 = sa2[~sa2['SA22023_V1'].isin(exclude_list)]
    sa2 = sa2.to_crs(3857)
    sa2 = sa2.reset_index().rename(columns={'index': 'sa2_id'})

    # Load national outline shapefile (assuming WGS84 or similar CRS)
    outline_filename = 'national_outline.shp'
    outline_path = os.path.join(DATA_PROCESSED, 'NZL', outline_filename)
    national_outline = gpd.read_file(outline_path)
    national_outline = national_outline.to_crs(3857)

    # Prepare plotting
    selected_days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']   #figsize=(8.5, 6.5), gridspec_kw={'bottom': 0.05}
    fig, axes = plt.subplots(nrows=2, ncols=len(selected_days), figsize=(6.5, 3), gridspec_kw={'bottom': 0.1}) #figsize=(7, 8)
    fig.suptitle("Spatio-temporal Power Outage Restoration for Scenarios 1 and 2", fontsize=11, y=1.04)
    fig.patch.set_facecolor('#f0f0f0')

    # Load scenarios 1–7
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')
    filenames = sorted([
        f for f in os.listdir(folder)
        if f.startswith(('scenario1', 'scenario2')) and f.endswith('.csv')
    ])

    for row, filename in enumerate(filenames):
        file_path = os.path.join(folder, filename)
        data = pd.read_csv(file_path)[['geometry', 'd1','d2','d3','d4','d5','d6','d7']] #get restoration data
        data['geometry'] = data['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
        gdf = gdf.to_crs(3857)
        joined = gpd.sjoin_nearest(service_areas, gdf,  how='left', distance_col='distance') #join service areas and data
        joined['geometry'] = joined['geometry'].representative_point() #convert from polygons to points
        point_data = joined[['service_area_id', 'd1','d2','d3','d4','d5','d6','d7']].drop_duplicates('service_area_id')
        enriched_service_areas = service_areas.merge(point_data, on='service_area_id', how='left') #merge data

        # Match service areas to SA2 centroids
        sa2_centroids = sa2.copy()
        sa2_centroids['geometry'] = sa2_centroids.centroid
        sa2_join = gpd.sjoin(sa2_centroids, enriched_service_areas, predicate='within', how='left')
        values = sa2_join[['sa2_id', 'd1','d2','d3','d4','d5','d6','d7']]
        sa2_with_data = sa2.merge(values, on='sa2_id', how='left')
        sa2_with_data = sa2_with_data[sa2_with_data['SA22023__1'] != 'Oceanic Canterbury Region'] #exclude

        for col, day in enumerate(selected_days):
            ax = axes[row, col]
            sa2_with_data.plot(
                column=day,
                ax=ax,
                cmap=ListedColormap(['lightgrey', 'darkred']),
                edgecolor='darkgrey',
                linewidth=0.1,
                legend=False,
                vmin=0,
                vmax=1
            )
            national_outline.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=.25)
            ctx.add_basemap(ax=ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'Day {day[1]}', fontsize=7, pad=.1)
            if col == 0:
                ax.annotate(f'Scenario {row+1}', xy=(0, 0.5), xycoords='axes fraction',
                            va='center', ha='right', fontsize=8, rotation=90)

    # Define legend patches
    power_patch = mpatches.Patch(color='lightgrey', label='Power')
    no_power_patch = mpatches.Patch(color='darkred', label='No Power')

    # Add the legend to the bottom center
    fig.legend(
        handles=[no_power_patch, power_patch],
        loc='lower center',
        ncol=2,
        fontsize=8,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.subplots_adjust(bottom=0.15)
    path_out = os.path.join(VIS, 'outage_panel_1_to_2.png')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


def plot_outage_areas_3_to_7():
    """
    Create a 4 (scenarios) × 4 (days) panel of choropleth maps showing spatio-temporal 
    impacts of a blackout across SA2 regions in NZ. 

    """
    # --- Base data ---
    # Load service areas
    filename = 'service_areas.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    service_areas = gpd.read_file(path_in)[['geometry']].to_crs(3857)
    service_areas = service_areas.reset_index().rename(columns={'index': 'service_area_id'})

    # Load SA2 polygons
    filename = 'statistical-area-2-2023-generalised.shp'
    path_in = os.path.join(DATA_RAW, filename)
    sa2 = gpd.read_file(path_in)
    exclude_list = ['258200', '259000', '259600']
    sa2 = sa2[~sa2['SA22023_V1'].isin(exclude_list)]
    sa2 = sa2.to_crs(3857)
    sa2 = sa2.reset_index().rename(columns={'index': 'sa2_id'})

    # Load national outline shapefile (assuming WGS84 or similar CRS)
    outline_filename = 'national_outline.shp'
    outline_path = os.path.join(DATA_PROCESSED, 'NZL', outline_filename)
    national_outline = gpd.read_file(outline_path)
    national_outline = national_outline.to_crs(3857)

    # Prepare plotting
    selected_days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    fig, axes = plt.subplots(nrows=5, ncols=len(selected_days), figsize=(8.5, 6.5), gridspec_kw={'bottom': 0.05}) #figsize=(7.5, 5.5)
    # plt.subplots_adjust(hspace=0.5, wspace=0.5) #using tight_layout() overules this

    fig.suptitle("Spatio-temporal Power Outage Restoration for Scenarios 3 to 7", fontsize=14)
    fig.patch.set_facecolor('#f0f0f0')

    # Load scenarios 3–7
    folder = os.path.join(DATA_PROCESSED, 'NZL', 'scenarios')
    filenames = sorted([
        f for f in os.listdir(folder)
        if f.startswith((
            'scenario3', 
            'scenario4', 
            'scenario5', 'scenario6', 'scenario7'
            )) and f.endswith('.csv')
    ])#[:4]

    for row, filename in enumerate(filenames):
        file_path = os.path.join(folder, filename)
        data = pd.read_csv(file_path)[['geometry', 'd1','d2','d3','d4','d5','d6','d7']] #get restoration data
        data['geometry'] = data['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
        gdf = gdf.to_crs(3857)
        joined = gpd.sjoin_nearest(service_areas, gdf,  how='left', distance_col='distance') #join service areas and data
        joined['geometry'] = joined['geometry'].representative_point() #convert from polygons to points
        point_data = joined[['service_area_id', 'd1','d2','d3','d4','d5','d6','d7']].drop_duplicates('service_area_id')
        enriched_service_areas = service_areas.merge(point_data, on='service_area_id', how='left') #merge data
        # enriched_service_areas.to_file(os.path.join(VIS,'enriched_service_areas.gpkg'))
        # Match service areas to SA2 centroids
        sa2_centroids = sa2.copy()
        sa2_centroids['geometry'] = sa2_centroids.centroid
        sa2_join = gpd.sjoin(sa2_centroids, enriched_service_areas, predicate='within', how='left')
        values = sa2_join[['sa2_id', 'd1','d2','d3','d4','d5','d6','d7']]
        sa2_with_data = sa2.merge(values, on='sa2_id', how='left')
        sa2_with_data = sa2_with_data[sa2_with_data['SA22023__1'] != 'Oceanic Canterbury Region'] #exclude
        sa2_with_data = sa2_with_data.dropna()
        # sa2_with_data
        # sa2_with_data.to_file(os.path.join(VIS,'sa2_with_data.gpkg'))
        
        for col, day in enumerate(selected_days):
            ax = axes[row, col]
            sa2_with_data.plot(
                column=day,
                ax=ax,
                cmap=ListedColormap(['lightgrey', 'darkred']),
                edgecolor='darkgrey',
                linewidth=0.1,
                legend=False,
                vmin=0,
                vmax=1
            )
            national_outline.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=.25)
            ctx.add_basemap(ax=ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'Day {day[1]}', fontsize=10, pad=2)
            if col == 0:
                ax.annotate(f'Scenario {row+3}', xy=(0, 0.5), xycoords='axes fraction',
                            va='center', ha='right', fontsize=10, rotation=90)

    # Define legend patches
    power_patch = mpatches.Patch(color='lightgrey', label='Power')
    no_power_patch = mpatches.Patch(color='darkred', label='No Power')

    # Add the legend to the bottom center
    fig.legend(
        handles=[no_power_patch, power_patch],
        loc='lower center',
        ncol=2,
        fontsize=10,
        frameon=False
    )

    fig.tight_layout(pad=0.2, w_pad=0.01, h_pad=0.2)
    path_out = os.path.join(VIS, 'outage_panel_3_to_7.png')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


def calc_voll():
    """
    
    """
    filename = "electricity_intensity_per_employee_broad_categories.csv"
    path_in = os.path.join(BASE_PATH, 'processed', 'NZL', filename)
    data = pd.read_csv(path_in)

    # Clean and wrap long sector names to max 20 characters per line
    def wrap_label(label, width=22):
        return '\n'.join(textwrap.wrap(label, width))
    
    # Apply wrapped labels to a new column for plotting
    data['Wrapped_Category'] = data['Broad_Category'].apply(wrap_label)
    data = data.sort_values('Wrapped_Category')  # keep consistent y-axis

    custom_order = [
        wrap_label('Agriculture, Forestry, and Fishing'),
        wrap_label('Mining'),
        wrap_label('Food Processing'),
        wrap_label('Wood, Pulp, Paper and Printing'),
        wrap_label('Chemicals'),
        wrap_label('Basic Metals'),
        wrap_label('Other Minor Sectors'),
        wrap_label('Commercial'),
        wrap_label('Transport'),
    ][::-1]
    data['Wrapped_Category'] = pd.Categorical(
        data['Wrapped_Category'],
        categories=custom_order,
        ordered=True
    )
    data = data.sort_values('Wrapped_Category')
    data['MWh_per_employee'] = data['GWh_per_employee']*1000
    sectors = data['Wrapped_Category']
    fig, axs = plt.subplots(2, 2, figsize=(9, 8), sharey=True)
    
    # Panel A: Electricity Consumption
    axs[0][0].barh(sectors, data['elec_consumption_gwh'], color='skyblue')
    axs[0][0].set_xlabel('Electricity Consumption (GWh)')
    axs[0][0].set_title('(A) Electricity by Sector')
    for i, v in enumerate(data['elec_consumption_gwh']):
        axs[0][0].text(v + 100, i, f'{v:,.0f}', va='center')

    # Panel B: Employment Count
    axs[0][1].barh(sectors, data['ec_count'], color='lightgreen')
    axs[0][1].set_xlabel('Employment Count (Millions)')
    axs[0][1].set_title('(B) Employment Count by Sector')
    for i, v in enumerate(data['ec_count']):
        axs[0][1].text(v + 20000, i, f'{v/1e6:.2f}', va='center')

    # Panel C: Electricity Consumption per Employee
    axs[1][0].barh(sectors, data['MWh_per_employee'], color='salmon')
    axs[1][0].set_xlabel('Electricity Consumption (MWh/employee)')
    axs[1][0].set_title('(C) Electricity Consumption Per Employee')
    for i, v in enumerate(data['MWh_per_employee']):
        axs[1][0].text(v+2, i, f'{v:,.1f}', va='center')

    # Panel D: Value of Lost Load (VoLL)
    axs[1][1].barh(sectors, data['VoLL_usd_MWh'], color='orange')
    axs[1][1].set_xlabel('Value of Lost Load (NZ$/MWh)')
    axs[1][1].set_title('(D) Value of Lost Load (VoLL)')
    for i, v in enumerate(data['VoLL_usd_MWh']):
        axs[1][1].text(v + 800, i, f'{v:,.0f}', va='center')

    axs[0][0].set_xlim(0, data['elec_consumption_gwh'].max() * 1.3)
    axs[0][1].set_xlim(0, data['ec_count'].max() * 1.3)
    axs[1][0].set_xlim(0, data['MWh_per_employee'].max() * 1.3)
    axs[1][1].set_xlim(0, data['VoLL_usd_MWh'].max() * 1.3)

    plt.subplots_adjust(hspace=0.3)
    plot_path = os.path.join(VIS, 'voll.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def _sector_labels():
    return {
        'A': 'Agriculture, Forestry & Fishing',
        'B': 'Mining',
        'C': 'Manufacturing',
        'D': 'Electricity, Gas, Water & Waste',
        'E': 'Construction',
        'F': 'Wholesale Trade',
        'G': 'Retail Trade',
        'H': 'Accommodation & Food Services',
        'I': 'Transport, Postal & Warehousing',
        'J': 'Information Media & Telecoms',
        'K': 'Financial & Insurance Services',
        'L': 'Rental, Hiring & Real Estate',
        'M': 'Professional, Scientific & Technical',
        'N': 'Administrative & Support Services',
        'O': 'Public Administration & Safety',
        'P': 'Education & Training',
        'Q': 'Health Care & Social Assistance',
        'R': 'Arts & Recreation Services',
        'S': 'Other Services'
    }


def _demand_leontief_files():
    return sorted(
        f for f in os.listdir(RESULTS)
        if f.startswith('demand_side_gdp_loss_by_sector_scenario')
        and f.endswith('_population_approach.csv')
    )


def _demand_leontief_labels():
    return {
        f'demand_side_gdp_loss_by_sector_scenario{i}_population_approach.csv': f'Scenario {i}'
        for i in range(1, 8)
    }


def _plot_aggregate_costs(filenames, label_map, title, plot_filename):
    sums = []

    for filename in filenames:
        data = pd.read_csv(os.path.join(RESULTS, filename))

        direct_sum = data['Direct Loss'].sum() / 1e3
        indirect_sum = data['Indirect Loss'].sum() / 1e3

        sums.append({
            'label': label_map.get(filename, filename),
            'direct': direct_sum,
            'indirect': indirect_sum
        })

    sums_df = pd.DataFrame(sums)
    if sums_df.empty:
        raise ValueError(f'No result files found for {plot_filename}')

    plt.figure(figsize=(12, 7))
    x = range(len(sums_df))
    plt.bar(x, sums_df['direct'], label='Direct')
    plt.bar(x, sums_df['indirect'], bottom=sums_df['direct'], label='Indirect')

    plt.xticks(x, sums_df['label'], rotation=30, ha='right', fontsize=12)
    plt.ylabel('Lost GDP (Billions NZ$)', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)

    totals = sums_df['direct'] + sums_df['indirect']
    for i, total in enumerate(totals):
        plt.text(i, total + totals.max() * 0.01,
                 f'${total:.2f} Bn', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(VIS, plot_filename), dpi=300)
    plt.close()


def _aggregate_loss_rows(filenames, label_map, model_name):
    rows = []

    for filename in filenames:
        data = pd.read_csv(os.path.join(RESULTS, filename))
        scenario_label = label_map.get(filename, filename)

        rows.append({
            'scenario': scenario_label,
            'model': model_name,
            'label': f'{scenario_label}\n{model_name}',
            'direct': data['Direct Loss'].sum() / 1e3,
            'indirect': data['Indirect Loss'].sum() / 1e3
        })

    return rows


def _supply_employment_files():
    return sorted(
        f for f in os.listdir(RESULTS)
        if f.startswith('gdp_loss_by_sector_scenario')
        and f.endswith('_employment_approach.csv')
    )


def _supply_survey_files():
    return sorted(
        f for f in os.listdir(RESULTS)
        if f.startswith('gdp_loss_by_sector_scenario')
        and f.endswith('_survey_approach.csv')
    )


def _supply_employment_labels():
    return {
        f'gdp_loss_by_sector_scenario{i}_employment_approach.csv': f'Scenario {i}'
        for i in range(1, 8)
    }


def _supply_survey_labels():
    return {
        f'gdp_loss_by_sector_scenario{i}_survey_approach.csv': f'Scenario {i}'
        for i in range(1, 8)
    }


def _plot_sector_costs(filenames, label_map, title, plot_filename):
    sector_labels = _sector_labels()

    all_results = []

    for filename in filenames:
        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum() / 1e3
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    if not all_results:
        raise ValueError(f'No result files found for {plot_filename}')

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())
    max_total = (result_df['Direct Loss'] + result_df['Indirect Loss']).max()
    x_limit = max_total * 1.25 if max_total > 0 else 1

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        df = result_df[result_df['Scenario'] == scenario].copy()
        df['Total Loss'] = df['Direct Loss'] + df['Indirect Loss']
        df = df.sort_values(by='Total Loss', ascending=False)

        sector_initials = df['SectorInitial'].tolist()
        y_labels = [sector_labels.get(s, s) for s in sector_initials]
        y = range(len(sector_initials))

        direct = df['Direct Loss']
        indirect = df['Indirect Loss']

        ax.barh(y, direct, label='Direct', color='tab:blue')
        ax.barh(y, indirect, left=direct, label='Indirect', color='tab:orange')
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.set_title(scenario)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1.75)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + x_limit * 0.01, j, f'{total:.2f}', va='center', fontsize=8)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=10, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Lost GDP (Billions NZ$)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])

    plot_path = os.path.join(VIS, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_aggregate_demand_costs_population_leontief():
    """
    Plot aggregate direct and indirect losses for the demand-side Leontief model.
    """
    _plot_aggregate_costs(
        _demand_leontief_files(),
        _demand_leontief_labels(),
        'Lost Direct and Indirect GDP by Scenario (Demand-Side Leontief)',
        'demand_side_summary_plot_leontief_population.png'
    )


def plot_sector_demand_costs_population_leontief():
    """
    Plot sector direct and indirect losses for the demand-side Leontief model.
    """
    _plot_sector_costs(
        _demand_leontief_files(),
        _demand_leontief_labels(),
        'Lost Direct and Indirect GDP by Industrial Sector and Scenario (Demand-Side Leontief)',
        'sector_demand_costs_leontief_population.png'
    )


def plot_aggregate_demand_costs(custom_labels=None):
    """
    Backward-compatible wrapper for the demand-side Leontief aggregate plot.
    """
    plot_aggregate_demand_costs_population_leontief()


def plot_sector_demand_costs():
    """
    Backward-compatible wrapper for the demand-side Leontief sector plot.
    """
    plot_sector_demand_costs_population_leontief()


def plot_aggregate_supply_costs_perc_voll():
    """
    Plot direct and indirect cost impacts from supply-side CSV files with larger font sizes
    and custom x-axis labels defined within the function.
    """

    custom_labels = _supply_employment_labels()
    filenames = _supply_employment_files()

    sums = []

    for filename in filenames:
        file_path = os.path.join(RESULTS, filename)
        data = pd.read_csv(file_path)

        direct_sum = data['Direct Loss'].sum() / 1e3  # Convert to billions
        indirect_sum = data['Indirect Loss'].sum() / 1e3

        label = custom_labels.get(filename, filename)

        sums.append({
            'label': label,
            'direct': direct_sum,
            'indirect': indirect_sum
        })

    # Convert to DataFrame
    sums_df = pd.DataFrame(sums)

    # Plotting
    plt.figure(figsize=(12, 7))
    x = range(len(sums_df))
    bar1 = plt.bar(x, sums_df['direct'], label='Direct')
    bar2 = plt.bar(x, sums_df['indirect'], bottom=sums_df['direct'], label='Indirect')

    plt.xticks(x, sums_df['label'], rotation=30, ha='right', fontsize=12)
    plt.ylabel('Lost GDP (Billions NZ$)', fontsize=14)
    plt.title('Lost Direct and Indirect GDP by Scenario', fontsize=16)
    plt.legend(fontsize=12)

    # Annotate bars with total value
    for i, (direct, indirect) in enumerate(zip(sums_df['direct'], sums_df['indirect'])):
        total = direct + indirect
        label_text = f'${total:.2f} Bn'
        plt.text(i, total + max(sums_df['direct'] + sums_df['indirect']) * 0.01,
                 label_text, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save to VIS folder
    plot_path = os.path.join(VIS, 'supply_side_summary_plot_perc_voll.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_aggregate_supply_costs_tp_voll():
    """
    Plot direct and indirect cost impacts from supply-side CSV files with larger font sizes
    and custom x-axis labels defined within the function.
    """

    custom_labels = _supply_survey_labels()
    filenames = _supply_survey_files()

    sums = []

    for filename in filenames:
        file_path = os.path.join(RESULTS, filename)
        data = pd.read_csv(file_path)

        direct_sum = data['Direct Loss'].sum() / 1e3  # Convert to billions
        indirect_sum = data['Indirect Loss'].sum() / 1e3

        label = custom_labels.get(filename, filename)

        sums.append({
            'label': label,
            'direct': direct_sum,
            'indirect': indirect_sum
        })

    # Convert to DataFrame
    sums_df = pd.DataFrame(sums)

    # Plotting
    plt.figure(figsize=(12, 7))
    x = range(len(sums_df))
    bar1 = plt.bar(x, sums_df['direct'], label='Direct')
    bar2 = plt.bar(x, sums_df['indirect'], bottom=sums_df['direct'], label='Indirect')

    plt.xticks(x, sums_df['label'], rotation=30, ha='right', fontsize=12)
    plt.ylabel('Lost GDP (Billions NZ$)', fontsize=14)
    plt.title('Lost Direct and Indirect GDP by Scenario (Using Transpower VoLL Data)', fontsize=16)
    plt.legend(fontsize=12)

    # Annotate bars with total value
    for i, (direct, indirect) in enumerate(zip(sums_df['direct'], sums_df['indirect'])):
        total = direct + indirect
        label_text = f'${total:.2f} Bn'
        plt.text(i, total + max(sums_df['direct'] + sums_df['indirect']) * 0.01,
                 label_text, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save to VIS folder
    plot_path = os.path.join(VIS, 'supply_side_summary_plot_survey_voll.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_aggregate_model_cost_comparison():
    """
    Plot aggregate losses for the Leontief, Ghosh, and VoLL model approaches
    in one grouped scenario comparison.
    """
    rows = []
    rows.extend(_aggregate_loss_rows(
        _demand_leontief_files(),
        _demand_leontief_labels(),
        'Leontief Demand-Side Model'
    ))
    rows.extend(_aggregate_loss_rows(
        _supply_employment_files(),
        _supply_employment_labels(),
        'Ghosh Supply-Side Model'
    ))
    rows.extend(_aggregate_loss_rows(
        _supply_survey_files(),
        _supply_survey_labels(),
        'Survey-Based VoLL Model'
    ))

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError('No result files found for aggregate model comparison')

    scenario_order = {f'Scenario {i}': i for i in range(1, 8)}
    model_order = {
        'Leontief Demand-Side Model': 0,
        'Survey-Based VoLL Model': 1,
        'Ghosh Supply-Side Model': 2,
    }
    comparison['scenario_order'] = comparison['scenario'].map(scenario_order)
    comparison['model_order'] = comparison['model'].map(model_order)
    comparison = comparison.sort_values(['scenario_order', 'model_order']).reset_index(drop=True)

    y = []
    labels = []
    group_spacing = 3.35
    for _, row in comparison.iterrows():
        y.append((row['scenario_order'] - 1) * group_spacing + row['model_order'])
        labels.append(row['model'].replace(' Model', ''))

    plt.figure(figsize=(12, 12))
    plt.barh(y, comparison['direct'], label='Direct')
    plt.barh(y, comparison['indirect'], left=comparison['direct'], label='Indirect')

    totals = comparison['direct'] + comparison['indirect']
    x_limit = totals.max() * 1.12
    for ypos, total in zip(y, totals):
        plt.text(total + totals.max() * 0.015, ypos,
                 f'${total:.2f} Bn', ha='left', va='center', fontsize=13)

    group_centers = [(i - 1) * group_spacing + 1 for i in range(1, 8)]
    plt.yticks(y, labels, fontsize=13)
    plt.xticks(fontsize=14)
    plt.xlim(0, x_limit)
    plt.xlabel('Lost GDP (Billions NZ$)', fontsize=17)
    plt.title('Lost Direct and Indirect GDP by Scenario and Model', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()

    for boundary in [(i - 1) * group_spacing + 2.5 + ((group_spacing - 3) / 2) for i in range(2, 8)]:
        plt.axhline(boundary, color='0.85', linewidth=0.8)

    ax = plt.gca()
    for center, scenario in zip(group_centers, [f'Scenario {i}' for i in range(1, 8)]):
        ax.text(-0.22, center, scenario, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(left=0.24, right=0.96)
    plt.savefig(os.path.join(VIS, 'aggregate_model_cost_comparison.png'), dpi=300)
    plt.close()


def plot_sector_supply_costs_perc_shock():
    """
    Aggregate scenario results based on the first letter of the 'NZSIOC' column.
    For each scenario, sum 'Direct Loss' and 'Indirect Loss' by sector initial.
    Plot horizontal stacked bar charts per scenario in a 4x2 grid.
    Legend is placed in the unused bottom-right subplot.
    Sectors are sorted by total loss in each scenario.

    """
    sector_labels = {
        'A': 'Agriculture, Forestry & Fishing',
        'B': 'Mining',
        'C': 'Manufacturing',
        'D': 'Electricity, Gas, Water & Waste',
        'E': 'Construction',
        'F': 'Wholesale Trade',
        'G': 'Retail Trade',
        'H': 'Accommodation & Food Services',
        'I': 'Transport, Postal & Warehousing',
        'J': 'Information Media & Telecoms',
        'K': 'Financial & Insurance Services',
        'L': 'Rental, Hiring & Real Estate',
        'M': 'Professional, Scientific & Technical',
        'N': 'Administrative & Support Services',
        'O': 'Public Administration & Safety',
        'P': 'Education & Training',
        'Q': 'Health Care & Social Assistance',
        'R': 'Arts & Recreation Services',
        'S': 'Other Services'
    }
    
    label_map = {
        'gdp_loss_by_sector_scenario1_employment_approach.csv': 'Scenario 1 (% VoLL)',
        'gdp_loss_by_sector_scenario2_employment_approach.csv': 'Scenario 2 (% VoLL)',
        'gdp_loss_by_sector_scenario3_employment_approach.csv': 'Scenario 3 (% VoLL)',
        'gdp_loss_by_sector_scenario4_employment_approach.csv': 'Scenario 4 (% VoLL)',
        'gdp_loss_by_sector_scenario5_employment_approach.csv': 'Scenario 5 (% VoLL)',
        'gdp_loss_by_sector_scenario6_employment_approach.csv': 'Scenario 6 (% VoLL)',
        'gdp_loss_by_sector_scenario7_employment_approach.csv': 'Scenario 7 (% VoLL)',
    }

    filenames = sorted([f for f in os.listdir(RESULTS) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    all_results = []

    for filename in filenames:

        if not 'employment_approach' in filename:
            continue

        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum() / 1e3
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):

        ax = axes[i]
        df = result_df[result_df['Scenario'] == scenario].copy()
        df['Total Loss'] = df['Direct Loss'] + df['Indirect Loss']
        df = df.sort_values(by='Total Loss', ascending=False)

        sector_initials = df['SectorInitial'].tolist()
        y_labels = [sector_labels.get(s, s) for s in sector_initials]
        y = range(len(sector_initials))

        direct = df['Direct Loss']
        indirect = df['Indirect Loss']

        ax.barh(y, direct, label='Direct', color='tab:blue')
        ax.barh(y, indirect, left=direct, label='Indirect', color='tab:orange')
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.set_title(scenario)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1.75)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + 0.01, j + 0.05, f'{total:.2f}', va='center', fontsize=9)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=10, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario (% VoLL)', fontsize=16)
    fig.supxlabel('Lost GDP (Billions NZ$)', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Leaves room for the suptitle

    plot_path = os.path.join(VIS, 'sector_supply_costs_perc_shock.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_sector_supply_costs_voll_survey():
    """
    Aggregate scenario results based on the first letter of the 'NZSIOC' column.
    For each scenario, sum 'Direct Loss' and 'Indirect Loss' by sector initial.
    Plot horizontal stacked bar charts per scenario in a 4x2 grid.
    Legend is placed in the unused bottom-right subplot.
    Sectors are sorted by total loss in each scenario.

    """
    sector_labels = {
        'A': 'Agriculture, Forestry & Fishing',
        'B': 'Mining',
        'C': 'Manufacturing',
        'D': 'Electricity, Gas, Water & Waste',
        'E': 'Construction',
        'F': 'Wholesale Trade',
        'G': 'Retail Trade',
        'H': 'Accommodation & Food Services',
        'I': 'Transport, Postal & Warehousing',
        'J': 'Information Media & Telecoms',
        'K': 'Financial & Insurance Services',
        'L': 'Rental, Hiring & Real Estate',
        'M': 'Professional, Scientific & Technical',
        'N': 'Administrative & Support Services',
        'O': 'Public Administration & Safety',
        'P': 'Education & Training',
        'Q': 'Health Care & Social Assistance',
        'R': 'Arts & Recreation Services',
        'S': 'Other Services'
    }
    
    label_map = {
        'gdp_loss_by_sector_scenario1_survey_approach.csv': 'Scenario 1 (VoLL Survey)',
        'gdp_loss_by_sector_scenario2_survey_approach.csv': 'Scenario 2 (VoLL Survey)',
        'gdp_loss_by_sector_scenario3_survey_approach.csv': 'Scenario 3 (VoLL Survey)',
        'gdp_loss_by_sector_scenario4_survey_approach.csv': 'Scenario 4 (VoLL Survey)',
        'gdp_loss_by_sector_scenario5_survey_approach.csv': 'Scenario 5 (VoLL Survey)',
        'gdp_loss_by_sector_scenario6_survey_approach.csv': 'Scenario 6 (VoLL Survey)',
        'gdp_loss_by_sector_scenario7_survey_approach.csv': 'Scenario 7 (VoLL Survey)',
    }

    filenames = sorted([f for f in os.listdir(RESULTS) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    all_results = []

    for filename in filenames:

        if not 'survey_approach' in filename:
            continue

        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum() / 1e3
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):

        ax = axes[i]
        df = result_df[result_df['Scenario'] == scenario].copy()
        df['Total Loss'] = df['Direct Loss'] + df['Indirect Loss']
        df = df.sort_values(by='Total Loss', ascending=False)

        sector_initials = df['SectorInitial'].tolist()
        y_labels = [sector_labels.get(s, s) for s in sector_initials]
        y = range(len(sector_initials))

        direct = df['Direct Loss']
        indirect = df['Indirect Loss']

        ax.barh(y, direct, label='Direct', color='tab:blue')
        ax.barh(y, indirect, left=direct, label='Indirect', color='tab:orange')
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.set_title(scenario)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1.75)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + 0.01, j + 0.05, f'{total:.2f}', va='center', fontsize=9)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=10, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario (VoLL Survey)', fontsize=16)
    fig.supxlabel('Lost GDP (Billions NZ$)', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Leaves room for the suptitle

    plot_path = os.path.join(VIS, 'sector_supply_costs_voll_survey.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    # plot_panel()

    # plot_outage_areas_1_to_2()

    # plot_outage_areas_3_to_7()

    # calc_voll()

    plot_aggregate_model_cost_comparison()

    # # plot_aggregate_demand_costs_population_leontief()

    # # plot_aggregate_supply_costs_perc_voll()

    # # plot_aggregate_supply_costs_tp_voll()

    plot_sector_supply_costs_perc_shock()

    plot_sector_supply_costs_voll_survey()

    plot_sector_demand_costs_population_leontief()
