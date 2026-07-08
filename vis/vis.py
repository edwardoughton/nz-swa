import os
import sys
import configparser
import numpy as np
import pandas as pd
try:
    import geopandas as gpd
    import contextily as ctx
    from shapely import wkt
except ImportError:
    gpd = None
    ctx = None
    wkt = None

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import textwrap
import re

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
VIS = os.path.join(BASE_PATH, '..', 'vis', 'figures')
RESULTS = os.path.join(BASE_PATH, '..', 'results')

mpl.rcParams['font.family'] = 'Times New Roman'

METHOD_DEFINITIONS = [
    {
        'method_id': 'demand_population',
        'method_label': 'Demand-Side Leontief (Population Shock)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_population_approach.csv',
        'summary_template': 'demand_side_summary_scenario{scenario}_population_approach.csv',
    },
    {
        'method_id': 'demand_survey_voll',
        'method_label': 'Demand-Side Leontief (Survey-Based VoLL)',
        'filename_template': 'demand_side_gdp_loss_by_sector_scenario{scenario}_survey_voll_approach.csv',
        'summary_template': 'demand_side_summary_scenario{scenario}_survey_voll_approach.csv',
    },
    {
        'method_id': 'supply_percent_shock',
        'method_label': 'Supply-Side Ghosh (% Shock)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_employment_approach.csv',
        'summary_template': 'gdp_loss_summary_scenario{scenario}_employment_approach.csv',
    },
    {
        'method_id': 'supply_survey_voll',
        'method_label': 'Supply-Side Ghosh (Survey-Based VoLL)',
        'filename_template': 'gdp_loss_by_sector_scenario{scenario}_survey_approach.csv',
        'summary_template': 'gdp_loss_summary_scenario{scenario}_survey_approach.csv',
    },
]

def plot_grid_map_panel():
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


def calc_voll_panel_plot():
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
    axs[1][1].barh(sectors, data['VoLL_nzd_MWh'], color='orange')
    axs[1][1].set_xlabel('Value of Lost Load (NZ$/MWh)')
    axs[1][1].set_title('(D) Value of Lost Load (VoLL)')
    for i, v in enumerate(data['VoLL_nzd_MWh']):
        axs[1][1].text(v + 800, i, f'{v:,.0f}', va='center')

    axs[0][0].set_xlim(0, data['elec_consumption_gwh'].max() * 1.3)
    axs[0][1].set_xlim(0, data['ec_count'].max() * 1.3)
    axs[1][0].set_xlim(0, data['MWh_per_employee'].max() * 1.3)
    axs[1][1].set_xlim(0, data['VoLL_nzd_MWh'].max() * 1.3)

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


def _demand_leontief_survey_voll_files():
    return sorted(
        f for f in os.listdir(RESULTS)
        if f.startswith('demand_side_gdp_loss_by_sector_scenario')
        and f.endswith('_survey_voll_approach.csv')
    )


def _demand_leontief_survey_voll_labels():
    return {
        f'demand_side_gdp_loss_by_sector_scenario{i}_survey_voll_approach.csv': f'Scenario {i}'
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
    plt.ylabel('Lost GDP (Billions 2026 NZ$)', fontsize=14)
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


def _sector_loss_x_limit_million_nzd():
    filenames = (
        _demand_leontief_files()
        + _demand_leontief_survey_voll_files()
        + _supply_employment_files()
        + _supply_survey_files()
    )

    max_total = 0
    for filename in filenames:
        filepath = os.path.join(RESULTS, filename)
        if not os.path.exists(filepath):
            continue

        data = pd.read_csv(filepath)
        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum()
        max_total = max(max_total, (grouped['Direct Loss'] + grouped['Indirect Loss']).max())

    return (np.ceil(max_total / 100) * 100) + 75 if max_total > 0 else 75

def _plot_sector_costs(filenames, label_map, title, plot_filename):
    sector_labels = _sector_labels()

    all_results = []

    for filename in filenames:
        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum()
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    if not all_results:
        raise ValueError(f'No result files found for {plot_filename}')

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())
    x_limit = _sector_loss_x_limit_million_nzd()

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
        ax.set_xlim(0, x_limit)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + x_limit * 0.01, j + 0.05, f'{total:.1f}', va='center', fontsize=9)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=12, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Lost GDP (Millions 2026 NZ$)', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Leaves room for the suptitle

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
        'Lost Direct and Indirect GDP by Scenario (Demand-Side Leontief, Population Shock)',
        'demand_side_summary_plot_leontief_population.png'
    )


def plot_sector_demand_costs_population_leontief():
    """
    Plot sector direct and indirect losses for the demand-side Leontief model.
    """
    _plot_sector_costs(
        _demand_leontief_files(),
        _demand_leontief_labels(),
        'Lost Direct and Indirect GDP by Industrial Sector (Demand-Side Leontief, Population Shock)',
        'sector_demand_costs_leontief_population.png'
    )


def plot_aggregate_demand_costs_survey_voll_leontief():
    """
    Plot aggregate direct and indirect losses for the demand-side Leontief
    model with Survey-Based VoLL weighting.
    """
    _plot_aggregate_costs(
        _demand_leontief_survey_voll_files(),
        _demand_leontief_survey_voll_labels(),
        'Lost Direct and Indirect GDP by Scenario (Demand-Side Leontief, Survey-Based VoLL)',
        'demand_side_summary_plot_leontief_survey_voll.png'
    )


def plot_sector_demand_costs_survey_voll_leontief():
    """
    Plot sector direct and indirect losses for the demand-side Leontief model
    with Survey-Based VoLL weighting.
    """
    _plot_sector_costs(
        _demand_leontief_survey_voll_files(),
        _demand_leontief_survey_voll_labels(),
        'Lost Direct and Indirect GDP by Industrial Sector (Demand-Side Leontief, Survey-Based VoLL)',
        'sector_demand_costs_leontief_survey_voll.png'
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
    plt.ylabel('Lost GDP (Billions 2026 NZ$)', fontsize=14)
    plt.title('Lost Direct and Indirect GDP by Scenario (Supply-Side Ghosh, % Shock)', fontsize=16)
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
    plt.ylabel('Lost GDP (Billions 2026 NZ$)', fontsize=14)
    plt.title('Lost Direct and Indirect GDP by Scenario (Supply-Side Ghosh, Survey-Based VoLL)', fontsize=16)
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
        'Demand-Side Leontief (Population Shock)'
    ))
    rows.extend(_aggregate_loss_rows(
        _demand_leontief_survey_voll_files(),
        _demand_leontief_survey_voll_labels(),
        'Demand-Side Leontief (Survey-Based VoLL)'
    ))
    rows.extend(_aggregate_loss_rows(
        _supply_employment_files(),
        _supply_employment_labels(),
        'Supply-Side Ghosh (% Shock)'
    ))
    rows.extend(_aggregate_loss_rows(
        _supply_survey_files(),
        _supply_survey_labels(),
        'Supply-Side Ghosh (Survey-Based VoLL)'
    ))

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError('No result files found for aggregate model comparison')

    scenario_order = {f'Scenario {i}': i for i in range(1, 8)}
    model_order = {
        'Demand-Side Leontief (Population Shock)': 0,
        'Demand-Side Leontief (Survey-Based VoLL)': 1,
        'Supply-Side Ghosh (% Shock)': 2,
        'Supply-Side Ghosh (Survey-Based VoLL)': 3,
    }
    comparison['scenario_order'] = comparison['scenario'].map(scenario_order)
    comparison['model_order'] = comparison['model'].map(model_order)
    comparison = comparison.sort_values(['scenario_order', 'model_order']).reset_index(drop=True)

    y = []
    labels = []
    n_models = len(model_order)
    bar_spacing = 1.5
    group_gap = 1.2
    group_spacing = n_models * bar_spacing + group_gap
    label_map = {
        'Demand-Side Leontief (Population Shock)': 'Demand-Side Leontief (Population-Weighted Shock)',
        'Demand-Side Leontief (Survey-Based VoLL)': 'Demand-Side Leontief (Survey-Based VoLL Shock)',
        'Supply-Side Ghosh (% Shock)': 'Supply-Side Ghosh (Employment-Weighted Shock)',
        'Supply-Side Ghosh (Survey-Based VoLL)': 'Supply-Side Ghosh (Survey-Based VoLL Shock)',
    }
    for _, row in comparison.iterrows():
        y.append((row['scenario_order'] - 1) * group_spacing + row['model_order'] * bar_spacing)
        labels.append(label_map[row['model']])

    plt.figure(figsize=(15, 14))
    plt.barh(y, comparison['direct'], label='Direct')
    plt.barh(y, comparison['indirect'], left=comparison['direct'], label='Indirect')

    totals = comparison['direct'] + comparison['indirect']
    x_limit = totals.max() * 1.15
    for ypos, total in zip(y, totals):
        plt.text(total + totals.max() * 0.01, ypos,
                 f'${total:.2f} Bn', ha='left', va='center', fontsize=15)

    group_centers = [
        (i - 1) * group_spacing + ((n_models - 1) * bar_spacing) / 2
        for i in range(1, 8)
    ]
    plt.yticks(y, labels, fontsize=16, linespacing=1.15)
    plt.xticks(fontsize=16)
    plt.xlim(0, x_limit)
    plt.xlabel('Lost GDP (Billions 2026 NZ$)', fontsize=18)
    plt.title('Lost Direct and Indirect GDP by Scenario and Method', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()

    for boundary in [(i - 1) * group_spacing - group_gap / 3 for i in range(2, 8)]:
        plt.axhline(boundary, color='0.85', linewidth=0.8)

    ax = plt.gca()
    for center, scenario in zip(group_centers, [f'Scenario {i}' for i in range(1, 8)]):
        ax.text(-0.70, center, scenario, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(left=0.45, right=0.96)
    plt.savefig(os.path.join(VIS, 'aggregate_model_cost_comparison.png'), dpi=300)
    plt.close()


def plot_benefit_cost_ratios():
    """
    Plot benefit-cost ratios by scenario and modelling method.
    """
    path_in = os.path.join(RESULTS, 'benefit_cost_ratios_scenario3_baseline.csv')
    if not os.path.exists(path_in):
        raise FileNotFoundError(
            f'Benefit-cost ratio file not found at {path_in}. Run scripts/process.py or '
            'process.export_benefit_cost_ratios() first.'
        )

    data = pd.read_csv(path_in)
    data = data[data['scenario_number'].isin(range(1, 8))].copy()
    if data.empty:
        raise ValueError('No benefit-cost ratio rows found for plotting')

    method_aliases = {
        'Demand-Side Leontief (Survey-Based Residential VoLL)': 'Demand-Side Leontief (Survey-Based VoLL)',
        'Supply-Side Ghosh (Customer-Class Survey VoLL)': 'Supply-Side Ghosh (Survey-Based VoLL)',
    }
    data['plot_method'] = data['method'].replace(method_aliases)

    model_order = {
        'Demand-Side Leontief (Population Shock)': 0,
        'Demand-Side Leontief (Survey-Based VoLL)': 1,
        'Supply-Side Ghosh (% Shock)': 2,
        'Supply-Side Ghosh (Survey-Based VoLL)': 3,
    }
    label_map = {
        'Demand-Side Leontief (Population Shock)': 'Demand-Side Leontief (Population-Weighted Shock)',
        'Demand-Side Leontief (Survey-Based VoLL)': 'Demand-Side Leontief (Survey-Based VoLL Shock)',
        'Supply-Side Ghosh (% Shock)': 'Supply-Side Ghosh (Employment-Weighted Shock)',
        'Supply-Side Ghosh (Survey-Based VoLL)': 'Supply-Side Ghosh (Survey-Based VoLL Shock)',
    }

    data['model_order'] = data['plot_method'].map(model_order)
    unmapped = sorted(data.loc[data['model_order'].isna(), 'method'].unique())
    if unmapped:
        raise ValueError(f'Unmapped BCR method label(s): {unmapped}')

    data = data.sort_values(['scenario_number', 'model_order']).reset_index(drop=True)
    data['plot_bcr'] = data['benefit_cost_ratio'].fillna(0)
    data['has_mitigation'] = data['mitigation_cost_million_2026_nzd'] > 0

    n_models = len(model_order)
    bar_spacing = 1.5
    group_gap = 1.2
    group_spacing = n_models * bar_spacing + group_gap

    y = []
    labels = []
    for _, row in data.iterrows():
        scenario_index = row['scenario_number'] - 1
        y.append(scenario_index * group_spacing + row['model_order'] * bar_spacing)
        labels.append(label_map[row['plot_method']])

    colors = np.where(data['has_mitigation'], '#2ca02c', '0.82')

    plt.figure(figsize=(15, 14))
    plt.barh(y, data['plot_bcr'], color=colors)
    plt.axvline(1, color='0.25', linewidth=1.2, linestyle='--')
    plt.text(
        0.66, 0.78,
        'Benefit-Cost Ratios > 1\nindicate benefits\noutweigh costs',
        transform=plt.gca().transAxes,
        ha='center',
        va='top',
        multialignment='center',
        fontsize=18,
        color='0.25',
        bbox={
            'boxstyle': 'square,pad=0.35',
            'facecolor': 'white',
            'edgecolor': 'black',
            'linewidth': 1.0,
        },
    )

    finite_bcr = data.loc[data['has_mitigation'], 'benefit_cost_ratio']
    max_bcr = finite_bcr.max() if not finite_bcr.empty else 1
    x_limit = max_bcr * 1.24
    no_investment_x = max_bcr * 0.01

    for ypos, (_, row) in zip(y, data.iterrows()):
        if row['has_mitigation']:
            bcr = row['benefit_cost_ratio']
            label = f'{bcr:.1f}' if abs(bcr) < 100 else f'{bcr:.0f}'
            plt.text(bcr + max_bcr * 0.01, ypos, label, ha='left', va='center', fontsize=15)
        else:
            plt.text(no_investment_x + 5, ypos, 'No mitigation investment',
                     ha='left', va='center', fontsize=16, color='0')

    group_centers = [
        (i - 1) * group_spacing + ((n_models - 1) * bar_spacing) / 2
        for i in range(1, 8)
    ]
    plt.yticks(y, labels, fontsize=16, linespacing=1.15)
    plt.xticks(fontsize=16)
    plt.xlim(0, x_limit)
    plt.xlabel('Benefit-Cost Ratio (Avoided GDP Loss / Mitigation Cost)', fontsize=18)
    plt.title('Benefit-Cost Ratios by Scenario and Method', fontsize=20)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()

    for boundary in [(i - 1) * group_spacing - group_gap / 3 for i in range(2, 8)]:
        plt.axhline(boundary, color='0.85', linewidth=0.8)

    ax = plt.gca()
    for center, scenario in zip(group_centers, [f'Scenario {i}' for i in range(1, 8)]):
        ax.text(-0.70, center, scenario, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(left=0.45, right=0.96)
    plt.savefig(os.path.join(VIS, 'benefit_cost_ratios_scenario3_baseline.png'), dpi=300)
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
        'gdp_loss_by_sector_scenario1_employment_approach.csv': 'Scenario 1',
        'gdp_loss_by_sector_scenario2_employment_approach.csv': 'Scenario 2',
        'gdp_loss_by_sector_scenario3_employment_approach.csv': 'Scenario 3',
        'gdp_loss_by_sector_scenario4_employment_approach.csv': 'Scenario 4',
        'gdp_loss_by_sector_scenario5_employment_approach.csv': 'Scenario 5',
        'gdp_loss_by_sector_scenario6_employment_approach.csv': 'Scenario 6',
        'gdp_loss_by_sector_scenario7_employment_approach.csv': 'Scenario 7',
    }

    filenames = sorted([f for f in os.listdir(RESULTS) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    all_results = []

    for filename in filenames:

        if not 'employment_approach' in filename:
            continue

        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum()
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())
    x_limit = _sector_loss_x_limit_million_nzd()

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
        ax.set_xlim(0, x_limit)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + x_limit * 0.01, j + 0.05, f'{total:.1f}', va='center', fontsize=9)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=12, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario (Supply-Side Ghosh, Employment Shock)', fontsize=16)
    fig.supxlabel('Lost GDP (Millions 2026 NZ$)', fontsize=12)
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
        'gdp_loss_by_sector_scenario1_survey_approach.csv': 'Scenario 1',
        'gdp_loss_by_sector_scenario2_survey_approach.csv': 'Scenario 2',
        'gdp_loss_by_sector_scenario3_survey_approach.csv': 'Scenario 3',
        'gdp_loss_by_sector_scenario4_survey_approach.csv': 'Scenario 4',
        'gdp_loss_by_sector_scenario5_survey_approach.csv': 'Scenario 5',
        'gdp_loss_by_sector_scenario6_survey_approach.csv': 'Scenario 6',
        'gdp_loss_by_sector_scenario7_survey_approach.csv': 'Scenario 7',
    }

    filenames = sorted([f for f in os.listdir(RESULTS) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    all_results = []

    for filename in filenames:

        if not 'survey_approach' in filename:
            continue

        filepath = os.path.join(RESULTS, filename)
        data = pd.read_csv(filepath)

        data['SectorInitial'] = data['NZSIOC'].str[0]
        grouped = data.groupby('SectorInitial')[['Direct Loss', 'Indirect Loss']].sum()
        grouped = grouped.reset_index()
        grouped['Scenario'] = label_map.get(filename, filename)

        all_results.append(grouped)

    result_df = pd.concat(all_results)
    scenarios = sorted(result_df['Scenario'].unique())
    x_limit = _sector_loss_x_limit_million_nzd()

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
        ax.set_xlim(0, x_limit)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + x_limit * 0.01, j + 0.05, f'{total:.1f}', va='center', fontsize=9)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=12, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario (Supply-Side Ghosh, Survey-Based VoLL)', fontsize=16)
    fig.supxlabel('Lost GDP (Millions 2026 NZ$)', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Leaves room for the suptitle

    plot_path = os.path.join(VIS, 'sector_supply_costs_voll_survey.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def export_four_method_comparison_table():
    """
    Export compact long and wide tables comparing all four methods.
    """
    rows = []

    for method in METHOD_DEFINITIONS:
        for scenario in range(1, 8):
            filename = method['filename_template'].format(scenario=scenario)
            path_in = os.path.join(RESULTS, filename)
            if not os.path.exists(path_in):
                continue

            data = pd.read_csv(path_in)
            direct = data['Direct Loss'].sum() / 1e3
            indirect = data['Indirect Loss'].sum() / 1e3
            total = data['Loss'].sum() / 1e3

            rows.append({
                'Scenario': f'Scenario {scenario}',
                'Method': method['method_label'],
                'Direct Loss (Bn 2026 NZD)': round(direct, 6),
                'Indirect Loss (Bn 2026 NZD)': round(indirect, 6),
                'Total Loss (Bn 2026 NZD)': round(total, 6),
            })

    if not rows:
        raise ValueError('No files found for four-method comparison export')

    long_df = pd.DataFrame(rows)
    scenario_order = {f'Scenario {i}': i for i in range(1, 8)}
    method_order = {m['method_label']: i for i, m in enumerate(METHOD_DEFINITIONS)}

    long_df['scenario_order'] = long_df['Scenario'].map(scenario_order)
    long_df['method_order'] = long_df['Method'].map(method_order)
    long_df = long_df.sort_values(['scenario_order', 'method_order']).drop(columns=['scenario_order', 'method_order'])

    long_out = os.path.join(RESULTS, 'method_comparison_four_methods_long.csv')
    long_df.to_csv(long_out, index=False)

    wide_df = long_df.pivot(index='Scenario', columns='Method', values='Total Loss (Bn 2026 NZD)').reset_index()
    wide_df['scenario_order'] = wide_df['Scenario'].map(scenario_order)
    wide_df = wide_df.sort_values('scenario_order').drop(columns=['scenario_order'])
    wide_out = os.path.join(RESULTS, 'method_comparison_four_methods_total_wide.csv')
    wide_df.to_csv(wide_out, index=False)


def export_manuscript_results_table():
    """
    Export a manuscript-ready compact table for the 4-method comparison.

    Values are total losses in billions of NZD and rounded to 3 decimals.
    """
    path_in = os.path.join(RESULTS, 'method_comparison_four_methods_total_wide.csv')
    if not os.path.exists(path_in):
        export_four_method_comparison_table()

    data = pd.read_csv(path_in)

    expected_columns = [
        'Scenario',
        'Demand-Side Leontief (Population Shock)',
        'Demand-Side Leontief (Survey-Based VoLL)',
        'Supply-Side Ghosh (% Shock)',
        'Supply-Side Ghosh (Survey-Based VoLL)',
    ]
    data = data[expected_columns].copy()

    rename_map = {
        'Demand-Side Leontief (Population Shock)': 'Leontief Demand (Population) [Bn 2026 NZD]',
        'Demand-Side Leontief (Survey-Based VoLL)': 'Leontief Demand (Survey-Based VoLL) [Bn 2026 NZD]',
        'Supply-Side Ghosh (% Shock)': 'Ghosh Supply (% Shock) [Bn 2026 NZD]',
        'Supply-Side Ghosh (Survey-Based VoLL)': 'Ghosh Supply (Survey-Based VoLL) [Bn 2026 NZD]',
    }
    data = data.rename(columns=rename_map)

    value_columns = [c for c in data.columns if c != 'Scenario']
    data[value_columns] = data[value_columns].round(3)

    path_out = os.path.join(RESULTS, 'manuscript_results_table_four_methods.csv')
    data.to_csv(path_out, index=False)


def validate_four_method_results(tolerance_million_nzd=0.05):
    """
    Validate accounting consistency for all scenarios and all four methods.

    Checks include:
    - Output Loss = Original Output - Shocked Output
    - Loss = Output Loss * value-added/output ratio
    - Indirect Loss = Loss - Direct Loss
    - Aggregate totals in dataframe sums vs summary text files
    """
    checks = []

    for method in METHOD_DEFINITIONS:
        for scenario in range(1, 8):
            filename = method['filename_template'].format(scenario=scenario)
            summary_filename = method['summary_template'].format(scenario=scenario)
            path_in = os.path.join(RESULTS, filename)
            summary_path = os.path.join(RESULTS, summary_filename)

            if not os.path.exists(path_in):
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'results_file_exists',
                    'Status': 'FAIL',
                    'Value': '',
                    'Expected': filename,
                    'Abs Error': '',
                })
                continue

            data = pd.read_csv(path_in)
            required_columns = [
                'Original Output',
                'Shocked Output',
                'Output Loss',
                'Value Added to Output Ratio',
                'Loss',
                'Direct Loss',
                'Indirect Loss',
            ]
            missing_columns = [c for c in required_columns if c not in data.columns]
            if missing_columns:
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'manuscript_accounting_columns_exist',
                    'Status': 'FAIL',
                    'Value': ', '.join(missing_columns),
                    'Expected': ', '.join(required_columns),
                    'Abs Error': '',
                })
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
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': check_name,
                    'Status': 'PASS' if error <= tolerance_million_nzd else 'FAIL',
                    'Value': round(float(error), 6),
                    'Expected': f'<= {tolerance_million_nzd}',
                    'Abs Error': round(float(error), 6),
                })

            data_direct = float(data['Direct Loss'].sum())
            data_indirect = float(data['Indirect Loss'].sum())
            data_total = float(data['Loss'].sum())

            if not os.path.exists(summary_path):
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'summary_file_exists',
                    'Status': 'FAIL',
                    'Value': '',
                    'Expected': summary_filename,
                    'Abs Error': '',
                })
                continue

            with open(summary_path, 'r') as f:
                summary_text = f.read()

            numeric_values = [float(v) for v in re.findall(r':\s*([-+]?\d*\.?\d+)\s+million', summary_text)]
            if len(numeric_values) >= 3:
                summary_direct, summary_indirect, summary_total = numeric_values[:3]

                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'summary_direct_matches_data_million_nzd',
                    'Status': 'PASS' if abs(data_direct - summary_direct) <= tolerance_million_nzd else 'FAIL',
                    'Value': round(data_direct, 6),
                    'Expected': round(summary_direct, 6),
                    'Abs Error': round(abs(data_direct - summary_direct), 6),
                })
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'summary_indirect_matches_data_million_nzd',
                    'Status': 'PASS' if abs(data_indirect - summary_indirect) <= tolerance_million_nzd else 'FAIL',
                    'Value': round(data_indirect, 6),
                    'Expected': round(summary_indirect, 6),
                    'Abs Error': round(abs(data_indirect - summary_indirect), 6),
                })
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'summary_total_matches_data_million_nzd',
                    'Status': 'PASS' if abs(data_total - summary_total) <= tolerance_million_nzd else 'FAIL',
                    'Value': round(data_total, 6),
                    'Expected': round(summary_total, 6),
                    'Abs Error': round(abs(data_total - summary_total), 6),
                })
            else:
                checks.append({
                    'Scenario': f'Scenario {scenario}',
                    'Method': method['method_label'],
                    'Check': 'summary_values_parseable',
                    'Status': 'FAIL',
                    'Value': '',
                    'Expected': '3 numeric values',
                    'Abs Error': '',
                })

    validation_df = pd.DataFrame(checks)
    validation_out = os.path.join(RESULTS, 'method_comparison_validation_report.csv')
    validation_df.to_csv(validation_out, index=False)

    summary = validation_df.groupby('Status').size().to_dict()
    print(f'Validation summary: {summary}')


if __name__ == "__main__":

    plot_grid_map_panel()

    plot_outage_areas_1_to_2()

    plot_outage_areas_3_to_7()

    calc_voll_panel_plot()

    plot_aggregate_model_cost_comparison()

    plot_benefit_cost_ratios()

    plot_sector_supply_costs_perc_shock()

    plot_sector_supply_costs_voll_survey()

    plot_sector_demand_costs_population_leontief()

    plot_sector_demand_costs_survey_voll_leontief()

    export_four_method_comparison_table()

    export_manuscript_results_table()

    validate_four_method_results()