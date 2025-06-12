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


def plot_sector_demand_costs():
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
        'combined_losses_scenario1.csv': 'Scenario 1',
        'combined_losses_scenario2.csv': 'Scenario 2',
        'combined_losses_scenario3.csv': 'Scenario 3',
        'combined_losses_scenario4.csv': 'Scenario 4',
        'combined_losses_scenario5.csv': 'Scenario 5',
        'combined_losses_scenario6.csv': 'Scenario 6',
        'combined_losses_scenario7.csv': 'Scenario 7',
    }

    folder = os.path.join(BASE_PATH, '..', 'results')
    filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]

    all_results = []

    for filename in filenames:
        filepath = os.path.join(folder, filename)
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
        ax.set_xlim(0, 0.35)

        for j, (d, idr) in enumerate(zip(direct, indirect)):
            total = d + idr
            ax.text(total + 0.01, j, f'{total:.2f}', va='center', fontsize=8)

    # Hide the unused 8th subplot and use it for the legend
    if len(scenarios) < len(axes):
        legend_ax = axes[len(scenarios)]
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='center', fontsize=10, frameon=False)

    # Hide any additional unused subplots
    for j in range(len(scenarios) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario', fontsize=16)
    fig.supxlabel('Lost GDP (Billions NZ$)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])

    plot_path = os.path.join(VIS, 'sector_costs_by_scenario_grid.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_aggregate_demand_costs(custom_labels=None):
    """
    Plot direct and indirect cost impacts with larger font sizes and optional custom x-axis labels.

    Parameters:
    custom_labels (dict): Optional mapping from original scenario labels to custom labels.

    """
    label_map = {
        'combined_losses_scenario1.csv': 'Scenario 1',
        'combined_losses_scenario2.csv': 'Scenario 2',
        'combined_losses_scenario3.csv': 'Scenario 3',
        'combined_losses_scenario4.csv': 'Scenario 4',
        'combined_losses_scenario5.csv': 'Scenario 5',
        'combined_losses_scenario6.csv': 'Scenario 6',
        'combined_losses_scenario7.csv': 'Scenario 7',
    }
    folder = os.path.join(BASE_PATH, '..', 'results')
    filenames = os.listdir(folder)

    sums = []

    for filename in filenames:
        data = pd.read_csv(os.path.join(folder, filename))

        direct_sum = data['Direct Loss'].sum() / 1e3
        indirect_sum = data['Indirect Loss'].sum() / 1e3

        label = label_map.get(filename, filename)
        if custom_labels and label in custom_labels:
            label = custom_labels[label]

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
    plt.xlabel('', fontsize=14)
    plt.title('Lost Direct and Indirect GDP by Scenario', fontsize=16)
    plt.legend(fontsize=12)

    # Annotate bars with total value
    for i, (direct, indirect) in enumerate(zip(sums_df['direct'], sums_df['indirect'])):
        total = direct + indirect
        plt.text(i, total + max(sums_df['direct'] + sums_df['indirect']) * 0.01,
                 f'{total:.3g}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save to VIS folder
    # plot_path = os.path.join(VIS, 'summary_plot.png')
    # plt.savefig(plot_path, dpi=300)


def plot_aggregate_supply_costs():
    """
    Plot direct and indirect cost impacts from supply-side CSV files with larger font sizes
    and custom x-axis labels defined within the function.
    """

    # Define custom labels for each scenario
    custom_labels = {
        "scenario1": "Scenario 1",
        "scenario2": "Scenario 2",
        "scenario3": "Scenario 3",
        "scenario4": "Scenario 4",
        "scenario5": "Scenario 5",
        "scenario6": "Scenario 6",
        "scenario7": "Scenario 7"
    }

    folder = RESULTS
    filenames = sorted([f for f in os.listdir(folder) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    sums = []

    for filename in filenames:
        file_path = os.path.join(folder, filename)
        data = pd.read_csv(file_path)

        direct_sum = data['Direct Loss'].sum() / 1e3  # Convert to billions
        indirect_sum = data['Indirect Loss'].sum() / 1e3

        scenario_key = filename.replace('gdp_loss_by_sector_', '').replace('.csv', '')
        label = custom_labels.get(scenario_key, scenario_key)

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
        label_text = f'NZ${total:.2f} Bn'
        plt.text(i, total + max(sums_df['direct'] + sums_df['indirect']) * 0.01,
                 label_text, ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save to VIS folder
    plot_path = os.path.join(VIS, 'supply_side_summary_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_sector_supply_costs():
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
        'gdp_loss_by_sector_scenario1.csv': 'Scenario 1',
        'gdp_loss_by_sector_scenario2.csv': 'Scenario 2',
        'gdp_loss_by_sector_scenario3.csv': 'Scenario 3',
        'gdp_loss_by_sector_scenario4.csv': 'Scenario 4',
        'gdp_loss_by_sector_scenario5.csv': 'Scenario 5',
        'gdp_loss_by_sector_scenario6.csv': 'Scenario 6',
        'gdp_loss_by_sector_scenario7.csv': 'Scenario 7',
    }

    filenames = sorted([f for f in os.listdir(RESULTS) if f.startswith('gdp_loss_by_sector_scenario') and f.endswith('.csv')])

    all_results = []

    for filename in filenames:

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

    fig.suptitle('Lost Direct and Indirect GDP by Industrial Sector and Scenario', fontsize=16)
    fig.supxlabel('Lost GDP (Billions NZ$)', fontsize=12)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])  # Leaves room for the suptitle

    plot_path = os.path.join(VIS, 'sector_supply_costs_by_scenario_grid.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    # plot_panel()

    # plot_outage_areas_1_to_2()

    # plot_outage_areas_3_to_7()

    # plot_aggregate_demand_costs()

    plot_aggregate_supply_costs()

    # plot_sector_supply_costs()