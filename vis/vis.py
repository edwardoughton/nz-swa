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
import matplotlib.colors as mcolors

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')
VIS = os.path.join(BASE_PATH, '..', 'vis', 'figures')

mpl.rcParams['font.family'] = 'Times New Roman'

def plot_panel():
    """
    Create a 2x2 panel plot to show the initial geographic context

    - Subplot A: number of earthed substations from 'unique_substations.gpkg'
    """

    # Load substations
    filename = 'unique_substations.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    substations = gpd.read_file(path_in)

    # Load national outline shapefile (assuming WGS84 or similar CRS)
    outline_filename = 'national_outline.shp'
    outline_path = os.path.join(DATA_PROCESSED, 'NZL', outline_filename)
    national_outline = gpd.read_file(outline_path)

    # Match CRS for plotting and web tiles
    substations = substations.to_crs(epsg=3857)
    national_outline = national_outline.to_crs(epsg=3857)

    # Create 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(7, 10), gridspec_kw={'hspace': 0.125, 'wspace': 0.1})
    fig.patch.set_facecolor('#f0f0f0')

    # Subplot A: Plot outline, substations and basemap
    # Get two discrete colors from the Plasma colormap
    cmap = cm.plasma
    colors = [cmap(i) for i in [0.2, 0.8]]  # Choose two well-separated points in the colormap
    status_color_map = {'E': colors[0], 'No': colors[1]}
    label_map = {'E': 'Earthed', 'No': 'Not Earthed'}
    national_outline.plot(ax=axs[0, 0], edgecolor='black', facecolor='none', linewidth=.7)
    for status in ['E', 'No']:
        group = substations[substations['Earthed'] == status]
        group.plot(ax=axs[0, 0], color=status_color_map[status], label=label_map[status], edgecolor='grey',
        linewidth=0.5, markersize=15)
    ctx.add_basemap(ax=axs[0, 0], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[0, 0].set_title('(A) Earthed Substations', loc='left', pad=0)
    axs[0, 0].legend(title='Earthing', loc='lower right')
    axs[0, 0].axis('off')

    # Subplot B: Plot substation count choropleth
    path_in = os.path.join(folder, 'substation_counts.gpkg')
    counts = gpd.read_file(path_in)
    bins = [0, 1, 2, 3, 4, float('inf')]
    labels = ['1', '2', '3', '4', '>4']
    counts['count_cat'] = pd.cut(counts['count'], bins=bins, labels=labels, right=True)
    counts = counts.to_crs(epsg=3857)
    national_outline.plot(ax=axs[0, 1], edgecolor='black', facecolor='none', linewidth=.7)
    counts = counts.sort_values(by='count_cat')
    counts.plot(
        ax=axs[0, 1],
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
    ctx.add_basemap(axs[0, 1], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[0, 1].legend(handles=patches, title='Count', loc='lower right', frameon=True)
    axs[0, 1].set_title('(B) Transformer Count', loc='left', pad=0)
    axs[0, 1].axis('off')

    # Subplot C: Plot population per substation
    filename = 'population_by_node.gpkg'
    folder_in = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder_in, filename)
    population = gpd.read_file(path_in)
    population = population.to_crs(epsg=3857)
    pop_bins = [0, 5000, 10000, 50000, 100000, float('inf')]
    pop_labels = ['<5k', '<10k', '<50k', '<100k', '≥100k']
    population['pop_cat'] = pd.cut(population['population'], bins=pop_bins, labels=pop_labels, right=False)
    population = population.sort_values(by='pop_cat')
    national_outline.plot(ax=axs[1, 0], edgecolor='black', facecolor='none', linewidth=.7)
    population = population.sort_values(by='pop_cat')
    population.plot(
        ax=axs[1, 0],
        column='pop_cat',
        cmap='plasma',
        legend=True,
        legend_kwds={'title': 'Population', 'loc':'lower right'},
        edgecolor='grey',
        linewidth=0.5,
        markersize=15,       
    )
    ctx.add_basemap(axs[1, 0], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[1, 0].set_title('(C) Substation Population', loc='left', pad=0)
    axs[1, 0].axis('off')

    # Subplot D: Plot transmission lines by voltage
    path_in = os.path.join(BASE_PATH, 'processed', 'NZL', 'transmission_lines.gpkg')
    lines = gpd.read_file(path_in)
    lines = lines.to_crs(epsg=3857)
    voltages = [66, 110, 220]
    label_map = {
        66: '66 kV',
        110: '110 kV',
        220: '220 kV'
    }
    # Define TP Std colors
    voltage_colors = {
        66: 'blue',
        110: 'red',
        220: 'orange'
    }
    national_outline.plot(ax=axs[1, 1], edgecolor='black', facecolor='none', linewidth=.7)
    for voltage in voltages:
        subset = lines[lines['Voltage (kV)'] == voltage]
        subset.plot(
            ax=axs[1, 1],
            color=voltage_colors[voltage],
            linewidth=1,
            label=label_map[voltage]
        )

    ctx.add_basemap(ax=axs[1, 1], source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
    axs[1, 1].set_title('(D) Transmission Lines', loc='left', pad=0)
    axs[1, 1].legend(title='Voltage', loc='lower right')
    axs[1, 1].axis('off')

    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    if not os.path.exists(VIS):
        os.makedirs(VIS)

    path_out = os.path.join(VIS, 'panel_plot.png')
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregate_costs(custom_labels=None):
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
    plot_path = os.path.join(VIS, 'summary_plot.png')
    plt.savefig(plot_path, dpi=300)



import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_sector_costs():
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
            ax.text(total + 0.01, j, f'{total:.3f}', va='center', fontsize=8)

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



if __name__ == "__main__":

    plot_panel()

    # plot_aggregate_costs()

    # plot_sector_costs()