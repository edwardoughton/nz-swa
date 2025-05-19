"""
Preprocess data.

Ed Oughton

January 2025

"""
# import sys
import os
import configparser
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
import json
from shapely.ops import voronoi_diagram
from rasterstats import zonal_stats
import pyproj
from sklearn.neighbors import BallTree
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__),'..', 'scripts', 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, '..', '..', 'data_raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def process_regions(iso3, level):
    """
    Function for processing the lowest desired subnational
    regions for the chosen country.

    Parameters
    ----------
    country : dict
        Contains all desired country information.

    """
    regions = []

    if not os.path.exists(os.path.join(DATA_PROCESSED, iso3)):
        os.makedirs(os.path.join(DATA_PROCESSED, iso3))

    for regional_level in range(1, int(level) + 1):

        filename = 'regions_{}_{}.shp'.format(regional_level, iso3)
        folder = os.path.join(DATA_PROCESSED, iso3, 'regions')
        path_processed = os.path.join(folder, filename)

        if os.path.exists(path_processed):
            continue

        print('Processing GID_{} region shapes'.format(regional_level))

        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = 'gadm36_{}.shp'.format(regional_level)
        path_regions = os.path.join(DATA_RAW, 'gadm36_levels_shp', filename)
        regions = gpd.read_file(path_regions)

        regions = regions[regions.GID_0 == iso3]

        regions = regions.copy()
        # regions["geometry"] = regions.geometry.simplify(
        #     tolerance=0.005, preserve_topology=True)

        # regions['geometry'] = regions.apply(remove_small_shapes, axis=1)

        glob_info_path = os.path.join(BASE_PATH, 'countries.csv')
        load_glob_info = pd.read_csv(glob_info_path, encoding = "ISO-8859-1",
            keep_default_na=False)
        regions = regions.merge(
            load_glob_info, left_on='GID_0', right_on='iso3')
    
        if regional_level == 2:
            exclude_names = ['Chatham Islands', 'Northern Islands', 'Southern Islands']
            regions = regions[~regions['NAME_1'].isin(exclude_names)]
        try:
            regions.to_file(path_processed, driver='ESRI Shapefile')
        except:
            print('Unable to write {}'.format(filename))
            pass

    return


def process_country_shapes(iso3):
    """
    Creates a single national boundary for the desired country.

    Parameters
    ----------
    country : dict
        Contains all desired country information.

    """
    path = os.path.join(DATA_PROCESSED, iso3)

    if os.path.exists(os.path.join(path, 'national_outline.shp')):
        return 'Completed national outline processing'

    print('Processing country shapes')

    if not os.path.exists(path):
        os.makedirs(path)

    shape_path = os.path.join(path, 'national_outline.shp')

    path = os.path.join(path, 'regions', 'regions_2_NZL.shp')
    country = gpd.read_file(path)

    national_outline = country.dissolve(by='GID_0')

    # Optionally, reset the index and drop the dummy column
    national_outline = national_outline.reset_index(drop=True)

    national_outline.to_file(shape_path)

    return


def process_settlement_layer(country):
    """
    Clip the settlement layer to the chosen country boundary
    and place in desired country folder.

    Parameters
    ----------
    country : dict
        Contains all desired country information.

    """
    iso3 = country['iso3']
    regional_level = 2 #country['regional_level']

    path_settlements = os.path.join(DATA_RAW,'settlement_layer',
        'ppp_2020_1km_Aggregated.tif')

    settlements = rasterio.open(path_settlements, 'r+')
    settlements.nodata = 255
    settlements.crs = {"init": "epsg:4326"}

    iso3 = country['iso3']
    path_country = os.path.join(BASE_PATH, 'processed', iso3,
        'national_outline.shp')

    if os.path.exists(path_country):
        country = gpd.read_file(path_country)
    else:
        print('Must generate national_outline.shp first' )

    path_country = os.path.join(BASE_PATH, 'processed', iso3)
    shape_path = os.path.join(path_country, 'settlements.tif')

    if os.path.exists(shape_path):
        return print('Completed settlement layer processing')

    print('----')
    print('Working on {} level {}'.format(iso3, regional_level))

    bbox = country.envelope
    geo = gpd.GeoDataFrame()

    geo = gpd.GeoDataFrame({'geometry': bbox})

    coords = [json.loads(geo.to_json())['features'][0]['geometry']]

    out_img, out_transform = mask(settlements, coords, crop=True)

    out_meta = settlements.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": 'epsg:4326'})

    with rasterio.open(shape_path, "w", **out_meta) as dest:
            dest.write(out_img)

    return print('Completed processing of settlement layer')


def get_regional_data(country):
    """
    Extract regional data including luminosity and population.

    Parameters
    ----------
    country : dict
        Contains all desired country information.

    """
    iso3 = country['iso3']
    level = 2# country['regional_level']
    gid_level = 'GID_{}'.format(level)

    filename = 'population.csv'
    folder = os.path.join(BASE_PATH, 'processed', iso3, 'population')
    if not os.path.exists(folder):
        os.mkdir(folder)
    path_output = os.path.join(folder, filename)

    # if os.path.exists(path_output):
    #     return print('Regional data already exists')

    path_country = os.path.join(BASE_PATH, 'processed', iso3,
        'national_outline.shp')

    single_country = gpd.read_file(path_country)

    path_settlements = os.path.join(BASE_PATH, 'processed', iso3,
        'settlements.tif')

    if not iso3 in ['MDV','COK','KIR','MHL','NIU']:
        filename = 'regions_{}_{}.shp'.format(level, iso3)
        folder = os.path.join(BASE_PATH, 'processed', iso3, 'regions')
        path = os.path.join(folder, filename)
        regions = gpd.read_file(path)#[:1]
    else:
        filename = 'national_outline.shp'
        folder = os.path.join(BASE_PATH, 'processed', iso3)
        path = os.path.join(folder, filename)
        regions = gpd.read_file(path)#[:1]

    results = []

    for index, region in regions.iterrows():

        with rasterio.open(path_settlements) as src:

            affine = src.transform
            array = src.read(1)
            array[array <= 0] = 0
            
            if region['geometry'] == None:
                continue

            population_summation = [d['sum'] for d in zonal_stats(
                region['geometry'],
                array,
                stats=['sum'],
                nodata=0,
                affine=affine
                )][0]

        area_km2 = round(area_of_polygon(region['geometry']) / 1e6)

        if area_km2 == 0:
            continue

        if area_km2 > 0:
            population_km2 = (
                population_summation / area_km2 if population_summation else 0)
        else:
            population_km2 = 0

        results.append({
            'GID_0': region['GID_0'],
            'country_name': country['country'],
            'GID_id': region[gid_level],
            'GID_level': gid_level,
            'population': round(population_summation if population_summation else 0,0),
            'area_km2': round(area_km2,1),
            'population_km2': round(population_km2,1),
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(path_output, index=False)

    print('Completed {}'.format(single_country.NAME_0.values[0]))

    return print('Completed regional data')


def area_of_polygon(geom):
    """
    Returns the area of a polygon. Assume WGS84 as crs.

    """
    geod = pyproj.Geod(ellps="WGS84")

    poly_area, poly_perimeter = geod.geometry_area_perimeter(
        geom
    )

    return abs(poly_area)


def process_substations(country):
    """
    Load in substation locations from .xlsx file. 
    Convert to GeoDataFrame using lat/lon.
    Export as GeoPackage. 
    """
    filename = "NZ network information.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    
    # Load Excel with second row as header
    df = pd.read_excel(path_in, sheet_name='NZ network', header=1)
    
    # Keep specific rows and columns
    df = df.iloc[2:530]
    df = df[['substation name', 'location', 'Node Number', 'Latitude (Degrees North)', 'Longitude (Degrees East)', 'Earthed']]
    
    # Rename columns for convenience
    df.rename(columns={
        'Latitude (Degrees North)': 'lat',
        'Longitude (Degrees East)': 'lon'
    }, inplace=True)
    
    # Create geometry from lat/lon
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Output path
    folder_out = os.path.join(BASE_PATH, 'processed', country['iso3'])
    os.makedirs(folder_out, exist_ok=True)
    path_out = os.path.join(folder_out, 'substations.gpkg')

    # Export as GeoPackage
    gdf.to_file(path_out, driver='GPKG')


def count_transformers(country: dict):
    """
    Count transformers at substations.

    Args:
        country (dict): Contains country info, including 'iso3'.
    """
    # Input path
    path_in = os.path.join(BASE_PATH, 'processed', country['iso3'], 'substations.gpkg')
    gdf = gpd.read_file(path_in)

    # Count transformers per substation_name
    counts = gdf.groupby('substation name').size().reset_index(name='count')

    # Join the counts back to the original GeoDataFrame (if you want to keep geometry)
    gdf_counts = gdf.drop_duplicates(subset='substation name').merge(counts, on='substation name')

    # Output path
    folder_out = os.path.join(BASE_PATH, 'processed', country['iso3'])
    os.makedirs(folder_out, exist_ok=True)
    path_out = os.path.join(folder_out, 'substation_counts.gpkg')
    gdf_counts.to_file(path_out, driver='GPKG')

    print(f"Substation counts written to: {path_out}")


def generate_voronoi_from_substations(country):
    """
    Load substations from GeoPackage.
    Snap overlapping points.
    Generate Voronoi polygons clipped to national outline.
    """
    # Load substations
    folder_in = os.path.join(BASE_PATH, 'processed', country['iso3'])
    path_in = os.path.join(folder_in, 'substations.gpkg')
    gdf = gpd.read_file(path_in)
    
    # Round coordinates to reduce precision for de-duplication
    gdf['x_rounded'] = gdf.geometry.x.round(5)
    gdf['y_rounded'] = gdf.geometry.y.round(5)

    # Group by rounded coordinates
    grouped = gdf.groupby(['x_rounded', 'y_rounded'])

    # Aggregate info for each unique point
    agg_df = grouped.agg({
        'Earthed': lambda x: 'E' if 'E' in x.values else 'No',
        'substation name': 'first',
        'location': 'first'
    }).reset_index()

    # Create geometry from rounded coordinates
    agg_df['geometry'] = agg_df.apply(lambda row: Point(row['x_rounded'], row['y_rounded']), axis=1)

    # Create GeoDataFrame
    gdf_unique = gpd.GeoDataFrame(agg_df.drop(columns=['x_rounded', 'y_rounded']), geometry='geometry', crs=gdf.crs)

    folder_out = os.path.join(BASE_PATH, 'processed', country['iso3'])
    path_out = os.path.join(folder_out, 'unique_substations.gpkg')
    gdf_unique.to_file(path_out, driver='GPKG')

    # Convert to projected CRS for Voronoi (e.g., NZTM: EPSG:2193 or UTM)
    gdf_proj = gdf_unique.to_crs(epsg=2193)

    # Build Voronoi diagram
    points = MultiPoint(gdf_proj.geometry.tolist())
    voronoi = voronoi_diagram(points, envelope=gdf_proj.unary_union.envelope, edges=False)
    
    # Create GeoDataFrame of Voronoi cells
    gdf_voronoi = gpd.GeoDataFrame(geometry=[poly for poly in voronoi.geoms], crs=gdf_proj.crs)

    # Load national outline and reproject to match Voronoi CRS
    outline_path = os.path.join(BASE_PATH, 'processed', country['iso3'], 'national_outline.shp')
    national_outline = gpd.read_file(outline_path).to_crs(gdf_voronoi.crs)

    # Clip Voronoi polygons to national outline
    gdf_voronoi_clipped = gpd.overlay(gdf_voronoi, national_outline, how='intersection')

    folder_out = os.path.join(BASE_PATH, 'processed', country['iso3'])
    path_out = os.path.join(folder_out, 'service_areas.gpkg')

    # Export as GeoPackage
    gdf_voronoi_clipped.to_file(path_out, driver='GPKG')

    return 


def estimate_population_by_node(country):
    """
    Estimate the population served by each substation node. 
    """
    # Load population grid
    filename = 'new-zealand-estimated-resident-population-grid-1-kilometre.shp'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    population_grid = gpd.read_file(path_in)
    population_grid['geometry'] = population_grid['geometry'].centroid
    population_grid = population_grid.to_crs("EPSG:4326")

    # Load substations
    filename = 'unique_substations.gpkg'
    folder_in = os.path.join(BASE_PATH, 'processed', country['iso3'])
    path_in = os.path.join(folder_in, filename)
    nodes = gpd.read_file(path_in)
    nodes = nodes.to_crs("EPSG:4326")

    # Prepare coordinates
    grid_coords = np.array([[geom.y, geom.x] for geom in population_grid.geometry])
    node_coords = np.array([[geom.y, geom.x] for geom in nodes.geometry])
    
    # Create spatial index using BallTree (Haversine distance, in radians)
    tree = BallTree(np.radians(node_coords), metric='haversine')
    distances, indices = tree.query(np.radians(grid_coords), k=1)

    # Assign population grid points to nearest node
    population_grid['nearest_node'] = indices.flatten()
    population_grid['population'] = population_grid['PopEst2023']  # Adjust to actual column name if different

    # Aggregate population by node
    population_sum = population_grid.groupby('nearest_node')['population'].sum().reset_index()
    nodes['population'] = 0
    nodes.loc[population_sum['nearest_node'], 'population'] = population_sum['population'].values
    nodes['population'] = round(nodes['population'])

    # Allocate island information
    filename = 'north_south_island_lut.csv'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_csv(path_in)
    lut = lut[['substation name','island']]
    nodes = pd.merge(nodes, lut, left_on='substation name', right_on='substation name')

    # Allocate substation GIC for Quebec 89
    filename = 'Transformer and substation extreme storm GIC.xlsx'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_excel(path_in, header=4, usecols=[4,5,6])
    lut = lut[:91]
    lut['Substation'] = lut['Substation'].str.replace("'", "", regex=False)
    nodes = pd.merge(nodes, lut, left_on='location', right_on='Substation', how='left')

    # Save results
    filename = 'population_by_node.gpkg'
    folder_out = os.path.join(BASE_PATH, 'processed', country['iso3'])
    os.makedirs(folder_out, exist_ok=True)
    path_out = os.path.join(folder_out, filename)
    nodes.to_file(path_out)


def process_lines(country):
    """
    Load in line information and process. 
    
    """
    filename = "NZ network information.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    
    # Load data to get node coord lookup table
    df_nodes = pd.read_excel(path_in, sheet_name='NZ network', header=1)
    df_nodes = df_nodes.iloc[2:530]
    df_nodes = df_nodes[['Node Number', 'Latitude (Degrees North)', 'Longitude (Degrees East)']]

    # Convert to dictionary: {node_number: (latitude, longitude)}
    node_dict = {
        row['Node Number']: (row['Latitude (Degrees North)'], row['Longitude (Degrees East)'])
        for _, row in df_nodes.iterrows()
    }

    # Load data to get node network information
    df_lines = pd.read_excel(path_in, sheet_name='NZ network', header=534)
    df_lines = df_lines.iloc[:1062]
    df_lines = df_lines[['node1', 'node2', 'Voltage of line (kV)']]

    # Allocate coordinates and create LineString geometries
    geometries = []
    voltages = []

    for _, row in df_lines.iterrows():
        coord1 = node_dict.get(row['node1'])
        coord2 = node_dict.get(row['node2'])

        if coord1 and coord2:

            if not row['Voltage of line (kV)'] > 0:
                continue

            # Note: Shapely expects coordinates in (longitude, latitude)
            line = LineString([(coord1[1], coord1[0]), (coord2[1], coord2[0])])
            geometries.append(line)
            voltages.append(row['Voltage of line (kV)'])

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'Voltage (kV)': voltages, 'geometry': geometries}, crs="EPSG:4326")

    # Example output
    output_path = os.path.join(BASE_PATH, 'processed', 'NZL', 'transmission_lines.gpkg')
    gdf.to_file(output_path)


def process_scenario1(country):
    """
    
    """
    filename = 'population_by_node.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in)

    data['outage'] = 1
    data['load_shedding'] = 0

    filename = 'scenario1.csv'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(folder, exist_ok=True)
    path_out = os.path.join(folder, filename)
    data.to_csv(path_out)


def process_scenario2(country):
    """
    
    """
    filename = 'population_by_node.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in)

    # Set outage = 1 for rows where island is 'north'
    data.loc[data['island'] == 'north', 'outage'] = 1

    # Set load_shedding = 1 for rows where island is 'south'
    data.loc[data['island'] == 'south', 'load_shedding'] = 1

    filename = 'scenario2.csv'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(folder, exist_ok=True)
    path_out = os.path.join(folder, filename)
    data.to_csv(path_out)


def process_sioc_lut(country):
    """
    Extract SIOC lookup table
    
    """
    filename = "national-accounts-input-output-tables-year-ended-march-2020-revised-22-december-2021.xlsx"
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    df = pd.read_excel(path_in, sheet_name='NZSIOC to ANZSIC06', header=3)
    df = df[['NZSIOC','Description']]
    df = df.drop_duplicates()#.reset_index()

    filename = "nzsioc_lut.csv"
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_out = os.path.join(folder, filename)
    df.to_csv(path_out, index=False)


if __name__ == "__main__":

    filename = "countries.csv"
    path = os.path.join(BASE_PATH, filename)

    countries = pd.read_csv(path, encoding='latin-1')

    for idx, country in countries.iterrows():

        if not country['iso3'] in ['NZL']: 
            continue

        # print('Working on process_regions')
        # process_regions(country['iso3'], int(country['gid_region']))

        # print('Working on process_country_shapes')
        # process_country_shapes(country['iso3'])

        # print('Processing process_settlement_layer')
        # process_settlement_layer(country)

        # print('Processing get_regional_data')
        # get_regional_data(country)

        # print('Processing get_regional_data')
        # process_substations(country)

        # print('Processing cluster_transformers')
        # count_transformers(country)

        # print('Processing generate_voronoi_from_substations')
        # generate_voronoi_from_substations(country)

        # print('Process estimate node population')
        # estimate_population_by_node(country)

        # print('Process lines')
        # process_lines(country)

        # print('processing process_scenario1')
        # process_scenario1(country)

        # print('processing process_scenario2')
        # process_scenario2(country)

        print('processing process_sioc_lut')
        process_sioc_lut(country)