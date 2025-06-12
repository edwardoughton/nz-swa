"""
Preprocess data.

Ed Oughton

January 2025

"""
# import sys
import os
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString, MultiPolygon
from shapely.ops import nearest_points
import rasterio
from rasterio.mask import mask
import json
from shapely.ops import voronoi_diagram
from rasterstats import zonal_stats
import pyproj
from sklearn.neighbors import BallTree
import networkx as nx

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

        # if os.path.exists(path_processed):
        #     continue

        print('Processing GID_{} region shapes'.format(regional_level))

        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = 'gadm36_{}.shp'.format(regional_level)
        path_regions = os.path.join(DATA_RAW, 'gadm36_levels_shp', filename)
        regions = gpd.read_file(path_regions)
        regions = regions.to_crs(27200)

        regions = regions[regions.GID_0 == iso3]

        regions = regions.copy()
        # regions["geometry"] = regions.geometry.simplify(
        #     tolerance=0.005, preserve_topology=True)

        regions['geometry'] = regions.apply(remove_small_shapes, axis=1)

        glob_info_path = os.path.join(BASE_PATH, 'countries.csv')
        load_glob_info = pd.read_csv(glob_info_path, encoding = "ISO-8859-1",
            keep_default_na=False)
        regions = regions.merge(
            load_glob_info, left_on='GID_0', right_on='iso3')
    
        # if regional_level == 2:
        exclude_names = ['Chatham Islands', 'Northern Islands', 'Southern Islands']
        regions = regions[~regions['NAME_1'].isin(exclude_names)]

        regions = regions.to_crs(4326)  
        regions.to_file(path_processed, driver='ESRI Shapefile')


def remove_small_shapes(x):
    """
    Remove small multipolygon shapes.

    Parameters
    ---------
    x : polygon
        Feature to simplify.

    Returns
    -------
    MultiPolygon : MultiPolygon
        Shapely MultiPolygon geometry without tiny shapes.

    """
    # if its a single polygon, just return the polygon geometry
    if x.geometry.geom_type == 'Polygon':
        return x.geometry
    
    # if its a multipolygon, we start trying to simplify
    # and remove shapes if its too big.
    elif x.geometry.geom_type == 'MultiPolygon':

        area1 = 0.01
        area2 = 50

        if x.geometry.area < area1:
            return x.geometry

        elif x.geometry.area > area2:
            threshold = 10000000
        else:
            threshold = 100000

        # save remaining polygons as new multipolygon 
        new_geom = []
        for y in x.geometry.geoms:
            if y.area > threshold:
                new_geom.append(y)

        return MultiPolygon(new_geom)
    

def process_country_shapes(iso3):
    """
    Creates a single national boundary for the desired country.

    Parameters
    ----------
    country : dict
        Contains all desired country information.

    """
    path = os.path.join(DATA_PROCESSED, iso3)

    # if os.path.exists(os.path.join(path, 'national_outline.shp')):
    #     return 'Completed national outline processing'

    print('Processing country shapes')

    if not os.path.exists(path):
        os.makedirs(path)

    shape_path = os.path.join(path, 'national_outline.shp')

    path = os.path.join(path, 'regions', 'regions_2_NZL.shp')
    country = gpd.read_file(path)
    country = country.to_crs(27200)
    country = country.explode(index_parts=True, ignore_index=True)
    area_threshold = 10000000  # in square meters
    country = country[country.geometry.area > area_threshold].reset_index(drop=True)
    country = country.dissolve(by='GID_0')

    # Optionally, reset the index and drop the dummy column
    country = country.reset_index(drop=True)
    country = country.to_crs(4326)
    country.to_file(shape_path)

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
    lut = lut[['substation name','island','islanding']]
    nodes = pd.merge(nodes, lut, left_on='substation name', right_on='substation name')

    # Allocate substation GIC for Quebec 89
    filename = 'Transformer and substation extreme storm GIC_v3.xlsx'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    lut = pd.read_excel(path_in, header=4, usecols=[10,11,12,13,14,15,16,17,18])
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


def process_hydro_locations(country):
    """
    Process hydro locations.

    """
    filename = 'fuel_gen.csv'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    data = pd.read_csv(path_in)
    data = data[data['Fuel_Code'] == 'Hydro']

    # Create geometry column from easting and northing
    geometry = [Point(xy) for xy in zip(data["NZTM easting"], data["NZTM northing"])]

    # Define the NZTM projection (EPSG:2193 is standard for NZTM2000)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:2193")
    
    filename = 'fuel_gen.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_out = os.path.join(folder, filename)
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(path_out)

    return


def process_employment_data(country):
    """
    
    """
    filename = 'geographic-units-by-industry-and-statistical-area-2000-2024-descending-order-february-2024.csv'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    data = pd.read_csv(path_in)

    # Get 2024
    data = data[data['year'] == 2024]

    # # Get sectors
    # # sectors_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S'] #,'T','U','V','X','W','Y','Z']
    # # data = data[data['anzsic06'].isin(sectors_list)]
    # data['anzsic06'] = data['anzsic06'].astype(str)
    # data = data[data['anzsic06'].str.len() == 4]

    filename = 'anzsic_concordance_table.xlsx'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    concordance_table = pd.read_excel(path_in, header=9)
    concordance_table = concordance_table[['ANZSIC06','Target Code']]

    filename = 'anzsic_industry_groupings.xlsx'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    concordance_table2 = pd.read_excel(path_in)
    
    concordance_table = pd.merge(concordance_table, concordance_table2, left_on='Target Code', right_on='industry_groupings', how='inner')

    data = pd.merge(data, concordance_table, left_on='anzsic06', right_on='ANZSIC06', how='inner')

    # Get statistical areas 2 (SA2)
    data = data[data['Area'].str.startswith('A')]

    filename = 'employment_lut.csv'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_out = os.path.join(folder, filename)
    data.to_csv(path_out, index=False)


def process_sa2_to_node_lut(country):
    """
    Find the closest node for each SA2 boundary.
    
    Export a .csv which maps the sa2_boundaries['SA22023_V1'] with the closest node id in gdf as nodes['location']
    """
    # Load SA2 boundaries
    filename = 'statistical-area-2-2023-generalised.shp'
    folder = os.path.join(BASE_PATH, 'raw')
    path_in = os.path.join(folder, filename)
    sa2_boundaries = gpd.read_file(path_in)
    sa2_boundaries = sa2_boundaries.to_crs(2193)
    sa2_boundaries = sa2_boundaries[['geometry', 'SA22023_V1', 'SA22023__1']]

    # Filter out invalid or missing geometries
    sa2_boundaries = sa2_boundaries[
        sa2_boundaries['geometry'].notna() & sa2_boundaries['geometry'].is_valid
    ].copy()

    # Apply representative point safely
    sa2_boundaries['geometry'] = sa2_boundaries['geometry'].apply(
        lambda geom: geom.representative_point()
    )

    # Load nodes
    filename = 'population_by_node.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    nodes = gpd.read_file(path_in)
    nodes = nodes.to_crs(2193)

    # Build spatial index
    nodes_sindex = nodes.sindex

    # Nearest node finder
    def find_nearest_node(point):
        idx = list(nodes_sindex.nearest([point], return_all=False))[1]
        return nodes.loc[idx, 'location'].values[0]

    # Apply to each SA2 point geometry
    sa2_boundaries['nearest_node'] = sa2_boundaries['geometry'].apply(find_nearest_node)

    # Export to CSV
    output_df = sa2_boundaries[['SA22023_V1', 'nearest_node']]
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    output_path = os.path.join(folder, f'sa2_to_node_lut_{country['iso3']}.csv')
    output_df.to_csv(output_path, index=False)


def get_employment_by_node(country):
    """
    
    """
    filename = 'employment_lut.csv'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, filename)
    data = pd.read_csv(path_in)
    data['Area'] = data['Area'].str.replace('A','')
    data = data[['sector_name','Area','ec_count']]

    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_in = os.path.join(folder, f'sa2_to_node_lut_{country['iso3']}.csv')
    node_lut = pd.read_csv(path_in)
    node_lut['SA22023_V1'] = node_lut['SA22023_V1'].astype(str)

    merged = pd.merge(data, node_lut, left_on='Area', right_on='SA22023_V1', how='inner')

    merged = merged[['sector_name','ec_count','nearest_node']]

    employment_by_node = merged.groupby(['nearest_node', 'sector_name'], as_index=False)['ec_count'].sum()

    folder = os.path.join(DATA_PROCESSED, 'NZL')
    output_path = os.path.join(folder, 'employment_per_node.csv')
    employment_by_node.to_csv(output_path, index=False)

    pivot_df = employment_by_node.pivot(index="nearest_node", columns="sector_name", values="ec_count").reset_index()

    # Load nodes
    filename = 'population_by_node.gpkg'
    path_in = os.path.join(folder, filename)
    nodes = gpd.read_file(path_in)  # Assuming it's a GeoPackage
    nodes = pd.merge(nodes, pivot_df, left_on='location', right_on='nearest_node', how='inner')

    # output
    output_path = os.path.join(BASE_PATH, 'processed', 'NZL', 'employment_by_node.gpkg')
    nodes.to_file(output_path)


def generate_restoration_sequence(country):
    """
    Generate restoration sequence based on hydro locations and transmission lines.

    """
    # Load hydro sites
    filename = 'fuel_gen.gpkg'
    folder = os.path.join(DATA_PROCESSED, country['iso3'])
    path_in = os.path.join(folder, filename)
    hydro = gpd.read_file(path_in)
    hydro = hydro.to_crs(2193)
    hydro['geometry'] = hydro['geometry'].buffer(10)

    # Load nodes
    filename = 'employment_by_node.gpkg'
    path_in = os.path.join(folder, filename)
    nodes = gpd.read_file(path_in)  # Assuming it's a GeoPackage
    nodes = nodes.to_crs(2193)
    nodes['restoration_stage'] = None  # Initialize restoration stage

    # Load transmission lines
    filename = 'transmission_lines.gpkg'
    path_in = os.path.join(folder, filename)
    lines = gpd.read_file(path_in)
    lines = lines.to_crs(2193)
    lines['geometry'] = lines['geometry'].buffer(100)

    # Step 1: Hydro-intersecting nodes get stage 1
    stage = 1
    hydro_nodes = gpd.sjoin(nodes, hydro, predicate='intersects')
    nodes.loc[hydro_nodes.index, 'restoration_stage'] = stage

    # Build a connectivity graph from lines and nodes
    G = nx.Graph()
    for idx, row in nodes.iterrows():
        G.add_node(idx, geometry=row.geometry)

    for _, line in lines.iterrows():
        intersecting_nodes = nodes[nodes.intersects(line.geometry)]
        for i in range(len(intersecting_nodes)):
            for j in range(i + 1, len(intersecting_nodes)):
                a, b = intersecting_nodes.index[i], intersecting_nodes.index[j]
                if not G.has_edge(a, b):
                    dist = nodes.loc[a].geometry.distance(nodes.loc[b].geometry)
                    G.add_edge(a, b, weight=dist)

    # BFS to assign restoration stages
    visited = set(hydro_nodes.index)
    queue = list(hydro_nodes.index)
    current_stage = 2

    while queue:
        next_queue = []
        for node in queue:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes.at[neighbor, 'restoration_stage'] = current_stage
                    next_queue.append(neighbor)
        queue = next_queue
        current_stage += 1

    nodes = nodes.drop(['Substation','nearest_node'], axis=1)

    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(DATA_PROCESSED, 'NZL')
    path_out = os.path.join(folder, filename)
    nodes = nodes.to_crs(epsg=4326)
    nodes.to_file(path_out)


def process_scenario1(country):
    """
    Write a scenario with a 7-day power outage duration (d1-d7), as follows:

    - d1, d2, d3: full blackout (1 = no power)
    - d4 to d7: restoration begins, based on the "restoration_stage" column
    - by d7, all areas have power (0 = power restored)

    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in)

    # Normalize restoration_stage to [0, 1]
    data = data.copy()
    if 'restoration_stage' not in data.columns:
        raise ValueError("'restoration_stage' column not found in input data")

    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    min_stage = data['restoration_stage'].min()
    max_stage = data['restoration_stage'].max()
    data['normalized_stage'] = ((data['restoration_stage'] - min_stage) / (max_stage - min_stage)) ** 0.5

    # Days 1-3: Full blackout
    for day in range(1, 4):
        data[f'd{day}'] = 1

    # Days 4-7: Gradual restoration
    for day in range(4, 8):
        threshold = (day - 3) / 4  # 0.25 for d4, 0.5 for d5, 0.75 for d6, 1.0 for d7
        data[f'd{day}'] = (data['normalized_stage'] > threshold).astype(int)

    # Drop helper column
    data = data.drop(columns=['normalized_stage'])

    # Save to CSV
    filename = 'scenario1.csv'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(folder, exist_ok=True)
    path_out = os.path.join(folder, filename)
    data.to_csv(path_out, index=False)


def process_scenario2(country):
    """
    Write a scenario with a 7-day power outage duration (d1-d7), as follows:

    South Island (data['island'] == 'south'):
    - d1-d3: full blackout (1 = no power),
    - d4-d7: restoration begins, based on the "restoration_stage" column
    - by d7, all areas have power (0 = power restored)

    North Island (data['island'] == 'north'):
    - d1-d7: load shedding of 20% (0.2 = load shedding),
    - by d7, all areas have power (0 = power restored)

    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in)

    if 'restoration_stage' not in data.columns:
        raise ValueError("'restoration_stage' column not found in input data")
    if 'island' not in data.columns:
        raise ValueError("'island' column not found in input data")

    data = data.copy()
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')

    # Normalize restoration stage only for South Island (recovery case)
    south_mask = data['island'].str.lower() == 'south'
    data.loc[south_mask, 'normalized_stage'] = (
        data.loc[south_mask].groupby('island')['restoration_stage'].transform(
            lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4
        )
    )

    # Assign outage values for each day
    for day in range(1, 8):
        col = f'd{day}'

        # South Island logic
        if day <= 3:
            data.loc[south_mask, col] = 1  # Full blackout
        else:
            threshold = (day - 3) / 4  # Gradual recovery d4–d7
            data.loc[south_mask, col] = (data.loc[south_mask, 'normalized_stage'] > threshold).astype(int)

        # North Island logic
        if day <= 6:
            data.loc[~south_mask, col] = 0.2  # Load shedding
        else:
            data.loc[~south_mask, col] = 0  # Fully restored

    # Clean up
    data = data.drop(columns=['normalized_stage'])

    # Save to CSV
    filename = 'scenario2.csv'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(folder, exist_ok=True)
    path_out = os.path.join(folder, filename)
    data.to_csv(path_out, index=False)


def process_scenario3(country):
    """
    Generates a 7-day power outage scenario based on GIC impact and island region:
    
    - Nodes with GIC > 500 A:
        - d1–d3: full blackout (1)
        - d4–d7: gradual restoration via 'restoration_stage'
        - d7: fully restored (0)
        
    - Nodes in the North Island not affected:
        - d1–d6: 20% load shedding (0.2)
        - d7: fully restored (0)
        
    - All others:
        - d1–d7: fully powered (0)
    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in).copy()

    # Validate columns
    required_cols = ['GIC [A](max)_baseline', 'restoration_stage', 'island']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"'{col}' column not found in input data")

    # Parse columns
    data['GIC_max'] = pd.to_numeric(data['GIC [A](max)_baseline'], errors='coerce')
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    data['island'] = data['island'].str.lower()

    # Define masks
    fail_mask = data['GIC_max'] > 500
    north_mask = data['island'] == 'north'
    unaffected_north_mask = north_mask & ~fail_mask
    unaffected_other_mask = ~fail_mask & ~north_mask

    # Normalize restoration stages only for affected nodes
    data['normalized_stage'] = None
    if fail_mask.any():
        data.loc[fail_mask, 'normalized_stage'] = (
            data.loc[fail_mask]
            .groupby('island')['restoration_stage']
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4)
        )

    # Initialize power status day-by-day
    for day in range(1, 8):
        col = f'd{day}'

        # Default: full power
        data[col] = 0.0

        # Load shedding for unaffected North Island
        if day <= 6:
            data.loc[unaffected_north_mask, col] = 0.2

        # Full blackout for failed nodes (d1–d3)
        if day <= 3:
            data.loc[fail_mask, col] = 1
        else:
            threshold = (day - 3) / 4  # d4=0.25, d5=0.5, etc.
            data.loc[fail_mask, col] = (
                data.loc[fail_mask, 'normalized_stage'].astype(float) > threshold
            ).astype(int)

    # Cleanup
    data.drop(columns=['normalized_stage', 'GIC_max'], inplace=True)

    # Save output
    out_folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(out_folder, exist_ok=True)
    path_out = os.path.join(out_folder, 'scenario3.csv')
    data.to_csv(path_out, index=False)


def process_scenario4(country):
    """
    Generates a 7-day power outage scenario using GIC values (switching variant):

    - Nodes with GIC > 500 A:
        - d1–d3: full blackout (1)
        - d4–d7: gradual restoration using 'restoration_stage'
        - d7: fully restored (0)

    - Unaffected North Island nodes:
        - d1–d6: 20% load shedding (0.2)
        - d7: fully restored (0)

    - Others (not affected, not in North Island): always have power (0)
    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in)

    data = data.copy()

    # Check required columns
    required_cols = ['GIC [A](max)_switching', 'restoration_stage', 'island']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"'{col}' column not found in input data")

    # Parse and clean data
    data['GIC_max'] = pd.to_numeric(data['GIC [A](max)_switching'], errors='coerce')
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    data['island'] = data['island'].str.lower()

    # Identify node masks
    fail_mask = data['GIC_max'] > 500
    north_mask = data['island'] == 'north'
    unaffected_north_mask = ~fail_mask & north_mask
    unaffected_other_mask = ~fail_mask & ~north_mask

    # Normalize restoration stage ONLY for affected nodes
    data['normalized_stage'] = None
    if fail_mask.any():
        norm_stage = (
            data.loc[fail_mask]
            .groupby('island')['restoration_stage']
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4)
        )
        data.loc[fail_mask, 'normalized_stage'] = norm_stage

    # Assign day-by-day outage scenario
    for day in range(1, 8):
        col = f'd{day}'

        # Default: full power
        data[col] = 0.0

        # Unaffected North Island nodes
        if day <= 6:
            data.loc[unaffected_north_mask, col] = 0.2
        else:
            data.loc[unaffected_north_mask, col] = 0

        # Affected nodes (failures)
        if day <= 3:
            data.loc[fail_mask, col] = 1  # full blackout
        else:
            threshold = (day - 3) / 4  # d4=0.25, d5=0.5, etc.
            data.loc[fail_mask, col] = (
                data.loc[fail_mask, 'normalized_stage'].astype(float) > threshold
            ).astype(int)

    # Clean up
    data = data.drop(columns=['normalized_stage', 'GIC_max'])

    # Output
    filename = 'scenario4.csv'
    out_folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(out_folder, exist_ok=True)
    path_out = os.path.join(out_folder, filename)
    data.to_csv(path_out, index=False)


def process_scenario5(country):
    """
    7-day outage scenario with GIC > 500 A failures and islanding exceptions.

    - Affected nodes: GIC > 500 A and not protected by islanding
        - d1–d3: full blackout (1)
        - d4–d7: restoration via 'restoration_stage'
        - d7: fully restored (0)

    - Unaffected North Island nodes:
        - d1–d6: 20% load shedding (0.2)
        - d7: fully restored (0)

    - All others: full power (0)
    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in).copy()

    # Ensure necessary columns are present
    required_columns = ['GIC [A](max)_switching', 'restoration_stage', 'island', 'islanding']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"'{col}' column not found in input data")

    # Clean and parse
    data['GIC_max'] = pd.to_numeric(data['GIC [A](max)_switching'], errors='coerce')
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    data['island'] = data['island'].str.lower()
    data['islanding'] = data['islanding'].str.lower()

    # Define protected tags
    protected_tags = {
        'waitaki-oamaru', 'gore-dunedin',
        'kikiwa-greymouth', 'manapouri-invercargill'
    }

    # Determine failed nodes (high GIC and not protected)
    fail_mask = (data['GIC_max'] > 500) & ~data['islanding'].isin(protected_tags)

    # Determine unaffected North Island nodes
    north_mask = data['island'] == 'north'
    unaffected_north_mask = ~fail_mask & north_mask
    unaffected_other_mask = ~fail_mask & ~north_mask

    # Normalize restoration only for failed nodes
    data['normalized_stage'] = None
    if fail_mask.any():
        data.loc[fail_mask, 'normalized_stage'] = (
            data.loc[fail_mask]
            .groupby('island')['restoration_stage']
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4)
        )

    # Assign scenario day-by-day
    for day in range(1, 8):
        col = f'd{day}'

        # Default: full power
        data[col] = 0.0

        # North Island load shedding
        if day <= 6:
            data.loc[unaffected_north_mask, col] = 0.2
        else:
            data.loc[unaffected_north_mask, col] = 0

        # Affected nodes
        if day <= 3:
            data.loc[fail_mask, col] = 1
        else:
            threshold = (day - 3) / 3
            data.loc[fail_mask, col] = (
                data.loc[fail_mask, 'normalized_stage'].astype(float) > threshold
            ).astype(int)

    # Final cleanup
    data.drop(columns=['normalized_stage', 'GIC_max'], inplace=True)

    # Save output
    out_folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, 'scenario5.csv')
    data.to_csv(out_path, index=False)


def process_scenario6(country):
    """
    6-day outage scenario using GIC [A](max)_blocker1.

    - Affected nodes (GIC > 500 A):
        - d1–d3: full blackout (1)
        - d4-d5: staged restoration via 'restoration_stage'
        - d6: fully restored (0)

    - Unaffected North Island nodes:
        - d1–d6: 20% load shedding (0.2)
        - d6: fully restored (0)

    - Others: always have power (0)
    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in).copy()

    # Validate required columns
    required_cols = ['GIC [A](max)_blocker1', 'restoration_stage', 'island']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"'{col}' column not found in input data")

    # Parse and clean
    data['GIC_max'] = pd.to_numeric(data['GIC [A](max)_blocker1'], errors='coerce')
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    data['island'] = data['island'].str.lower()

    # Identify node categories
    fail_mask = data['GIC_max'] > 500
    north_mask = data['island'] == 'north'
    unaffected_north_mask = ~fail_mask & north_mask
    unaffected_other_mask = ~fail_mask & ~north_mask

    # Normalize restoration stage only for affected nodes
    data['normalized_stage'] = None
    if fail_mask.any():
        data.loc[fail_mask, 'normalized_stage'] = (
            data.loc[fail_mask]
            .groupby('island')['restoration_stage']
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4)
        )

    # Assign power status day-by-day
    for day in range(1, 8):
        col = f'd{day}'

        # Default: full power
        data[col] = 0.0

        # Load shedding for unaffected North Island
        if day <= 5:
            data.loc[unaffected_north_mask, col] = 0.2
        else:
            data.loc[unaffected_north_mask, col] = 0

        # Affected nodes
        if day <= 3:
            data.loc[fail_mask, col] = 1
        else:
            threshold = (day - 3) / 2 
            data.loc[fail_mask, col] = (
                data.loc[fail_mask, 'normalized_stage'].astype(float) > threshold
            ).astype(int)

    # Cleanup
    data.drop(columns=['normalized_stage', 'GIC_max'], inplace=True)

    # Save output
    out_folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(out_folder, exist_ok=True)
    path_out = os.path.join(out_folder, 'scenario6.csv')
    data.to_csv(path_out, index=False)


def process_scenario7(country):
    """
    4-day outage scenario using GIC [A](max)_blocker2 values.

    - Affected nodes (GIC > 500 A):
        - d1–d3: full blackout (1)
        - d4: staged restoration via 'restoration_stage'
        - d5: fully restored (0)

    - Unaffected North Island nodes:
        - d1–d4: 20% load shedding (0.2)
        - d5: fully restored (0)

    - All others: always have power (0)
    """
    filename = 'restoration_sequence.gpkg'
    folder = os.path.join(BASE_PATH, 'processed', 'NZL')
    path_in = os.path.join(folder, filename)
    data = gpd.read_file(path_in).copy()

    # Validate required columns
    required_cols = ['GIC [A](max)_blocker2', 'restoration_stage', 'island']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"'{col}' column not found in input data")

    # Parse and clean data
    data['GIC_max'] = pd.to_numeric(data['GIC [A](max)_blocker2'], errors='coerce')
    data['restoration_stage'] = pd.to_numeric(data['restoration_stage'], errors='coerce')
    data['island'] = data['island'].str.lower()

    # Identify masks
    fail_mask = data['GIC_max'] > 500
    north_mask = data['island'] == 'north'
    unaffected_north_mask = ~fail_mask & north_mask
    unaffected_other_mask = ~fail_mask & ~north_mask

    # Normalize restoration stage for affected nodes only
    data['normalized_stage'] = None
    if fail_mask.any():
        data.loc[fail_mask, 'normalized_stage'] = (
            data.loc[fail_mask]
            .groupby('island')['restoration_stage']
            .transform(lambda x: ((x - x.min()) / (x.max() - x.min())) ** 0.4)
        )

    # 6-day scenario: d1 to d6
    for day in range(1, 8):
        col = f'd{day}'
        
        # Default: full power
        data[col] = 0.0

        # Unaffected North Island nodes
        if day <= 4:
            data.loc[unaffected_north_mask, col] = 0.2
        else:
            data.loc[unaffected_north_mask, col] = 0

        # Affected nodes
        if day <= 3:
            data.loc[fail_mask, col] = 1  # full blackout
        else:
            threshold = (day - 3) / 1  # d3=0.33, d4=0.66, d5=1.0
            data.loc[fail_mask, col] = (
                data.loc[fail_mask, 'normalized_stage'].astype(float) > threshold
            ).astype(int)

    # Clean up
    data.drop(columns=['normalized_stage', 'GIC_max'], inplace=True)

    # Save
    out_folder = os.path.join(BASE_PATH, 'processed', 'NZL', 'scenarios')
    os.makedirs(out_folder, exist_ok=True)
    path_out = os.path.join(out_folder, 'scenario7.csv')
    data.to_csv(path_out, index=False)


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

        # print('processing process_sioc_lut')
        # process_sioc_lut(country)

        # print('processing process_hydro_locations')
        # process_hydro_locations(country)

        # print('processing process_employment_data')
        # process_employment_data(country)

        # print('processing process_sa2_to_node_lut')
        # process_sa2_to_node_lut(country)

        # print('processing get_employment_by_node')
        # get_employment_by_node(country)

        # print('processing generate_restoration_sequence')
        # generate_restoration_sequence(country)

        # print('processing process_scenario1')
        # process_scenario1(country)

        # print('processing process_scenario2')
        # process_scenario2(country)

        print('processing process_scenario3')
        process_scenario3(country)

        print('processing process_scenario4')
        process_scenario4(country)

        print('processing process_scenario5')
        process_scenario5(country)

        print('processing process_scenario6')
        process_scenario6(country)

        print('processing process_scenario7')
        process_scenario7(country)