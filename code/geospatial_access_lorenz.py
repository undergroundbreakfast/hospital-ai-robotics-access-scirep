#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# ==============================================================================
#  geospatial_hospital_ai_robotics_analysis.py
#  Author:  Aaron Johnson
#  Created: 2025-07-14
#  Revised: 2026-01-05
#  VERSION 14.0 - Corrects the k-NN graph visualization logic to prevent
#                 spurious connections to non-contiguous states by filtering
#                 the dataset before graph creation.
#
#  VERSION 12.0 - This version incorporates a new k-Nearest Neighbors (k-NN)
#                 graph analysis to visualize the network connectivity
#                 between hospitals based on geographic proximity.
#
#  VERSION 11.0 - This version incorporates feedback to refine visualizations
#                 and strengthen the core regression model.
#                 1. Adds a 3rd Lorenz curve for baseline hospital access.
#                 2. Annotates regression plot with coefficients and p-values.
#                 3. Fixes the centering of the contiguous US LISA map.
#                 4. Implements a new regression model using composite health
#                    indices and controls for population and census division.
#
#  End-to-end workflow analyzing spatial patterns of Generative AI and
#  robotics adoption in US hospitals (AHA 2024). This script performs:
#   1. Proximity analysis with robust outlier removal.
#   2. Population coverage analysis for AI/robotics hospitals.
#   3. Inequality analysis using Lorenz curves and Gini coefficients.
#   4. Regression analysis linking technology access to Years of Potential Life Lost (YPLL).
#   5. Hot spot (LISA) analysis to find spatial clusters of adoption.
#   6. k-NN graph analysis to visualize hospital network structure.
#   7. Generates publication-quality maps and figures.
# ==============================================================================
import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
import traceback

# Configure GDAL to auto-restore missing .shx files for shapefiles
# MUST be set before importing geopandas
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Suppress numerical warnings from scipy/sklearn (common in logistic regression with extreme values)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# --- Core Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from tqdm import tqdm

# --- Analysis Libraries ---
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
from numpy import trapezoid
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import statsmodels.formula.api as smf # For detailed regression output
import statsmodels.stats.weightstats as wstats # Add this import for weighted stats

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Optional Geospatial Libraries (with graceful failure) ---
try:
    import contextily as cx  # type: ignore
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    cx = None  # type: ignore
    print("Warning: 'contextily' not found. Basemaps will not be added to maps.")

try:
    import esda  # type: ignore
    import libpysal  # type: ignore
    from splot.esda import lisa_cluster  # type: ignore
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    esda = None  # type: ignore
    libpysal = None  # type: ignore
    lisa_cluster = None  # type: ignore
    print("Warning: 'pysal', 'esda', 'splot' not found. Hot spot analysis will be skipped.")

# Use Albers Equal Area projection specifically designed for US maps
from matplotlib.figure import Figure
try:
    from cartopy.crs import AlbersEqualArea  # type: ignore
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    AlbersEqualArea = None  # type: ignore
    print("Warning: 'cartopy' not found. Some map projections will not be available.")

# ======================= CONFIGURATION ==========================
BASE_DIR = Path(__file__).resolve().parent
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"
CSV_DIR = BASE_DIR / "publication_outputs"  # For publication-ready CSV/TXT files
LOG_FILE = BASE_DIR / "analysis_log.txt"
CX_CACHE = BASE_DIR / "tile_cache"

COUNTY_FILE = SHAPE_DIR / "tl_2024_us_county.shp"
POP_FILE = SHAPE_DIR / "USA_BlockGroups_2020Pop.geojson"

DRIVE_MINUTES = 30
AVG_SPEED_MPH = 40.0
RADIUS_MILES = AVG_SPEED_MPH * DRIVE_MINUTES / 60.0  # Straight-line distance: 20 miles

# Circuity factor adjustment: Roads are ~30% longer than straight-line distance
# This converts straight-line distance to effective road network distance
# Source: Typical US road network circuity (Levinson & El-Geneidy, 2009)
CIRCUITY_FACTOR = 1.3
EFFECTIVE_RADIUS_MILES = RADIUS_MILES / CIRCUITY_FACTOR  # ~15.4 miles on road network

EARTH_RADIUS_MILES = 3958.8

WGS84_CRS = "EPSG:4326"
AEA_CRS = "EPSG:5070"

# ======================= SETUP ================================
def setup_environment():
    """Create necessary directories and configure logging."""
    for p in (SHAPE_DIR, FIG_DIR, CSV_DIR, CX_CACHE):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("=" * 60)
    logging.info(f"RUN STARTED: {datetime.now().isoformat()}")
    logging.info("=" * 60)

# ======================= DATA LOADING & PREPROCESSING =======================
def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        db_password = os.getenv("POSTGRESQL_KEY")
        if not db_password:
            raise ValueError("POSTGRESQL_KEY environment variable not set.")
        engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{db_password}@{os.getenv('PGHOST', 'localhost')}/{os.getenv('PGDATABASE', 'Research_TEST')}",
            pool_pre_ping=True, connect_args={"connect_timeout": 10},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info(f"Postgres connection successful: {engine.url.host}/{engine.url.database}")
        return engine
    except Exception as e:
        logging.exception("FATAL: Failed to connect to PostgreSQL database.")
        sys.exit(1)

def load_hospital_data(engine):
    """Loads hospital data from the aha_fy2024 table with purchased geocoding data."""
    query = """
    SELECT
        aha.id AS hospital_id, 
        aha.mname, 
        aha.mlocaddr AS address, 
        aha.mloccity AS city, 
        aha.mlocstcd AS state, 
        aha.mloczip AS zipcode, 
        aha.fcounty AS county_fips,
        CAST(geo.lat AS FLOAT) AS latitude, 
        CAST(geo.long AS FLOAT) AS longitude, 
        aha.bsc, 
        aha.sysname, 
        aha.robohos, 
        aha.robosys, 
        aha.roboven, 
        aha.adjpd,
        aha.ftemd, 
        aha.ftern, 
        aha.wfaipsn, 
        aha.wfaippd, 
        aha.wfaiss, 
        aha.wfaiart, 
        aha.wfaioacw, 
        aha.gfeet, 
        aha.ceamt
    FROM aha_survey_data aha
    JOIN aha_fy2024 geo ON aha.id = geo.id
    WHERE geo.lat IS NOT NULL 
      AND geo.long IS NOT NULL
      AND geo.lat != '0' 
      AND geo.long != '0'
      AND geo.lat != ''
      AND geo.long != '';
    """
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Successfully loaded {len(df)} records from aha_fy2024 with purchased geocoding data.")
        return df
    except Exception as e:
        logging.exception("FATAL: Error executing query on hospital data.")
        sys.exit(1)

def load_population_data(pop_filepath):
    """Loads and preprocesses population data from a GeoJSON file."""
    logging.info(f"Attempting to load population data from: {pop_filepath.name}")
    if not pop_filepath.exists():
        logging.error(f"FATAL: Population GeoJSON file not found at: {pop_filepath}")
        return None
    try:
        pop_gdf = gpd.read_file(pop_filepath)
        logging.info(f"Successfully loaded {len(pop_gdf)} records from {pop_filepath.name}")
        if 'population' in pop_gdf.columns and 'POPULATION' not in pop_gdf.columns:
            logging.info("Renaming 'population' column to 'POPULATION' for consistency.")
            pop_gdf.rename(columns={'population': 'POPULATION'}, inplace=True)

        if 'POPULATION' not in pop_gdf.columns:
            logging.error("FATAL: 'POPULATION' column not found in GeoJSON.")
            return None

        pop_gdf['POPULATION'] = pd.to_numeric(pop_gdf['POPULATION'], errors='coerce').fillna(0)
        pop_gdf = pop_gdf[pop_gdf['POPULATION'] > 0] # Remove zero-pop block groups
        logging.info(f"{len(pop_gdf)} block groups remain after filtering for non-zero population.")
        return pop_gdf
    except Exception as e:
        logging.error(f"Failed to load or process population GeoJSON file: {e}", exc_info=True)
        return None

def preprocess_data(df):
    """Cleans data, creates analysis columns, filters to US states, and returns a GeoDataFrame."""
    logging.info("Preprocessing data...")
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    df = df[(pd.to_numeric(df['latitude'], errors='coerce').notna()) & (pd.to_numeric(df['longitude'], errors='coerce').notna())]
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    logging.info(f"{len(df)} hospitals remaining after cleaning coordinates.")

    ai_cols = ["wfaipsn", "wfaippd", "wfaiss", "wfaiart", "wfaioacw"]
    for col in ai_cols + ['robohos', 'bsc']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['ai_flag'] = (df[ai_cols].max(axis=1) > 0).astype(int)
    df['ai_intensity'] = df[ai_cols].sum(axis=1)
    df['robo_flag'] = (df['robohos'] > 0).astype(int)

    def get_tech_type(row):
        if row['ai_flag'] == 1 and row['robo_flag'] == 1: return 'AI & Robotics'
        elif row['ai_flag'] == 1: return 'AI Only'
        elif row['robo_flag'] == 1: return 'Robotics Only'
        else: return 'Neither'
    df['tech_type'] = df.apply(get_tech_type, axis=1)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=WGS84_CRS
    )
    logging.info("Created initial GeoDataFrame with all hospital locations.")

    logging.info("\n--- Filtering out distant US territories ---")
    original_count = len(gdf)
    excluded_territories = ['PR', 'GU', 'AS', 'VI', 'MP']
    gdf_filtered = gdf[~gdf['state'].isin(excluded_territories)].copy()

    num_removed = original_count - len(gdf_filtered)
    if num_removed > 0:
        removed_hospitals = gdf[gdf['state'].isin(excluded_territories)]
        logging.info(f"Removed {num_removed} hospitals located in distant US territories.")
        logging.info(f"Removed facilities are in territories: {list(removed_hospitals['state'].unique())}")
    else:
        logging.info("No hospitals found in excluded US territories.")

    logging.info(f"Dataset for analysis (including AK & HI) now contains {len(gdf_filtered)} hospitals.")

    # Export summary statistics for all hospitals
    tech_counts = gdf_filtered['tech_type'].value_counts()
    summary_stats = pd.DataFrame({
        'Metric': ['Total_Hospitals', 'AI_And_Robotics', 'AI_Only', 'Robotics_Only', 'Neither',
                   'AI_Flag_Total', 'Robo_Flag_Total', 'States_Represented', 'Territories_Excluded'],
        'Value': [
            len(gdf_filtered),
            tech_counts.get('AI & Robotics', 0),
            tech_counts.get('AI Only', 0),
            tech_counts.get('Robotics Only', 0),
            tech_counts.get('Neither', 0),
            gdf_filtered['ai_flag'].sum(),
            gdf_filtered['robo_flag'].sum(),
            gdf_filtered['state'].nunique(),
            num_removed
        ]
    })
    summary_stats.to_csv(CSV_DIR / "hospital_summary_statistics.csv", index=False)
    logging.info(f"Exported hospital summary statistics to {CSV_DIR / 'hospital_summary_statistics.csv'}")

    return gdf_filtered

# ======================= ANALYSIS FUNCTIONS ==========================
def calculate_proximity_metrics(gdf):
    """
    Calculates proximity metrics and removes statistical outliers based on distance.
    This method preserves valid data from Alaska and Hawaii.
    """
    logging.info("\n" + "="*20 + " PROXIMITY ANALYSIS & OUTLIER REMOVAL " + "="*20)
    if len(gdf) < 4:
        logging.warning("Not enough hospitals (< 4) to calculate k=3 neighbors. Skipping.")
        gdf['nearest_miles'], gdf['k3_avg_miles'] = np.nan, np.nan
        return gdf

    coords_rad = np.deg2rad(gdf[['latitude', 'longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')
    distances, _ = tree.query(coords_rad, k=4)
    distances_miles = distances * EARTH_RADIUS_MILES
    gdf['nearest_miles'] = distances_miles[:, 1]
    gdf['k3_avg_miles'] = distances_miles[:, 1:4].mean(axis=1)

    logging.info("Calculated initial proximity metrics for all hospitals.")
    logging.info(f"Pre-filtering stats: Mean distance={gdf['nearest_miles'].mean():.2f}, Max distance={gdf['nearest_miles'].max():.2f}")

    Q1 = gdf['nearest_miles'].quantile(0.25)
    Q3 = gdf['nearest_miles'].quantile(0.75)
    IQR = Q3 - Q1
    iqr_multiplier = 3.0 # A generous multiplier to keep legitimate remote hospitals
    upper_bound = Q3 + (IQR * iqr_multiplier)

    logging.info(f"IQR-based outlier detection: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    logging.info(f"Calculated upper bound for nearest distance: {upper_bound:.2f} miles.")

    outliers = gdf[gdf['nearest_miles'] > upper_bound]

    if not outliers.empty:
        logging.warning(
            f"IDENTIFIED AND REMOVED {len(outliers)} distance outliers using IQR method (nearest_miles > {upper_bound:.2f} miles)."
        )
        logging.warning("Removed outlier details:\n"
                     f"{outliers[['hospital_id', 'mname', 'city', 'state', 'nearest_miles']].to_string(index=False)}")
        gdf_filtered = gdf[gdf['nearest_miles'] <= upper_bound].copy()
        logging.info(f"{len(gdf_filtered)} hospitals remain after outlier removal.")
    else:
        logging.info("No significant distance outliers found using the IQR method.")
        gdf_filtered = gdf.copy()

    final_gdf = gdf_filtered
    logging.info("\n--- Proximity Metrics on Final, Cleaned Dataset ---")
    logging.info(f"Mean distance to nearest hospital: {final_gdf['nearest_miles'].mean():.2f} miles (SD: {final_gdf['nearest_miles'].std():.2f})")
    logging.info(f"Median distance to nearest hospital: {final_gdf['nearest_miles'].median():.2f} miles")
    logging.info(f"Max distance in final dataset: {final_gdf['nearest_miles'].max():.2f} miles")
    logging.info(f"Mean avg distance to 3 nearest: {final_gdf['k3_avg_miles'].mean():.2f} miles (SD: {final_gdf['k3_avg_miles'].std():.2f})")

    # Export proximity metrics summary statistics
    proximity_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean_Nearest_Miles', 'Median_Nearest_Miles', 'SD_Nearest_Miles', 
                   'Min_Nearest_Miles', 'Max_Nearest_Miles', 'Q1_Nearest_Miles', 'Q3_Nearest_Miles',
                   'Mean_K3_Avg_Miles', 'SD_K3_Avg_Miles', 'Outliers_Removed'],
        'Value': [
            len(final_gdf),
            final_gdf['nearest_miles'].mean(),
            final_gdf['nearest_miles'].median(),
            final_gdf['nearest_miles'].std(),
            final_gdf['nearest_miles'].min(),
            final_gdf['nearest_miles'].max(),
            final_gdf['nearest_miles'].quantile(0.25),
            final_gdf['nearest_miles'].quantile(0.75),
            final_gdf['k3_avg_miles'].mean(),
            final_gdf['k3_avg_miles'].std(),
            len(outliers) if not outliers.empty else 0
        ]
    })
    proximity_stats.to_csv(CSV_DIR / "proximity_metrics_summary.csv", index=False)
    logging.info(f"Exported proximity metrics summary to {CSV_DIR / 'proximity_metrics_summary.csv'}")

    return final_gdf

def analyze_population_coverage(hospital_gdf, pop_gdf):
    """Analyzes population within a 30-min drive of AI/Robotics hospitals."""
    logging.info("\n" + "="*20 + " POPULATION COVERAGE ANALYSIS " + "="*20)
    if pop_gdf is None:
        logging.error("Population data not available. Skipping population coverage analysis.")
        return

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping coverage map.")
        return

    try:
        hosp_proj = hospital_gdf.to_crs(AEA_CRS)
        pop_proj = pop_gdf.to_crs(AEA_CRS)
        total_population = pop_proj['POPULATION'].sum()

        if total_population == 0:
            logging.error("Total population is zero. Check input data. Aborting population analysis.")
            return

        logging.info(f"Total US population from block groups: {total_population:,.0f}")
        logging.info(f"Using {DRIVE_MINUTES}-minute drive time proxy ({EFFECTIVE_RADIUS_MILES:.1f} mile buffer, circuity-adjusted from {RADIUS_MILES:.1f} mile straight-line).")

        def calculate_coverage(tech_gdf, tech_name):
            if tech_gdf.empty:
                logging.warning(f"No hospitals found for '{tech_name}'. Coverage is 0.")
                return None, 0, 0.0
            # Use circuity-adjusted effective radius for road network distance
            buffer_radius_meters = EFFECTIVE_RADIUS_MILES * 1609.34
            
            coverage_area = gpd.GeoSeries(tech_gdf.buffer(buffer_radius_meters), crs=tech_gdf.crs).union_all()
            
            pop_proj['centroid'] = pop_proj.geometry.representative_point()
            covered_blocks = pop_proj.set_geometry('centroid')[pop_proj.set_geometry('centroid').within(coverage_area)]
            covered_pop = covered_blocks['POPULATION'].sum()
            percentage_covered = (covered_pop / total_population) * 100 if total_population > 0 else 0
            logging.info(f"[{tech_name}] Coverage: {covered_pop:,.0f} people ({percentage_covered:.2f}% of total).")
            return coverage_area, covered_pop, percentage_covered

        ai_area, ai_covered_pop, ai_pct = calculate_coverage(hosp_proj[hosp_proj['ai_flag'] == 1], 'AI-Enabled')
        robo_area, robo_covered_pop, robo_pct = calculate_coverage(hosp_proj[hosp_proj['robo_flag'] == 1], 'Robotics')

        # Export population coverage results
        coverage_results = pd.DataFrame({
            'Technology_Type': ['AI_Enabled', 'Robotics'],
            'Hospitals_Count': [
                len(hospital_gdf[hospital_gdf['ai_flag'] == 1]),
                len(hospital_gdf[hospital_gdf['robo_flag'] == 1])
            ],
            'Population_Covered': [ai_covered_pop, robo_covered_pop],
            'Percent_Covered': [ai_pct, robo_pct],
            'Total_Population': [total_population, total_population],
            'Drive_Minutes': [DRIVE_MINUTES, DRIVE_MINUTES],
            'Effective_Radius_Miles': [EFFECTIVE_RADIUS_MILES, EFFECTIVE_RADIUS_MILES],
            'Circuity_Factor': [CIRCUITY_FACTOR, CIRCUITY_FACTOR]
        })
        coverage_results.to_csv(CSV_DIR / "population_coverage_analysis.csv", index=False)
        logging.info(f"Exported population coverage analysis to {CSV_DIR / 'population_coverage_analysis.csv'}")

        us_counties = gpd.read_file(COUNTY_FILE)
        # Set CRS if missing (Census shapefiles are typically NAD83)
        if us_counties.crs is None:
            logging.warning("County shapefile has no CRS defined. Setting to EPSG:4269 (NAD83).")
            us_counties = us_counties.set_crs("EPSG:4269")
        us_counties = us_counties.to_crs(AEA_CRS)
        
        # Filter to contiguous US if STATEFP column exists
        if 'STATEFP' in us_counties.columns:
            us_contig = us_counties[~us_counties['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]
        else:
            logging.warning("County shapefile missing STATEFP column. Using all geometries for basemap.")
            us_contig = us_counties
            
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Plot the basemap (already in AEA projection)
        us_contig.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        
        if ai_area is not None: gpd.GeoSeries([ai_area], crs=AEA_CRS).plot(ax=ax, color='blue', alpha=0.5, label=f'AI Hospital {DRIVE_MINUTES}min Coverage')
        if robo_area is not None: gpd.GeoSeries([robo_area], crs=AEA_CRS).plot(ax=ax, color='red', alpha=0.5, label=f'Robotics Hospital {DRIVE_MINUTES}min Coverage')
        ax.set_title(f'Population Coverage by AI & Robotics Hospitals ({DRIVE_MINUTES}-Min Drive Time Proxy)', fontsize=16)
        ax.set_axis_off()
        if ai_area is not None or robo_area is not None:
            ax.legend(loc='lower left')

        if CONTEXTILY_AVAILABLE:
            cx.add_basemap(ax, crs=AEA_CRS, source=cx.providers.CartoDB.Positron, zoom='auto')

        fig_path = FIG_DIR / "population_coverage_map.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved population coverage map to {fig_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error during population coverage analysis: {e}", exc_info=True)

def analyze_inequality_with_lorenz(hospital_gdf, pop_gdf):
    """
    Computes and plots Lorenz curves and Gini coefficients to measure
    inequality of geographic access to AI, Robotics, and All hospitals.
    """
    logging.info("\n" + "="*20 + " INEQUALITY (LORENZ CURVE) ANALYSIS " + "="*20)
    if pop_gdf is None:
        logging.error("Population data not available. Skipping Lorenz curve analysis.")
        return

    # --- 1. Prepare data ---
    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1].copy()
    hospitals_robo = hospital_gdf[hospital_gdf['robo_flag'] == 1].copy()
    hospitals_all = hospital_gdf.copy()

    if hospitals_ai.empty or hospitals_robo.empty:
        logging.warning("No AI or Robotics hospitals found. Cannot perform inequality analysis.")
        return

    # --- 2. Compute shortest distance from each CBG to nearest tech hospital ---
    logging.info("Projecting CBGs to AEA for accurate centroid calculation.")
    cbg_gdf_proj = pop_gdf.to_crs(AEA_CRS)
    cbg_centroids_proj = cbg_gdf_proj.geometry.representative_point()
    
    cbg_centroids_wgs84 = cbg_centroids_proj.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())
    
    def get_nearest_distances(tech_gdf, cbg_coords_rad, tech_name):
        """Calculates shortest distance from CBGs to a set of hospitals."""
        logging.info(f"Calculating shortest distance to {len(tech_gdf)} '{tech_name}' hospitals for {len(cbg_coords_rad)} CBGs.")
        tech_coords_rad = np.deg2rad(tech_gdf[['latitude', 'longitude']].values)
        tree = BallTree(tech_coords_rad, metric='haversine')
        distances_rad, _ = tree.query(cbg_coords_rad, k=1)
        distances_miles = distances_rad.flatten() * EARTH_RADIUS_MILES
        
        distances_miles[distances_miles == 0] = 0.1
        logging.info(f"--> Applied 0.1 mile floor to {np.sum(distances_miles == 0.1)} co-located block groups.")
        
        return distances_miles

    cbg_gdf = pop_gdf.copy()
    cbg_gdf['dist_ai'] = get_nearest_distances(hospitals_ai, cbg_coords_rad, 'AI')
    cbg_gdf['dist_robo'] = get_nearest_distances(hospitals_robo, cbg_coords_rad, 'Robotics')
    cbg_gdf['dist_all'] = get_nearest_distances(hospitals_all, cbg_coords_rad, 'All')

    # --- 2. Decile Analysis (Reviewer Request A) ---
    logging.info("\n" + "-"*15 + " Decile Analysis " + "-"*15)
    logging.info("Analyzing travel burden by population decile.")

    def create_decile_table(df, dist_col, pop_col):
        """Creates a population-weighted decile analysis table."""
        df_sorted = df.sort_values(dist_col)
        df_sorted['cum_pop_share'] = df_sorted[pop_col].cumsum() / df_sorted[pop_col].sum()
        df_sorted['decile'] = pd.qcut(df_sorted['cum_pop_share'], 10, labels=range(1, 11))
        df_sorted['travel_burden'] = df_sorted[dist_col] * df_sorted[pop_col]
        total_burden = df_sorted['travel_burden'].sum()

        def weighted_avg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            return (d * w).sum() / w.sum()

        decile_summary = df_sorted.groupby('decile').apply(
            lambda g: pd.Series({
                'mean_dist_mi': weighted_avg(g, dist_col, pop_col),
                'median_dist_mi': g[dist_col].median(),
                'travel_burden_share': g['travel_burden'].sum() / total_burden
            })
        ).reset_index()
        decile_summary['cum_travel_burden_share'] = decile_summary['travel_burden_share'].cumsum()
        return decile_summary

    for tech in ['all', 'ai', 'robo']:
        logging.info(f"\n--- Decile Table for '{tech.upper()}' Hospital Access ---")
        decile_table = create_decile_table(cbg_gdf, f'dist_{tech}', 'POPULATION')
        logging.info("\n" + decile_table.to_string(index=False))
        
        # Export decile table
        decile_table.to_csv(CSV_DIR / f"decile_analysis_{tech}_access.csv", index=False)
        logging.info(f"Exported decile analysis to {CSV_DIR / f'decile_analysis_{tech}_access.csv'}")

    # --- 3. Absolute Gap Analysis (P90/P10) (User Request #2) ---
    logging.info("\n" + "-"*15 + " Absolute Gap Analysis (P90/P10) " + "-"*15)
    gap_data = []
    for tech, name in [('all', 'All Hospitals'), ('ai', 'AI'), ('robo', 'Robotics')]:
        dist_col = f'dist_{tech}'
        weighted_stats = wstats.DescrStatsW(cbg_gdf[dist_col], weights=cbg_gdf['POPULATION'])
        p10 = float(weighted_stats.quantile(0.10).iloc[0])
        p90 = float(weighted_stats.quantile(0.90).iloc[0])
        gap_data.append({
            'Access Type': name,
            'P10_dist_mi': p10,
            'P90_dist_mi': p90,
            'Absolute_Gap_mi': p90 - p10,
            'Ratio_Gap (P90/P10)': p90 / p10
        })
    gap_df = pd.DataFrame(gap_data)
    logging.info("Population-weighted distance percentiles and gaps:")
    logging.info("\n" + gap_df.to_string(index=False))
    
    # Export gap analysis
    gap_df.to_csv(CSV_DIR / "absolute_gap_analysis_p90_p10.csv", index=False)
    logging.info(f"Exported gap analysis to {CSV_DIR / 'absolute_gap_analysis_p90_p10.csv'}")

    # --- 4. Core Lorenz/Gini Plotting (Unchanged) ---
    logging.info("\n" + "-"*15 + " Lorenz Curve & Gini Coefficient Analysis " + "-"*15)

    def lorenz_xy(distance, weight):
        df = pd.DataFrame({'d': distance, 'w': weight}).sort_values('d')
        x = df['w'].cumsum() / df['w'].sum()
        y = (df['d'] * df['w']).cumsum() / (df['d'] * df['w']).sum()
        return pd.concat([pd.Series([0]), x]).values, pd.concat([pd.Series([0]), y]).values

    def calculate_gini(x, y):
        return 1 - 2 * trapezoid(y, x)

    x_ai, y_ai = lorenz_xy(cbg_gdf.dist_ai, cbg_gdf.POPULATION)
    gini_ai = calculate_gini(x_ai, y_ai)
    x_robo, y_robo = lorenz_xy(cbg_gdf.dist_robo, cbg_gdf.POPULATION)
    gini_robo = calculate_gini(x_robo, y_robo)
    x_all, y_all = lorenz_xy(cbg_gdf.dist_all, cbg_gdf.POPULATION)
    gini_all = calculate_gini(x_all, y_all)
    
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(x_ai, y_ai, label=f"AI (Gini={gini_ai:.3f})", color='#0072B2', lw=2)
    ax1.plot(x_robo, y_robo, label=f"Robotics (Gini={gini_robo:.3f})", color='#D55E00', lw=2)
    ax1.plot(x_all, y_all, label=f"All Hospitals (Gini={gini_all:.3f})", color='#009E73', lw=2, linestyle=':')
    ax1.plot([0, 1], [0, 1], c='black', ls='--', label='Line of Perfect Equality')
    ax1.set_xlabel("Cumulative Share of Population", fontsize=12)
    ax1.set_ylabel("Cumulative Share of Travel Burden (Distance)", fontsize=12)
    ax1.set_title("Inequality of Access to Hospital Technology", fontsize=14, pad=15)
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    gini_data = pd.DataFrame({
        'Technology': ['AI', 'Robotics', 'All Hospitals'],
        'Gini': [gini_ai, gini_robo, gini_all]
    })
    
    palette = {'AI': '#0072B2', 'Robotics': '#D55E00', 'All Hospitals': '#009E73'}
    sns.barplot(x='Technology', y='Gini', data=gini_data, ax=ax2, palette=palette, hue='Technology', legend=False)

    ax2.set_xlabel("Technology / Access Type", fontsize=12)
    ax2.set_ylabel("Gini Coefficient of Access", fontsize=12)
    ax2.set_title("Gini Coefficient", fontsize=14, pad=15)
    plt.suptitle("Geographic Inequality in Access to AI and Robotics in U.S. Hospitals", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    fig_path = FIG_DIR / "lorenz_gini_inequality_analysis_final.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved main Lorenz/Gini inequality plot to {fig_path}")
    plt.close(fig)

    # Export Gini coefficients
    gini_results = pd.DataFrame({
        'Technology': ['AI', 'Robotics', 'All_Hospitals'],
        'Gini_Coefficient': [gini_ai, gini_robo, gini_all]
    })
    gini_results.to_csv(CSV_DIR / "gini_coefficients.csv", index=False)
    logging.info(f"Exported Gini coefficients to {CSV_DIR / 'gini_coefficients.csv'}")

    # --- 5. Robustness Checks (CV, Epsilon-Shift Gini, Atkinson) ---
    logging.info("\n" + "-"*15 + " Robustness Checks " + "-"*15)

    logging.info("\n--- Coefficient of Variation (SD / Mean) ---")
    cv_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'CV': [
            cbg_gdf['dist_all'].std() / cbg_gdf['dist_all'].mean(),
            cbg_gdf['dist_ai'].std() / cbg_gdf['dist_ai'].mean(),
            cbg_gdf['dist_robo'].std() / cbg_gdf['dist_robo'].mean()
        ]
    }
    logging.info("\n" + pd.DataFrame(cv_data).to_string(index=False))

    logging.info("\n--- Gini Coefficient with 0.25-mile Epsilon-Shift ---")
    dist_shifted_all = cbg_gdf['dist_all'].clip(lower=0.25)
    x_s_all, y_s_all = lorenz_xy(dist_shifted_all, cbg_gdf.POPULATION)
    gini_shifted_all = calculate_gini(x_s_all, y_s_all)

    dist_shifted_ai = cbg_gdf['dist_ai'].clip(lower=0.25)
    x_s_ai, y_s_ai = lorenz_xy(dist_shifted_ai, cbg_gdf.POPULATION)
    gini_shifted_ai = calculate_gini(x_s_ai, y_s_ai)

    dist_shifted_robo = cbg_gdf['dist_robo'].clip(lower=0.25)
    x_s_robo, y_s_robo = lorenz_xy(dist_shifted_robo, cbg_gdf.POPULATION)
    gini_shifted_robo = calculate_gini(x_s_robo, y_s_robo)

    gini_shift_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'Original Gini': [gini_all, gini_ai, gini_robo],
        'Shifted Gini (0.25mi floor)': [gini_shifted_all, gini_shifted_ai, gini_shifted_robo]
    }
    logging.info("\n" + pd.DataFrame(gini_shift_data).to_string(index=False))

    logging.info("\n--- Atkinson Index (epsilon = 0.5) ---")
    def calculate_atkinson(values, weights, epsilon=0.5):
        """Calculates the Atkinson index for a given distribution."""
        if epsilon == 1:
            ratio = np.log(values)
            geo_mean = np.exp(np.average(ratio, weights=weights))
            arith_mean = np.average(values, weights=weights)
            return 1 - (geo_mean / arith_mean)
        else:
            weighted_mean = np.average(values, weights=weights)
            term = (values / weighted_mean)**(1 - epsilon)
            weighted_avg_term = np.average(term, weights=weights)
            return 1 - weighted_avg_term**(1 / (1 - epsilon))
    
    atkinson_all = calculate_atkinson(cbg_gdf['dist_all'], cbg_gdf['POPULATION'])
    atkinson_ai = calculate_atkinson(cbg_gdf['dist_ai'], cbg_gdf['POPULATION'])
    atkinson_robo = calculate_atkinson(cbg_gdf['dist_robo'], cbg_gdf['POPULATION'])
    
    atkinson_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'Atkinson Index (e=0.5)': [atkinson_all, atkinson_ai, atkinson_robo]
    }
    logging.info("\n" + pd.DataFrame(atkinson_data).to_string(index=False))

    # Export comprehensive robustness checks
    robustness_df = pd.DataFrame({
        'Access_Type': ['All_Hospitals', 'AI', 'Robotics'],
        'Gini_Original': [gini_all, gini_ai, gini_robo],
        'Gini_Shifted_0.25mi': [gini_shifted_all, gini_shifted_ai, gini_shifted_robo],
        'Coefficient_of_Variation': [
            cbg_gdf['dist_all'].std() / cbg_gdf['dist_all'].mean(),
            cbg_gdf['dist_ai'].std() / cbg_gdf['dist_ai'].mean(),
            cbg_gdf['dist_robo'].std() / cbg_gdf['dist_robo'].mean()
        ],
        'Atkinson_Index_e0.5': [atkinson_all, atkinson_ai, atkinson_robo]
    })
    robustness_df.to_csv(CSV_DIR / "robustness_checks_inequality.csv", index=False)
    logging.info(f"Exported robustness checks to {CSV_DIR / 'robustness_checks_inequality.csv'}")

def sensitivity_analysis_exposure_construction(engine, hospital_gdf, pop_gdf):
    """
    Comprehensive sensitivity analysis for exposure construction:
    1. Travel time thresholds: 15, 30, 45, 60 minutes
    2. Alternative weighting schemes: unweighted, adjpd-weighted, bed-weighted
    3. Dose-response curves using continuous exposure
    
    Addresses reviewer concerns about knife-edge design choices.
    """
    logging.info("\n" + "="*80)
    logging.info(" SENSITIVITY ANALYSIS: EXPOSURE CONSTRUCTION ")
    logging.info("="*80)
    
    if pop_gdf is None:
        logging.error("Population data not available. Skipping sensitivity analysis.")
        return
    
    # Load health outcomes
    query = """
    SELECT
        m.county_fips,
        m.social_economic_factors_score,
        m.health_behaviors_score,
        m.physical_environment_score,
        m.population,
        m.census_division,
        v.premature_death_raw_value as ypll
    FROM vw_conceptual_model_adjpd m
    JOIN vw_conceptual_model_variables v ON m.county_fips = v.county_fips;
    """
    try:
        health_data = pd.read_sql(query, engine)
        logging.info(f"Loaded {len(health_data)} county records for sensitivity analysis.")
    except Exception as e:
        logging.error(f"Failed to load health data: {e}")
        return
    
    # Prepare hospital and population data
    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1].copy()
    
    # Check for required columns and ensure numeric types
    if 'bsc' not in hospital_gdf.columns:
        logging.warning("'bsc' (bed size) column not found. Bed-weighted analysis will use adjpd as proxy.")
        hospitals_ai['bed_weight'] = pd.to_numeric(hospitals_ai.get('adjpd', 1.0), errors='coerce').fillna(1.0)
    else:
        hospitals_ai['bed_weight'] = pd.to_numeric(hospitals_ai['bsc'], errors='coerce').fillna(
            pd.to_numeric(hospitals_ai.get('adjpd', 1.0), errors='coerce').fillna(1.0)
        )
    
    hospitals_ai['adjpd_weight'] = pd.to_numeric(hospitals_ai.get('adjpd', 1.0), errors='coerce').fillna(1.0)
    
    # Ensure all weights are positive
    hospitals_ai['bed_weight'] = hospitals_ai['bed_weight'].clip(lower=1.0)
    hospitals_ai['adjpd_weight'] = hospitals_ai['adjpd_weight'].clip(lower=1.0)
    
    cbg_gdf_proj = pop_gdf.to_crs(AEA_CRS)
    cbg_centroids_proj = cbg_gdf_proj.geometry.representative_point()
    cbg_centroids_wgs84 = cbg_centroids_proj.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())
    hosp_coords_rad = np.deg2rad(hospitals_ai[['latitude', 'longitude']].values)
    
    # Build BallTree for distance queries
    tree = BallTree(hosp_coords_rad, metric='haversine')
    
    # Define sensitivity parameters
    time_thresholds = [15, 30, 45, 60]  # minutes
    avg_speed_mph = 40.0
    
    # Weighting schemes
    weight_schemes = {
        'unweighted': np.ones(len(hospitals_ai)),
        'adjpd_weighted': hospitals_ai['adjpd_weight'].values,
        'bed_weighted': hospitals_ai['bed_weight'].values
    }
    
    # Storage for results
    sensitivity_results = []
    
    logging.info("\n" + "-"*80)
    logging.info("COMPUTING EXPOSURES ACROSS TIME THRESHOLDS AND WEIGHTING SCHEMES")
    logging.info("-"*80)
    
    # Compute exposure for each combination
    for time_min in time_thresholds:
        radius_miles = avg_speed_mph * time_min / 60.0
        radius_rad = radius_miles / EARTH_RADIUS_MILES
        
        for weight_name, weights in weight_schemes.items():
            logging.info(f"\nProcessing: {time_min} min threshold, {weight_name} scheme")
            
            # Find all hospitals within threshold for each CBG
            indices_list = tree.query_radius(cbg_coords_rad, r=radius_rad)
            
            cbg_exposure = []
            for idx, hosp_indices in enumerate(indices_list):
                if len(hosp_indices) == 0:
                    cbg_exposure.append(0.0)
                else:
                    # Exposure is the sum of weights for accessible hospitals
                    # Normalized by total weight in the system (to make it a share/percentage)
                    accessible_weight = weights[hosp_indices].sum()
                    total_weight = weights.sum()
                    cbg_exposure.append(100 * accessible_weight / total_weight if total_weight > 0 else 0.0)
            
            # Aggregate to county level (population-weighted average)
            cbg_temp = pop_gdf.copy()
            cbg_temp['exposure'] = cbg_exposure
            cbg_temp['county_fips'] = cbg_temp['GEOID'].str[:5]
            
            county_exposure = cbg_temp.groupby('county_fips').apply(
                lambda x: np.average(x['exposure'], weights=x['POPULATION']) if x['POPULATION'].sum() > 0 else 0.0
            ).reset_index()
            county_exposure.columns = ['county_fips', 'exposure']
            
            # Merge with health data and run regression
            analysis_df = health_data.merge(county_exposure, on='county_fips', how='inner')
            analysis_df = analysis_df.dropna(subset=['ypll', 'exposure', 'social_economic_factors_score', 
                                                      'health_behaviors_score', 'physical_environment_score'])
            
            if len(analysis_df) < 50:
                logging.warning(f"Insufficient data for {time_min}min, {weight_name}: only {len(analysis_df)} counties")
                continue
            
            # Standardize covariates
            covariates = ['exposure', 'social_economic_factors_score', 'health_behaviors_score',
                         'physical_environment_score', 'population']
            scaler = StandardScaler()
            analysis_df[covariates] = scaler.fit_transform(analysis_df[covariates])
            
            # Run OLS with robust SE
            formula = f"ypll ~ {' + '.join(covariates)} + C(census_division)"
            try:
                model = smf.ols(formula, data=analysis_df).fit(cov_type='HC1')
                
                exposure_coef = model.params['exposure']
                exposure_se = model.bse['exposure']
                exposure_pval = model.pvalues['exposure']
                exposure_ci_low = model.conf_int().loc['exposure', 0]
                exposure_ci_high = model.conf_int().loc['exposure', 1]
                
                sensitivity_results.append({
                    'time_threshold_min': time_min,
                    'weight_scheme': weight_name,
                    'n_counties': len(analysis_df),
                    'coef': exposure_coef,
                    'se': exposure_se,
                    'pval': exposure_pval,
                    'ci_low': exposure_ci_low,
                    'ci_high': exposure_ci_high,
                    'r_squared': model.rsquared
                })
                
                logging.info(f"  β = {exposure_coef:.2f}, SE = {exposure_se:.2f}, p = {exposure_pval:.4f}, R² = {model.rsquared:.3f}")
            
            except Exception as e:
                logging.error(f"  Regression failed: {e}")
                continue
    
    # Save results table
    if sensitivity_results:
        results_df = pd.DataFrame(sensitivity_results)
        results_path = CSV_DIR / "sensitivity_exposure_thresholds_weights.csv"
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nSaved sensitivity results to {results_path}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel A: Effect by time threshold
        ax1 = axes[0]
        for weight_name in weight_schemes.keys():
            subset = results_df[results_df['weight_scheme'] == weight_name]
            ax1.errorbar(subset['time_threshold_min'], subset['coef'], 
                        yerr=1.96*subset['se'], marker='o', label=weight_name, capsize=5)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('Travel Time Threshold (minutes)', fontsize=12)
        ax1.set_ylabel('Coefficient (YPLL per 1 SD exposure)', fontsize=12)
        ax1.set_title('Panel A: Sensitivity to Travel Time Threshold', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel B: Effect by weighting scheme
        ax2 = axes[1]
        for time_min in time_thresholds:
            subset = results_df[results_df['time_threshold_min'] == time_min]
            x_pos = np.arange(len(weight_schemes))
            ax2.errorbar(x_pos + time_min*0.02, subset['coef'], 
                        yerr=1.96*subset['se'], marker='o', label=f'{time_min} min', capsize=5)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_xticks(range(len(weight_schemes)))
        ax2.set_xticklabels(list(weight_schemes.keys()))
        ax2.set_xlabel('Weighting Scheme', fontsize=12)
        ax2.set_ylabel('Coefficient (YPLL per 1 SD exposure)', fontsize=12)
        ax2.set_title('Panel B: Sensitivity to Weighting Scheme', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = FIG_DIR / "sensitivity_exposure_construction.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved sensitivity visualization to {fig_path}")
        plt.close()
    
    # ===========================================================================
    # DOSE-RESPONSE CURVE (Continuous Exposure, 30-min threshold, adjpd-weighted)
    # ===========================================================================
    logging.info("\n" + "-"*80)
    logging.info("DOSE-RESPONSE ANALYSIS (CONTINUOUS EXPOSURE)")
    logging.info("-"*80)
    
    # Use 30-minute threshold with adjpd weights for dose-response (matches main analysis)
    radius_miles_30 = 40.0 * 30 / 60.0
    radius_rad_30 = radius_miles_30 / EARTH_RADIUS_MILES
    weights_adjpd = weight_schemes['adjpd_weighted']
    
    indices_list_30 = tree.query_radius(cbg_coords_rad, r=radius_rad_30)
    cbg_exposure_30 = []
    for hosp_indices in indices_list_30:
        if len(hosp_indices) == 0:
            cbg_exposure_30.append(0.0)
        else:
            accessible_weight = weights_adjpd[hosp_indices].sum()
            total_weight = weights_adjpd.sum()
            cbg_exposure_30.append(100 * accessible_weight / total_weight if total_weight > 0 else 0.0)
    
    cbg_dose = pop_gdf.copy()
    cbg_dose['exposure'] = cbg_exposure_30
    cbg_dose['county_fips'] = cbg_dose['GEOID'].str[:5]
    
    county_dose = cbg_dose.groupby('county_fips').apply(
        lambda x: np.average(x['exposure'], weights=x['POPULATION']) if x['POPULATION'].sum() > 0 else 0.0
    ).reset_index()
    county_dose.columns = ['county_fips', 'exposure']
    
    # Merge with health data
    dose_df = health_data.merge(county_dose, on='county_fips', how='inner')
    dose_df = dose_df.dropna(subset=['ypll', 'exposure', 'social_economic_factors_score',
                                      'health_behaviors_score', 'physical_environment_score'])
    
    if len(dose_df) < 50:
        logging.warning("Insufficient data for dose-response analysis")
    else:
        # Fit spline regression for dose-response
        from scipy.interpolate import UnivariateSpline
        
        # Standardize covariates (but keep exposure continuous and unstandardized for interpretability)
        covariates_to_std = ['social_economic_factors_score', 'health_behaviors_score',
                             'physical_environment_score', 'population']
        scaler_dose = StandardScaler()
        dose_df[covariates_to_std] = scaler_dose.fit_transform(dose_df[covariates_to_std])
        
        # Run regression with continuous exposure
        formula_dose = "ypll ~ exposure + social_economic_factors_score + health_behaviors_score + physical_environment_score + population + C(census_division)"
        model_dose = smf.ols(formula_dose, data=dose_df).fit(cov_type='HC1')
        
        logging.info("\n--- Dose-Response Regression (Continuous Exposure) ---")
        logging.info(f"Exposure coefficient (linear): {model_dose.params['exposure']:.2f} (p = {model_dose.pvalues['exposure']:.4f})")
        logging.info(f"R-squared: {model_dose.rsquared:.3f}")
        
        # Generate predictions for dose-response curve
        exposure_range = np.linspace(dose_df['exposure'].min(), dose_df['exposure'].max(), 100)
        
        # Create prediction dataframe with mean values for other covariates
        pred_df = pd.DataFrame({
            'exposure': exposure_range,
            'social_economic_factors_score': dose_df['social_economic_factors_score'].mean(),
            'health_behaviors_score': dose_df['health_behaviors_score'].mean(),
            'physical_environment_score': dose_df['physical_environment_score'].mean(),
            'population': dose_df['population'].mean(),
            'census_division': dose_df['census_division'].mode()[0]
        })
        
        predictions = model_dose.predict(pred_df)
        pred_se = model_dose.get_prediction(pred_df).se_mean
        ci_low = predictions - 1.96 * pred_se
        ci_high = predictions + 1.96 * pred_se
        
        # Plot dose-response curve
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(exposure_range, predictions, color='navy', linewidth=2.5, label='Predicted YPLL')
        ax.fill_between(exposure_range, ci_low, ci_high, alpha=0.3, color='navy', label='95% CI')
        
        # Add scatter of binned means for visual reference
        dose_df['exposure_bin'] = pd.cut(dose_df['exposure'], bins=20)
        binned_means = dose_df.groupby('exposure_bin', observed=True).agg({
            'exposure': 'mean',
            'ypll': 'mean'
        }).dropna()
        ax.scatter(binned_means['exposure'], binned_means['ypll'], 
                  color='darkred', s=40, alpha=0.6, zorder=3, label='Binned means')
        
        ax.set_xlabel('AI Access (% of county population with 30-min access, adjpd-weighted)', fontsize=12)
        ax.set_ylabel('Premature Death (YPLL per 100,000)', fontsize=12)
        ax.set_title('Dose-Response Curve: AI Access vs. Premature Mortality\n(Controlling for SDOH and Census Division)', 
                    fontsize=14, pad=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Annotate with key statistics
        textstr = f'Linear slope: {model_dose.params["exposure"]:.1f} YPLL per 1% increase\np-value: {model_dose.pvalues["exposure"]:.4f}\nR² = {model_dose.rsquared:.3f}\nN = {len(dose_df)} counties'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        fig_dose_path = FIG_DIR / "dose_response_curve_ai_access.png"
        plt.savefig(fig_dose_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved dose-response curve to {fig_dose_path}")
        plt.close()
        
        # Test for monotonicity
        slope = np.diff(predictions)
        is_monotone = np.all(slope <= 0) or np.all(slope >= 0)
        logging.info(f"\nMonotonicity check: {'MONOTONE' if is_monotone else 'NON-MONOTONE'}")
        logging.info(f"Slope direction: {'Negative (decreasing YPLL)' if np.mean(slope) < 0 else 'Positive (increasing YPLL)'}")
        
        # Export dose-response data
        dose_response_data = pd.DataFrame({
            'Exposure_Percent': exposure_range,
            'Predicted_YPLL': predictions,
            'CI_Lower': ci_low,
            'CI_Upper': ci_high,
            'SE': pred_se
        })
        dose_response_data.to_csv(CSV_DIR / "dose_response_continuous_exposure.csv", index=False)
        logging.info(f"Exported dose-response curve data to {CSV_DIR / 'dose_response_continuous_exposure.csv'}")
    
    logging.info("\n" + "="*80)
    logging.info(" SENSITIVITY ANALYSIS COMPLETED ")
    logging.info("="*80)

def model_adoption_drivers(gdf):
    """Models AI/Robotics adoption based on competition and hospital size."""
    logging.info("\n" + "="*20 + " MODELING ADOPTION DRIVERS " + "="*20)
    
    adoption_results = []  # Collect results for export
    
    def run_logit(df, target_col, predictors):
        logging.info(f"--- Logistic Regression for: {target_col} ---")
        model_df = df[[target_col] + predictors].dropna()
        if model_df[target_col].nunique() < 2:
            logging.warning(f"Target '{target_col}' has only one class. Cannot build model.")
            return

        X = model_df[predictors].values
        y = model_df[target_col].values
        
        # Check for infinite or extremely large values
        if not np.all(np.isfinite(X)):
            logging.warning(f"Non-finite values detected in predictors for {target_col}. Cleaning data...")
            finite_mask = np.all(np.isfinite(X), axis=1)
            X = X[finite_mask]
            y = y[finite_mask]
            logging.info(f"Removed {(~finite_mask).sum()} rows with non-finite values.")
        
        if len(X) < 10:
            logging.warning(f"Insufficient data ({len(X)} samples) for {target_col} after cleaning. Skipping.")
            return
            
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use increased max_iter and solver that's more robust to numerical issues
            model = LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs'
            ).fit(X_scaled, y)
            
            coeffs = pd.Series(model.coef_[0], index=predictors)
            accuracy = model.score(X_scaled, y)
            
            logging.info(f"Predictors: {predictors}")
            logging.info(f"Model Accuracy: {accuracy:.3f}")
            logging.info("Coefficients (log-odds):\n" + coeffs.to_string())
            
            logging.info("NOTE on 'k3_avg_miles' coefficient: A positive coefficient suggests that hospitals with greater distances to their nearest neighbors (i.e., less competition) are more likely to adopt the technology. A negative coefficient would have suggested a competitive driver.")
            
            logging.info("Classification Report:\n" + classification_report(y, model.predict(X_scaled), digits=3, zero_division=0))
            
            # Store results for export
            for pred in predictors:
                adoption_results.append({
                    'Target': target_col,
                    'Predictor': pred,
                    'Coefficient_Log_Odds': coeffs[pred],
                    'N_Samples': len(X),
                    'Model_Accuracy': accuracy
                })
        
        except Exception as e:
            logging.error(f"Failed to fit logistic regression for {target_col}: {e}")
            logging.info("This may be due to perfect separation or numerical instability in the data.")

    predictors = ['k3_avg_miles', 'bsc']
    run_logit(gdf, 'ai_flag', predictors)
    run_logit(gdf, 'robo_flag', predictors)
    
    # Export adoption drivers results
    if adoption_results:
        adoption_df = pd.DataFrame(adoption_results)
        adoption_df.to_csv(CSV_DIR / "adoption_drivers_logistic_regression.csv", index=False)
        logging.info(f"Exported adoption drivers results to {CSV_DIR / 'adoption_drivers_logistic_regression.csv'}")


def perform_hotspot_analysis(hospital_gdf):
    """Performs Local Moran's I (LISA) analysis for adoption clusters."""
    logging.info("\n" + "="*20 + " HOT SPOT (LISA) ANALYSIS " + "="*20)
    if not PYSAL_AVAILABLE:
        logging.warning("PySAL not installed. Skipping hot spot analysis.")
        return
    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}. Skipping hot spot analysis.")
        return

    try:
        county_gdf = gpd.read_file(COUNTY_FILE)
        # Set CRS if missing (Census shapefiles are typically NAD83)
        if county_gdf.crs is None:
            logging.warning("County shapefile has no CRS defined. Setting to EPSG:4269 (NAD83).")
            county_gdf = county_gdf.set_crs("EPSG:4269")
        if 'GEOID' not in county_gdf.columns or 'STATEFP' not in county_gdf.columns:
            logging.error("County shapefile must have 'GEOID' and 'STATEFP' columns.")
            return

        hosp_proj = hospital_gdf.to_crs(county_gdf.crs)
        hosp_with_county = gpd.sjoin(hosp_proj, county_gdf[['GEOID', 'STATEFP', 'geometry']], how='left', predicate='within')
        county_agg = hosp_with_county.groupby('GEOID').agg(
            ai_count=('ai_flag', 'sum'),
            robo_count=('robo_flag', 'sum'),
            ai_intensity_sum=('ai_intensity', 'sum'),
            hospital_count=('hospital_id', 'count')
        ).reset_index()
        county_agg['ai_rate'] = (county_agg['ai_count'] / county_agg['hospital_count']).fillna(0)
        county_agg['robo_rate'] = (county_agg['robo_count'] / county_agg['hospital_count']).fillna(0)
        analysis_gdf_full = county_gdf.merge(county_agg, on='GEOID', how='left').fillna(0)
        
        analysis_gdf_full_proj = analysis_gdf_full.to_crs(AEA_CRS)

        def plot_lisa_map(gdf, lisa, title, filename):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            lisa_cluster(lisa, gdf, p=0.05, ax=ax, legend_kwds={'loc': 'lower left'})
            ax.set_title(title, fontsize=16)
            ax.set_axis_off()
            if CONTEXTILY_AVAILABLE:
                cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logging.info(f"Saved LISA map to {filename}")
            plt.close(fig)

        def build_canonical_weights(gdf_in):
            """
            Create globally normalized symmetric KNN weights and a row-standardized comparison.
            Global normalization here means every neighbor link shares the same constant weight,
            summing to 1 across the entire graph (not row-by-row).
            """
            # Use KNN k=5 on representative points
            gdf_points = gdf_in.copy()
            gdf_points['geometry'] = gdf_points.geometry.representative_point()
            
            w_bin = libpysal.weights.KNN.from_dataframe(gdf_points, k=5, use_index=False, silence_warnings=True)
            w_bin.transform = 'b'  # binary
            
            total_links = sum(len(v) for v in w_bin.neighbors.values())
            if total_links == 0:
                return w_bin, w_bin
            global_weight = 1.0 / total_links
            neighbors = {k: v[:] for k, v in w_bin.neighbors.items()}
            weights = {k: [global_weight] * len(v) for k, v in neighbors.items()}
            w_global = libpysal.weights.W(neighbors, weights, id_order=w_bin.id_order, silence_warnings=True)

            # Row standardization: each row sums to 1
            w_row = libpysal.weights.KNN.from_dataframe(gdf_points, k=5, use_index=False, silence_warnings=True)
            w_row.transform = 'r'
            return w_global, w_row

        def run_lisa_with_comparison(gdf_in, y_vals, label: str):
            """
            Run LISA with globally normalized weights and compare to row-standardized classifications.
            Change % is computed on significant clusters only (p<0.05); non-sig coded as 0.
            """
            w_global, w_row = build_canonical_weights(gdf_in)

            # Suppress expected numerical warnings from permutation testing
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
                lisa_global = esda.Moran_Local(y_vals, w_global)
                lisa_row = esda.Moran_Local(y_vals, w_row)

            sig_global = np.where(lisa_global.p_sim < 0.05, lisa_global.q, 0)
            sig_row = np.where(lisa_row.p_sim < 0.05, lisa_row.q, 0)
            change_pct = float(np.mean(sig_global != sig_row) * 100)
            logging.info(f"    {label}: HH/HL/LL/LH changed for {change_pct:.2f}% of counties (global vs row-std, p<0.05).")
            
            # Export LISA results for this variable
            lisa_results = pd.DataFrame({
                'County_Index': range(len(y_vals)),
                'Variable': label,
                'Value': y_vals,
                'Local_I': lisa_global.Is,
                'P_Value': lisa_global.p_sim,
                'Cluster_Type': lisa_global.q,
                'Significant': lisa_global.p_sim < 0.05
            })
            safe_label = label.replace(' ', '_').replace('/', '_')
            lisa_results.to_csv(CSV_DIR / f"lisa_results_{safe_label}.csv", index=False)
            logging.info(f"    Exported LISA results to {CSV_DIR / f'lisa_results_{safe_label}.csv'}")
            
            return lisa_global, change_pct

        for col in ['ai_rate', 'robo_rate', 'ai_intensity_sum']:
            logging.info(f"--- Running LISA for: {col} (Full US) ---")
            gdf = analysis_gdf_full_proj[analysis_gdf_full_proj.geometry.is_valid & ~analysis_gdf_full_proj.geometry.is_empty].copy()
            y = gdf[col].values
            lisa, change_pct = run_lisa_with_comparison(gdf, y, f"{col} full US")
            title = f"Hot Spots of {col.replace('_', ' ').title()} by County (Full US)"
            filename = FIG_DIR / f"lisa_hotspot_{col}_full_us.png"
            plot_lisa_map(gdf, lisa, title, filename)
            logging.info(f"    Note: canonical global-normalized weights used; row-std comparison change={change_pct:.2f}%.")

        logging.info("\n--- Creating focused hot spot map for Contiguous US ---")
        excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
        
        gdf_contig_proj = analysis_gdf_full_proj[~analysis_gdf_full_proj['STATEFP'].isin(excluded_fips)].copy()
        gdf_contig_proj = gdf_contig_proj[gdf_contig_proj.geometry.is_valid & ~gdf_contig_proj.geometry.is_empty]

        column_to_plot = 'ai_intensity_sum'
        logging.info(f"--- Running LISA for: {column_to_plot} (Contiguous US) ---")
        y_contig = gdf_contig_proj[column_to_plot].values
        lisa_contig, change_pct_contig = run_lisa_with_comparison(gdf_contig_proj, y_contig, f"{column_to_plot} contiguous")

        us_contig_basemap = analysis_gdf_full[~analysis_gdf_full['STATEFP'].isin(excluded_fips)]
        fig_contig, ax_contig = plt.subplots(1, 1, figsize=(12, 8))
        
        # Ensure basemap is in AEA projection for plotting
        us_contig_basemap = us_contig_basemap.to_crs(AEA_CRS)
        us_contig_basemap.plot(ax=ax_contig, color='#f0f0f0', edgecolor='white', linewidth=0.5)
        
        gdf_for_plot = gdf_contig_proj.to_crs(AEA_CRS)
        lisa_cluster(lisa_contig, gdf_for_plot, p=0.05, ax=ax_contig, legend_kwds={'loc': 'lower left'})
        
        ax_contig.set_title("Hot Spots of AI Adoption Intensity by County (Contiguous US)", fontsize=16)
        ax_contig.set_axis_off()
        # Let matplotlib auto-scale based on the data extent in AEA projection
        
        fig_path_contig = FIG_DIR / "lisa_hotspot_ai_intensity_sum_contiguous_us_clean.png"
        plt.savefig(fig_path_contig, dpi=300, bbox_inches='tight')
        logging.info(f"Saved CLEAN Contiguous US LISA cluster map to {fig_path_contig}")
        logging.info(f"    Contiguous change vs row-std weights: {change_pct_contig:.2f}% of counties reclassified.")
        plt.close(fig_contig)

    except Exception as e:
        logging.error(f"Error during hot spot analysis: {e}", exc_info=True)

# ======================= K-NN GRAPH ANALYSIS (REFINED) =======================
def create_knn_graph(gdf, k):
    """
    Creates a k-Nearest Neighbors graph from a GeoDataFrame.
    It's critical that the input gdf is already filtered to the desired geography.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame with point geometries.
        k (int): The number of nearest neighbors to find for each point.

    Returns:
        list: A list of tuples, where each tuple represents an edge
              connecting the integer-location (iloc) of two neighboring hospitals.
    """
    logging.info(f"\n" + "="*20 + f" CREATING K-NN GRAPH (k={k}) " + "="*20)
    if len(gdf) <= k:
        logging.warning(f"Number of hospitals ({len(gdf)}) is less than or equal to k ({k}). Cannot build graph.")
        return []

    # Reset index to ensure iloc-based results from BallTree are unambiguous
    gdf = gdf.reset_index(drop=True)

    coords_rad = np.deg2rad(gdf[['latitude', 'longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')

    # Query for k+1 neighbors because the first neighbor is the point itself
    # The `indices` array will contain the row's integer position for each neighbor
    _, indices = tree.query(coords_rad, k=k+1)

    # The indices array has shape (n_points, k+1)
    source_nodes = indices[:, 0]
    neighbor_nodes = indices[:, 1:]

    edges = []
    for i in range(len(source_nodes)):
        source_iloc = source_nodes[i]
        for j in range(k):
            neighbor_iloc = neighbor_nodes[i, j]
            edges.append((source_iloc, neighbor_iloc))

    logging.info(f"Successfully created k-NN graph with {len(gdf)} nodes and {len(edges)} edges.")
    
    # Export k-NN graph statistics
    knn_stats = pd.DataFrame({
        'Metric': ['K_Value', 'Number_of_Nodes', 'Number_of_Edges', 'Avg_Degree'],
        'Value': [k, len(gdf), len(edges), len(edges) / len(gdf) if len(gdf) > 0 else 0]
    })
    knn_stats.to_csv(CSV_DIR / f"knn_graph_statistics_k{k}.csv", index=False)
    logging.info(f"Exported k-NN graph statistics to {CSV_DIR / f'knn_graph_statistics_k{k}.csv'}")
    
    # Export edge list
    edges_df = pd.DataFrame(edges, columns=['Source_Node_Index', 'Target_Node_Index'])
    edges_df.to_csv(CSV_DIR / f"knn_graph_edges_k{k}.csv", index=False)
    logging.info(f"Exported k-NN graph edge list to {CSV_DIR / f'knn_graph_edges_k{k}.csv'}")
    
    return edges

def plot_knn_graph(gdf, edges, k, filename, title):
    """
    Generates a publication-quality visualization of the k-NN graph.
    (Simplified version assumes gdf and edges are already for the desired geography)

    Args:
        gdf (GeoDataFrame): The hospital data (pre-filtered for contiguous US).
        edges (list): The list of graph edges (from create_knn_graph).
        k (int): The 'k' value for titling.
        filename (str or Path): The path to save the output figure.
        title (str): The title for the plot.
    """
    logging.info(f"Generating visualization for the k={k} graph...")

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping k-NN map.")
        return

    # Reset index of the plotting GDF to match the iloc-based edges
    gdf = gdf.reset_index(drop=True)

    # --- Prepare Basemap ---
    us_basemap = gpd.read_file(COUNTY_FILE)
    # Set CRS if missing (Census shapefiles are typically NAD83)
    if us_basemap.crs is None:
        logging.warning("County shapefile has no CRS defined. Setting to EPSG:4269 (NAD83).")
        us_basemap = us_basemap.set_crs("EPSG:4269")
    
    # Filter to contiguous US if STATEFP column exists
    if 'STATEFP' in us_basemap.columns:
        excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
        us_contig_basemap = us_basemap[~us_basemap['STATEFP'].isin(excluded_fips)]
    else:
        logging.warning("County shapefile missing STATEFP column. Using all geometries for basemap.")
        us_contig_basemap = us_basemap

    # --- Create the Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # 1. Plot the basemap (projected)
    us_contig_basemap = us_contig_basemap.to_crs(AEA_CRS)
    us_contig_basemap.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)

    # 2. Plot the k-NN graph edges
    # The indices in `edges` now directly and correctly correspond to the rows in `gdf`
    # Ensure gdf is projected for plotting
    gdf_proj = gdf.to_crs(AEA_CRS)
    
    for start_iloc, end_iloc in tqdm(edges, desc="Plotting graph edges"):
        start_geom = gdf_proj.iloc[start_iloc].geometry
        end_geom = gdf_proj.iloc[end_iloc].geometry
        ax.plot(
            [start_geom.x, end_geom.x],
            [start_geom.y, end_geom.y],
            color='darkgray',
            linewidth=0.5,
            alpha=0.6,
            zorder=1
        )

    # 3. Plot the hospital locations (nodes) on top
    gdf_proj.plot(
        ax=ax,
        column='tech_type',
        categorical=True,
        legend=True,
        markersize=25,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.5,
        zorder=2,
        legend_kwds={'title': "Technology Type", 'loc': 'lower left'}
    )

    # 4. Final Touches
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    # Let matplotlib auto-scale based on the data extent in AEA projection

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logging.info(f"Saved k-NN graph visualization to {filename}")
    plt.close(fig)

# ======================= SUTVA TESTING =======================
def test_sutva_assumptions(engine, hospital_gdf, pop_gdf):
    """
    Tests for violations of the Stable Unit Treatment Value Assumption (SUTVA).
    
    SUTVA requires:
    1. No interference: One unit's treatment doesn't affect another's outcome
    2. No hidden treatment versions: Treatment is uniform across all treated units
    
    This function implements heuristic checks for both components.
    """
    logging.info("\n" + "="*20 + " SUTVA VIOLATION TESTING " + "="*20)
    
    if pop_gdf is None or not PYSAL_AVAILABLE:
        logging.warning("Population data or PySAL not available. Skipping SUTVA tests.")
        return
    
    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found. Skipping SUTVA tests.")
        return
    
    try:
        # --- Load county-level health data ---
        query = """
        SELECT
            m.county_fips,
            v.premature_death_raw_value as ypll,
            m.social_economic_factors_score,
            m.health_behaviors_score,
            m.physical_environment_score,
            m.population
        FROM vw_conceptual_model_adjpd m
        JOIN vw_conceptual_model_variables v ON m.county_fips = v.county_fips;
        """
        health_data = pd.read_sql(query, engine)
        
        # --- Calculate county-level treatment status ---
        logging.info("Calculating county-level AI adoption metrics...")
        
        # Spatial join hospitals to counties
        county_gdf = gpd.read_file(COUNTY_FILE)
        # Set CRS if missing (Census shapefiles are typically NAD83)
        if county_gdf.crs is None:
            logging.warning("County shapefile has no CRS defined. Setting to EPSG:4269 (NAD83).")
            county_gdf = county_gdf.set_crs("EPSG:4269")
        
        # Check for required columns
        if 'GEOID' not in county_gdf.columns or 'STATEFP' not in county_gdf.columns:
            logging.error("County shapefile is incomplete - missing GEOID/STATEFP columns. Cannot perform SUTVA testing.")
            return
            
        hosp_proj = hospital_gdf.to_crs(county_gdf.crs)
        hosp_with_county = gpd.sjoin(hosp_proj, county_gdf[['GEOID', 'STATEFP', 'geometry']], 
                                     how='left', predicate='within')
        
        # Aggregate to county level
        county_tech = hosp_with_county.groupby('GEOID').agg(
            ai_count=('ai_flag', 'sum'),
            robo_count=('robo_flag', 'sum'),
            hospital_count=('hospital_id', 'count'),
            ai_intensity_mean=('ai_intensity', 'mean'),
            bed_size_mean=('bsc', 'mean')
        ).reset_index()
        
        county_tech['ai_proportion'] = county_tech['ai_count'] / county_tech['hospital_count']
        county_tech['has_ai'] = (county_tech['ai_count'] > 0).astype(int)
        
        # Merge with county geography and health data
        analysis_gdf = county_gdf.merge(county_tech, left_on='GEOID', right_on='GEOID', how='left')
        analysis_gdf = analysis_gdf.merge(health_data, left_on='GEOID', right_on='county_fips', how='inner')
        analysis_gdf = analysis_gdf.fillna(0)
        
        # Focus on contiguous US
        excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
        analysis_gdf = analysis_gdf[~analysis_gdf['STATEFP'].isin(excluded_fips)].copy()
        
        logging.info(f"Analyzing {len(analysis_gdf)} counties for SUTVA violations.")
        
        # ============================================================
        # TEST 1: SPILLOVER EFFECTS (Interference)
        # ============================================================
        logging.info("\n" + "-"*50)
        logging.info("TEST 1: SPILLOVER/INTERFERENCE DETECTION")
        logging.info("-"*50)
        
        # Create spatial weights matrix (KNN k=5 on representative points for better spillover proxy)
        # Calculate representative points for distance-based weights
        # We create a temporary dataframe with points to ensure KNN uses point distances
        centroid_gdf = analysis_gdf.copy()
        centroid_gdf['geometry'] = centroid_gdf.geometry.representative_point()
        
        # Use KNN weights (k=5) instead of Queen contiguity
        # This captures "nearest neighbors" regardless of shared borders (better for hospital access)
        w = libpysal.weights.KNN.from_dataframe(centroid_gdf, k=5, use_index=False, silence_warnings=True)
        
        # Row-standardize weights so lag represents AVERAGE neighbor intensity, not sum
        w.transform = 'r'
        
        # Calculate spatial lag of treatment (neighbor treatment intensity)
        analysis_gdf['neighbor_ai_proportion'] = libpysal.weights.lag_spatial(w, analysis_gdf['ai_proportion'].values)
        analysis_gdf['neighbor_has_ai'] = (analysis_gdf['neighbor_ai_proportion'] > 0).astype(int)
        
        # Test 1a: Compare treated units with high vs. low neighbor treatment
        treated_counties = analysis_gdf[analysis_gdf['has_ai'] == 1].copy()
        
        if len(treated_counties) >= 10:
            high_neighbor_treatment = treated_counties['neighbor_ai_proportion'] > treated_counties['neighbor_ai_proportion'].median()
            
            ypll_high_neighbor = treated_counties[high_neighbor_treatment]['ypll'].mean()
            ypll_low_neighbor = treated_counties[~high_neighbor_treatment]['ypll'].mean()
            
            logging.info("\n>>> Test 1a: Outcome differences by neighbor treatment intensity")
            logging.info(f"Among treated counties:")
            logging.info(f"  - YPLL when surrounded by high-treatment neighbors: {ypll_high_neighbor:.1f}")
            logging.info(f"  - YPLL when surrounded by low-treatment neighbors:  {ypll_low_neighbor:.1f}")
            logging.info(f"  - Difference: {ypll_high_neighbor - ypll_low_neighbor:.1f}")
            
            if abs(ypll_high_neighbor - ypll_low_neighbor) > 500:
                logging.warning("  ⚠️  POTENTIAL SUTVA VIOLATION: Large outcome difference suggests spillover effects!")
            else:
                logging.info("  ✓ No strong evidence of spillover (difference < 500 YPLL)")
        
        # Test 1b: Spatial autocorrelation of residuals
        # Run a simple model and check if residuals are spatially correlated
        from statsmodels.regression.linear_model import OLS
        
        model_vars = ['has_ai', 'social_economic_factors_score', 'health_behaviors_score', 
                     'physical_environment_score']
        model_df = analysis_gdf[['ypll'] + model_vars].dropna()
        
        if len(model_df) >= 50:
            X = sm.add_constant(model_df[model_vars])
            y = model_df['ypll']
            
            naive_model = OLS(y, X).fit()
            residuals = naive_model.resid
            
            # Subset weights to match model_df
            # Use KNN on representative points for consistency with Test 1a
            subset_gdf = analysis_gdf.loc[model_df.index].copy()
            subset_gdf['geometry'] = subset_gdf.geometry.representative_point()
            
            w_subset = libpysal.weights.KNN.from_dataframe(
                subset_gdf, 
                k=5, 
                use_index=False, 
                silence_warnings=True
            )
            w_subset.transform = 'r'
            
            moran = esda.Moran(residuals.values, w_subset)
            
            logging.info("\n>>> Test 1b: Spatial autocorrelation of regression residuals")
            logging.info(f"  - Moran's I statistic: {moran.I:.4f}")
            logging.info(f"  - p-value: {moran.p_sim:.4f}")
            
            if moran.p_sim < 0.05 and moran.I > 0.1:
                logging.warning("  ⚠️  POTENTIAL SUTVA VIOLATION: Residuals show spatial clustering!")
                logging.warning("     This suggests unmeasured spillover effects between neighboring counties.")
            else:
                logging.info("  ✓ No strong spatial autocorrelation in residuals")
        
        # Test 1c: Border discontinuity test
        logging.info("\n>>> Test 1c: Treatment discontinuity at borders")
        
        # Find county pairs that share a border but differ in treatment
        border_pairs = []
        for i, county in analysis_gdf.iterrows():
            neighbors_idx = w[analysis_gdf.index.get_loc(i)]
            for n_idx in neighbors_idx:
                neighbor = analysis_gdf.iloc[n_idx]
                if county['has_ai'] != neighbor['has_ai']:
                    border_pairs.append({
                        'county1_ypll': county['ypll'],
                        'county2_ypll': neighbor['ypll'],
                        'treated_ypll': county['ypll'] if county['has_ai'] else neighbor['ypll'],
                        'control_ypll': neighbor['ypll'] if county['has_ai'] else county['ypll']
                    })
        
        if len(border_pairs) >= 10:
            border_df = pd.DataFrame(border_pairs)
            ypll_diff = border_df['treated_ypll'].mean() - border_df['control_ypll'].mean()
            
            logging.info(f"  - Found {len(border_pairs)} adjacent county pairs with different treatment status")
            logging.info(f"  - Avg YPLL difference at borders: {ypll_diff:.1f}")
            logging.info(f"  - If spillover exists, this should be smaller than the true treatment effect")
        
        # ============================================================
        # TEST 2: TREATMENT VERSION HETEROGENEITY
        # ============================================================
        logging.info("\n" + "-"*50)
        logging.info("TEST 2: HIDDEN TREATMENT VERSIONS")
        logging.info("-"*50)
        
        # Test 2a: Heterogeneity by hospital size
        treated_hospitals = hosp_with_county[hosp_with_county['ai_flag'] == 1].copy()
        
        if len(treated_hospitals) >= 20:
            treated_hospitals['size_category'] = pd.qcut(
                treated_hospitals['bsc'], 
                q=3, 
                labels=['Small', 'Medium', 'Large'],
                duplicates='drop'
            )
            
            # Aggregate back to counties by size category
            county_by_size = treated_hospitals.groupby(['GEOID', 'size_category']).size().reset_index(name='count')
            county_by_size = county_by_size.sort_values('count', ascending=False).drop_duplicates('GEOID')
            
            county_outcomes = analysis_gdf[['GEOID', 'ypll']].merge(
                county_by_size[['GEOID', 'size_category']], 
                on='GEOID', 
                how='inner'
            )
            
            logging.info("\n>>> Test 2a: Outcome variation by dominant hospital size")
            for size in ['Small', 'Medium', 'Large']:
                size_ypll = county_outcomes[county_outcomes['size_category'] == size]['ypll'].mean()
                size_count = len(county_outcomes[county_outcomes['size_category'] == size])
                logging.info(f"  - Counties with mostly {size} AI hospitals: YPLL = {size_ypll:.1f} (n={size_count})")
            
            # Test for significant heterogeneity
            from scipy import stats
            size_groups = [county_outcomes[county_outcomes['size_category'] == s]['ypll'].dropna() 
                          for s in ['Small', 'Medium', 'Large'] 
                          if len(county_outcomes[county_outcomes['size_category'] == s]) > 0]
            
            if len(size_groups) >= 2:
                f_stat, p_val = stats.f_oneway(*size_groups)
                logging.info(f"\n  - ANOVA F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}")
                
                if p_val < 0.05:
                    logging.warning("  ⚠️  POTENTIAL SUTVA VIOLATION: Treatment effects vary by hospital size!")
                    logging.warning("     This suggests 'AI adoption' is not a uniform treatment.")
                else:
                    logging.info("  ✓ No significant heterogeneity by hospital size")
        
        # Test 2b: Heterogeneity by AI intensity
        logging.info("\n>>> Test 2b: Dose-response relationship (AI intensity)")
        
        treated_with_intensity = analysis_gdf[analysis_gdf['has_ai'] == 1].copy()
        
        if len(treated_with_intensity) >= 20:
            treated_with_intensity['intensity_tertile'] = pd.qcut(
                treated_with_intensity['ai_intensity_mean'],
                q=3,
                labels=['Low', 'Medium', 'High'],
                duplicates='drop'
            )
            
            for tertile in ['Low', 'Medium', 'High']:
                tertile_data = treated_with_intensity[treated_with_intensity['intensity_tertile'] == tertile]
                if len(tertile_data) > 0:
                    ypll = tertile_data['ypll'].mean()
                    logging.info(f"  - {tertile} AI intensity counties: YPLL = {ypll:.1f} (n={len(tertile_data)})")
            
            # Check for monotonic relationship
            low_ypll = treated_with_intensity[treated_with_intensity['intensity_tertile'] == 'Low']['ypll'].mean()
            high_ypll = treated_with_intensity[treated_with_intensity['intensity_tertile'] == 'High']['ypll'].mean()
            
            if not pd.isna(low_ypll) and not pd.isna(high_ypll):
                if high_ypll > low_ypll:
                    logging.warning("  ⚠️  WARNING: Higher AI intensity associated with WORSE outcomes!")
                    logging.warning("     This could indicate treatment heterogeneity or confounding.")
                else:
                    logging.info("  ✓ Dose-response relationship is in expected direction")
        
        # Test 2c: Implementation heterogeneity check
        logging.info("\n>>> Test 2c: Technology type heterogeneity")
        
        # Check if different AI technologies show different patterns
        ai_tech_cols = ['wfaipsn', 'wfaippd', 'wfaiss', 'wfaiart', 'wfaioacw']
        
        tech_patterns = []
        for col in ai_tech_cols:
            has_tech = hosp_with_county[hosp_with_county[col] > 0].groupby('GEOID').size()
            counties_with_tech = analysis_gdf[analysis_gdf['GEOID'].isin(has_tech.index)]
            
            if len(counties_with_tech) >= 5:
                avg_ypll = counties_with_tech['ypll'].mean()
                tech_patterns.append({
                    'technology': col,
                    'n_counties': len(counties_with_tech),
                    'avg_ypll': avg_ypll
                })
        
        if tech_patterns:
            tech_df = pd.DataFrame(tech_patterns)
            logging.info("\n  Average YPLL by AI technology type:")
            for _, row in tech_df.iterrows():
                logging.info(f"    - {row['technology']}: {row['avg_ypll']:.1f} (n={row['n_counties']})")
            
            ypll_range = tech_df['avg_ypll'].max() - tech_df['avg_ypll'].min()
            if ypll_range > 1000:
                logging.warning(f"  ⚠️  Large variation in outcomes across tech types (range: {ypll_range:.1f})")
                logging.warning("     This suggests different AI technologies may have different effects.")
        
        # ============================================================
        # SUMMARY RECOMMENDATIONS
        # ============================================================
        logging.info("\n" + "="*50)
        logging.info("SUTVA TESTING SUMMARY")
        logging.info("="*50)
        logging.info("\nInterpretation Guidelines:")
        logging.info("  - Spillover effects would violate SUTVA's no-interference assumption")
        logging.info("  - Treatment heterogeneity violates SUTVA's uniform treatment assumption")
        logging.info("  - Both types of violations can bias causal effect estimates")
        logging.info("\nRecommended Actions if violations detected:")
        logging.info("  1. Use spatial econometric models (SAR, SEM) for spillover")
        logging.info("  2. Stratify analyses by treatment implementation details")
        logging.info("  3. Report E-values for unmeasured confounding sensitivity")
        logging.info("  4. Acknowledge SUTVA limitations in study interpretation")
        
        # Export SUTVA test results summary
        sutva_summary = pd.DataFrame({
            'Test': [
                'Spatial_Lag_AI_Exposure',
                'Moran_I_Residuals',
                'Treatment_Heterogeneity_Size',
                'Treatment_Heterogeneity_Intensity'
            ],
            'Description': [
                'Spillover from neighboring counties AI exposure',
                'Spatial autocorrelation in regression residuals',
                'Outcome variation by hospital size category',
                'Dose-response relationship by AI intensity'
            ],
            'Status': [
                'See log for spatial lag coefficient',
                'See log for Moran I statistic',
                'See log for ANOVA results',
                'See log for tertile comparison'
            ]
        })
        sutva_summary.to_csv(CSV_DIR / "sutva_violation_tests_summary.csv", index=False)
        logging.info(f"\nExported SUTVA test summary to {CSV_DIR / 'sutva_violation_tests_summary.csv'}")
        
    except Exception as e:
        logging.error(f"Error during SUTVA testing: {e}", exc_info=True)

# ======================= YPLL ANALYSIS & VISUALIZATION =======================
def analyze_ypll_and_technology_access(engine, hospital_gdf, pop_gdf):
    """
    Performs a county-level multiple regression to test if access to technology
    predicts premature death (YPLL), controlling for composite SDOH indices,
    population, and census division.
    """
    logging.info("\n" + "="*20 + " YPLL REGRESSION ANALYSIS (V2) " + "="*20)
    
    # Check if population data is available
    if pop_gdf is None:
        logging.error("Population data not available. Skipping YPLL regression analysis.")
        return
    
    # --- 1. Load County-Level Health and Socioeconomic Data (New Model) ---
    logging.info("Loading county-level data for new regression model.")
    query = """
    SELECT
        m.county_fips,
        m.social_economic_factors_score, -- IV4
        m.health_behaviors_score,        -- IV3
        m.physical_environment_score,    -- IV2
        m.population,                    -- ct1
        m.census_division,               -- ct2
        v.premature_death_raw_value      -- DV21 (YPLL)
    FROM vw_conceptual_model_adjpd m
    JOIN vw_conceptual_model_variables v ON m.county_fips = v.county_fips;
    """
    try:
        health_data = pd.read_sql(query, engine)
        health_data.rename(columns={'premature_death_raw_value': 'ypll'}, inplace=True)
        logging.info(f"Successfully loaded {len(health_data)} county records for new YPLL analysis.")
    except Exception as e:
        logging.error(f"FATAL: Could not load YPLL data for new model. Skipping analysis. Error: {e}")
        return

    # --- 2. Calculate County-Level Technology Access Metrics (Same as before) ---
    logging.info("Aggregating CBG-level distance metrics to county-level.")
    if 'GEOID' not in pop_gdf.columns:
        logging.error("FATAL: Population GeoJSON must contain a 'GEOID' column for county aggregation. Skipping.")
        return

    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1]
    hospitals_robo = hospital_gdf[hospital_gdf['robo_flag'] == 1]
    
    cbg_gdf_proj = pop_gdf.to_crs(AEA_CRS)
    cbg_centroids_proj = cbg_gdf_proj.geometry.representative_point()
    cbg_centroids_wgs84 = cbg_centroids_proj.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())

    def get_nearest_distances(tech_gdf, cbg_coords, tech_name):
        tech_coords_rad = np.deg2rad(tech_gdf[['latitude', 'longitude']].values)
        tree = BallTree(tech_coords_rad, metric='haversine')
        distances_rad, _ = tree.query(cbg_coords, k=1)
        return distances_rad.flatten() * EARTH_RADIUS_MILES
    
    cbg_gdf = pop_gdf.copy()
    cbg_gdf['dist_ai'] = get_nearest_distances(hospitals_ai, cbg_coords_rad, 'AI')
    cbg_gdf['dist_robo'] = get_nearest_distances(hospitals_robo, cbg_coords_rad, 'Robotics')
    cbg_gdf['county_fips'] = cbg_gdf['GEOID'].str[:5]

    cbg_gdf['pop_dist_ai'] = cbg_gdf['dist_ai'] * cbg_gdf['POPULATION']
    cbg_gdf['pop_dist_robo'] = cbg_gdf['dist_robo'] * cbg_gdf['POPULATION']
    
    county_access = cbg_gdf.groupby('county_fips').agg(
        pop_dist_ai_sum=('pop_dist_ai', 'sum'),
        pop_dist_robo_sum=('pop_dist_robo', 'sum'),
        total_pop=('POPULATION', 'sum')
    ).reset_index()

    county_access['county_avg_dist_to_ai'] = county_access['pop_dist_ai_sum'] / county_access['total_pop']
    county_access['county_avg_dist_to_robo'] = county_access['pop_dist_robo_sum'] / county_access['total_pop']
    logging.info(f"Calculated population-weighted average access distances for {len(county_access)} counties.")
    
    # --- Export County-Level Distance Metrics to CSV ---
    county_distance_export = county_access[['county_fips', 'county_avg_dist_to_ai', 'county_avg_dist_to_robo', 'total_pop']].copy()
    county_distance_export.columns = [
        'county_fips',
        'pop_weighted_distance_to_ai_miles',
        'pop_weighted_distance_to_robotics_miles',
        'total_population'
    ]
    county_distance_export = county_distance_export.sort_values('county_fips')
    
    county_distance_path = CSV_DIR / "county_population_weighted_distances.csv"
    county_distance_export.to_csv(county_distance_path, index=False)
    logging.info(f"✓ Exported county-level distance metrics to {county_distance_path}")
    logging.info(f"  Contains {len(county_distance_export)} counties with population-weighted distances to AI and robotics hospitals.")
    
    # --- 3. Merge Datasets and Prepare for Regression ---
    final_df = pd.merge(health_data, county_access[['county_fips', 'county_avg_dist_to_ai', 'county_avg_dist_to_robo']], on='county_fips', how='inner')
    
    cols_to_standardize = [
        'county_avg_dist_to_ai', 'county_avg_dist_to_robo',
        'social_economic_factors_score', 'health_behaviors_score',
        'physical_environment_score', 'population'
    ]
    for col in cols_to_standardize + ['ypll']:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
    final_df.dropna(subset=['ypll', 'census_division'] + cols_to_standardize, inplace=True)
    
    # Check for infinite or extreme values that could cause numerical issues
    for col in cols_to_standardize + ['ypll']:
        if not np.all(np.isfinite(final_df[col])):
            logging.warning(f"Non-finite values detected in '{col}'. Removing affected rows.")
            final_df = final_df[np.isfinite(final_df[col])]
    
    if len(final_df) < 50:
        logging.error(f"Insufficient data after cleaning ({len(final_df)} counties). Skipping regression.")
        return
    
    df_for_simulation = final_df.copy()

    scaler = StandardScaler()
    final_df[cols_to_standardize] = scaler.fit_transform(final_df[cols_to_standardize])
    logging.info(f"Final dataset for regression has {len(final_df)} counties after cleaning.")
    
    ai_dist_idx = scaler.feature_names_in_.tolist().index('county_avg_dist_to_ai')
    ai_dist_sd = scaler.scale_[ai_dist_idx]
    logging.info(f"NOTE: For this model, one standard deviation in 'county_avg_dist_to_ai' is equal to {ai_dist_sd:.2f} miles.")

    # --- 4. Build and Run the OLS Regression Model ---
    formula = "ypll ~ " + " + ".join(cols_to_standardize) + " + C(census_division)"
    model = smf.ols(formula, data=final_df).fit(cov_type='HC1') # Use robust standard errors
    
    logging.info("--- OLS Regression Results (V2): Predicting YPLL ---")
    logging.info(f"\n{model.summary()}\n")
    
    # Export OLS regression results
    regression_results = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        'T_Statistic': model.tvalues.values,
        'P_Value': model.pvalues.values,
        'CI_Lower': model.conf_int()[0].values,
        'CI_Upper': model.conf_int()[1].values
    })
    regression_summary = pd.DataFrame({
        'Metric': ['R_Squared', 'Adj_R_Squared', 'F_Statistic', 'F_Pvalue', 'N_Observations', 'DF_Residuals'],
        'Value': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue, model.nobs, model.df_resid]
    })
    
    regression_results.to_csv(CSV_DIR / "ypll_regression_coefficients.csv", index=False)
    regression_summary.to_csv(CSV_DIR / "ypll_regression_summary.csv", index=False)
    logging.info(f"Exported regression results to {CSV_DIR / 'ypll_regression_coefficients.csv'}")
    logging.info(f"Exported regression summary to {CSV_DIR / 'ypll_regression_summary.csv'}")
    
    # --- DIAGNOSTIC: Check physical_environment coefficient ---
    logging.info("\n" + "="*60)
    logging.info("DIAGNOSTIC: Physical Environment Coefficient")
    logging.info("="*60)
    phys_env_coef = model.params.get('physical_environment_score', np.nan)
    phys_env_pval = model.pvalues.get('physical_environment_score', np.nan)
    phys_env_stderr = model.bse.get('physical_environment_score', np.nan)
    logging.info(f"  Coefficient: {phys_env_coef:.4f}")
    logging.info(f"  Std Error:   {phys_env_stderr:.4f}")
    logging.info(f"  P-value:     {phys_env_pval:.4f}")
    logging.info(f"  Significant: {'Yes' if phys_env_pval < 0.05 else 'No'}")
    if phys_env_coef > 0:
        logging.info(f"  Interpretation: Higher physical environment score → Higher YPLL (WORSE health)")
        logging.info(f"                  Check CHR score directionality - this may be a scale issue!")
    elif phys_env_coef < 0:
        logging.info(f"  Interpretation: Higher physical environment score → Lower YPLL (BETTER health)")
    else:
        logging.info(f"  Interpretation: No association detected")
    logging.info(f"  Note: If this coefficient is near-zero or positive, the Shapley decomposition")
    logging.info(f"        will correctly assign 0% or negative contribution to physical environment,")
    logging.info(f"        indicating it's either non-contributory or a suppressor variable.")
    logging.info("="*60 + "\n")
    
    # --- 5. Visualize Key Coefficients with Annotations ---
    logging.info("Generating annotated coefficient plot for key model variables.")
    
    coef_df = pd.DataFrame({
        'coef': model.params,
        'err': model.bse,
        'pvalue': model.pvalues
    })
    
    plot_vars = [v for v in coef_df.index if 'Intercept' not in v and 'census_division' not in v]
    coef_to_plot = coef_df.loc[plot_vars].copy()
    
    coef_to_plot['variable'] = [
        'Access: Avg. Dist. to AI', 'Access: Avg. Dist. to Robotics',
        'Socioeconomic Factors Score', 'Health Behaviors Score',
        'Physical Environment Score', 'Population'
    ]
    coef_to_plot['error_margin'] = coef_to_plot['err'] * 1.96

    fig, ax = plt.subplots(figsize=(12, 8))
    coef_to_plot.plot(kind='barh', x='variable', y='coef', xerr='error_margin',
                      ax=ax, legend=False, color=sns.color_palette("viridis", len(coef_to_plot)))
    
    ax.axvline(x=0, color='black', linestyle='--')
    
    for i, row in coef_to_plot.iterrows():
        p_val_text = f"p < 0.001" if row['pvalue'] < 0.001 else f"p = {row['pvalue']:.3f}"
        annotation = f"β = {row['coef']:.0f}\n{p_val_text}"
        text_x_pos = row['coef'] + row['error_margin'] + 50 if row['coef'] >= 0 else row['coef'] - row['error_margin'] - 50
        ha = 'left' if row['coef'] >= 0 else 'right'
        ax.text(text_x_pos, ax.get_yticks()[coef_to_plot.index.get_loc(i)], annotation, 
                va='center', ha=ha, fontsize=9)

    ax.set_title('Effect of Technology Access and SDOH on Premature Death (YPLL)', fontsize=16, pad=20)
    ax.set_xlabel('Change in YPLL (per 1 SD change in variable)', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(ax.get_xlim()[0] * 1.3, ax.get_xlim()[1] * 1.3)
    plt.tight_layout()
    
    fig_path = FIG_DIR / "ypll_regression_coefficients_v2.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved annotated YPLL regression coefficient plot to {fig_path}")
    plt.close(fig)
    
    # --- 6. Run the 'What-If' Simulation & National Impact Estimate ---
    deserts_df, oases_df = run_what_if_simulation(model, df_for_simulation, scaler)
    calculate_national_impact_estimate(model, df_for_simulation, scaler)
    
    # Store model artifacts for Sankey diagram (will be used in main())
    return model, df_for_simulation, scaler, deserts_df, oases_df

def run_what_if_simulation(model, df_unscaled, scaler):
    """
    Runs a 'what-if' simulation to estimate how much of the mortality gap
    could be reduced by improving technology access (keeping SDOH constant).
    
    Returns:
        tuple: (deserts_df, oases_df) for use in empirical IV decomposition
    """
    logging.info("\n" + "="*20 + " 'WHAT-IF' SIMULATION " + "="*20)
    logging.info("Estimating the impact of improving technology access on the mortality gap between high- and low-access counties.")

    desert_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.90)
    oasis_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.10)
    deserts_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] >= desert_threshold]
    oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] <= oasis_threshold]

    logging.info(f"Identified {len(deserts_df)} 'Technology Desert' counties (worst 10% access).")
    logging.info(f"Identified {len(oases_df)} 'Technology Oasis' counties (best 10% access).")

    ypll_desert_actual = deserts_df['ypll'].mean()
    ypll_oasis_actual = oases_df['ypll'].mean()
    observed_mortality_gap = ypll_desert_actual - ypll_oasis_actual

    logging.info(f"\n[STEP 1] Observed Data:")
    logging.info(f"  - Avg. YPLL in Technology Deserts: {ypll_desert_actual:,.0f}")
    logging.info(f"  - Avg. YPLL in Technology Oases:  {ypll_oasis_actual:,.0f}")
    logging.info(f"  - Total Observed Mortality Gap:   {observed_mortality_gap:,.0f} years of potential life lost")

    model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
    typical_desert_profile = deserts_df[model_predictors].mean().to_frame().T
    hypothetical_profile = typical_desert_profile.copy()
    hypothetical_profile['county_avg_dist_to_ai'] = oases_df['county_avg_dist_to_ai'].mean()
    
    logging.info(f"\n[STEP 2] Hypothetical Scenario:")
    logging.info("  - Simulating a county with 'Desert-level' socioeconomic factors but 'Oasis-level' AI access.")

    cols_to_scale = scaler.feature_names_in_
    hypothetical_profile_scaled = scaler.transform(hypothetical_profile[cols_to_scale])
    hypothetical_df_scaled = pd.DataFrame(hypothetical_profile_scaled, columns=cols_to_scale)
    hypothetical_df_scaled['census_division'] = df_unscaled['census_division'].mode()[0]
    predicted_ypll = model.predict(hypothetical_df_scaled)[0]

    ypll_reduction_from_technology = ypll_desert_actual - predicted_ypll
    pct_gap_explained_by_technology = (ypll_reduction_from_technology / observed_mortality_gap) * 100 if observed_mortality_gap > 0 else 0

    logging.info(f"\n[STEP 3] Simulation Results:")
    logging.info(f"  - The actual YPLL in a typical desert county is {ypll_desert_actual:,.0f}.")
    logging.info(f"  - The model predicts that if this county got best-in-class AI access, its YPLL would fall to {predicted_ypll:,.0f}.")
    logging.info(f"  - This implies that improving technology access could reduce YPLL by {ypll_reduction_from_technology:,.0f}.")
    logging.info(f"\n[CONCLUSION] >> Approximately {pct_gap_explained_by_technology:.1f}% of the mortality gap between the best- and worst-access counties could be addressed by improving technology access to best-in-class levels (while {100-pct_gap_explained_by_technology:.1f}% remains attributable to differences in SDOH).")
    logging.info("="*52)
    
    # Export what-if simulation results
    simulation_results = pd.DataFrame({
        'Metric': [
            'Desert_Counties_Count',
            'Oasis_Counties_Count',
            'Avg_YPLL_Desert_Actual',
            'Avg_YPLL_Oasis_Actual',
            'Observed_Mortality_Gap',
            'Predicted_YPLL_After_Intervention',
            'YPLL_Reduction_From_Technology',
            'Percent_Gap_Explained_By_Technology',
            'Percent_Gap_From_SDOH_Differences'
        ],
        'Value': [
            len(deserts_df),
            len(oases_df),
            ypll_desert_actual,
            ypll_oasis_actual,
            observed_mortality_gap,
            predicted_ypll,
            ypll_reduction_from_technology,
            pct_gap_explained_by_technology,
            100 - pct_gap_explained_by_technology
        ]
    })
    simulation_results.to_csv(CSV_DIR / "what_if_simulation_results.csv", index=False)
    logging.info(f"Exported what-if simulation results to {CSV_DIR / 'what_if_simulation_results.csv'}")
    
    # Return deserts and oases dataframes for empirical IV decomposition
    return deserts_df, oases_df

def calculate_national_impact_estimate(model, df_unscaled, scaler):
    """
    Estimates the total national reduction in YPLL and equivalent lives saved if all
    non-oasis counties were given best-in-class AI technology access.
    """
    logging.info("\n" + "="*20 + " NATIONAL IMPACT ESTIMATE " + "="*20)

    oasis_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.10)
    oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] <= oasis_threshold]
    non_oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] > oasis_threshold].copy()
    target_ai_access = oases_df['county_avg_dist_to_ai'].mean()

    logging.info(f"Estimating national impact by improving AI access for {len(non_oases_df)} counties.")
    logging.info(f"Target 'Best-in-Class' AI Access (avg. of best 10%): {target_ai_access:.2f} miles.")

    cols_to_scale = scaler.feature_names_in_
    
    non_oases_df_scaled = pd.DataFrame(scaler.transform(non_oases_df[cols_to_scale]), columns=cols_to_scale, index=non_oases_df.index)
    non_oases_df_scaled['census_division'] = non_oases_df['census_division']
    predicted_ypll_before = model.predict(non_oases_df_scaled)

    hypothetical_df = non_oases_df.copy()
    hypothetical_df['county_avg_dist_to_ai'] = target_ai_access
    hypothetical_df_scaled = pd.DataFrame(scaler.transform(hypothetical_df[cols_to_scale]), columns=cols_to_scale, index=hypothetical_df.index)
    hypothetical_df_scaled['census_division'] = hypothetical_df['census_division']
    predicted_ypll_after = model.predict(hypothetical_df_scaled)
    
    ypll_reduction_per_county = predicted_ypll_before - predicted_ypll_after
    total_national_ypll_reduction = np.sum(ypll_reduction_per_county)

    YPLL_PER_DEATH = 25
    equivalent_lives_saved = total_national_ypll_reduction / YPLL_PER_DEATH
    
    logging.info("\n[NATIONAL IMPACT ESTIMATE] Results:")
    logging.info(f"  - Total potential reduction in YPLL across all improved counties: {total_national_ypll_reduction:,.0f}")
    logging.info(f"  - Assuming {YPLL_PER_DEATH} years of life lost per preventable death, this is equivalent to approximately {equivalent_lives_saved:,.0f} fewer deaths nationally.")
    logging.info("="*56)
    
    # Export national impact estimate
    national_impact = pd.DataFrame({
        'Metric': [
            'Counties_Improved',
            'Target_AI_Access_Miles',
            'Total_YPLL_Reduction',
            'YPLL_Per_Death_Assumption',
            'Equivalent_Lives_Saved'
        ],
        'Value': [
            len(non_oases_df),
            target_ai_access,
            total_national_ypll_reduction,
            YPLL_PER_DEATH,
            equivalent_lives_saved
        ]
    })
    national_impact.to_csv(CSV_DIR / "national_impact_estimate.csv", index=False)
    logging.info(f"Exported national impact estimate to {CSV_DIR / 'national_impact_estimate.csv'}")

def calculate_empirical_iv_proportions(model, deserts_df, oases_df, scaler):
    """
    Calculates empirical IV proportions using counterfactual decomposition.
    
    For each CHR determinant (IV2, IV3, IV4), we estimate how much of the SDOH
    contribution to the mortality gap would be eliminated if that determinant
    alone improved from desert to oasis levels (holding others constant).
    
    This is a sequential decomposition:
    1. IV2 (Physical Environment): Contribution when improved alone
    2. IV3 (Health Behaviors): Additional contribution when improved
    3. IV4 (Socioeconomic Factors): Remaining contribution
    4. IV1 (Public Health Policy): Residual (since not directly in model)
    
    Args:
        model: Fitted OLS model with standardized predictors
        deserts_df: DataFrame of desert counties (unscaled)
        oases_df: DataFrame of oasis counties (unscaled)
        scaler: StandardScaler used to standardize predictors
    
    Returns:
        dict: {'iv1': float, 'iv2': float, 'iv3': float, 'iv4': float}
              where values sum to 1.0 and represent proportions of SDOH contribution
    """
    logging.info("\n" + "="*20 + " EMPIRICAL IV DECOMPOSITION " + "="*20)
    
    # Get model predictors
    model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
    
    # Create baseline desert profile (typical desert county)
    baseline_profile = deserts_df[model_predictors].mean().to_frame().T
    
    # Scale the baseline profile
    cols_to_scale = scaler.feature_names_in_
    baseline_scaled = scaler.transform(baseline_profile[cols_to_scale])
    baseline_df = pd.DataFrame(baseline_scaled, columns=cols_to_scale)
    baseline_df['census_division'] = deserts_df['census_division'].mode()[0]
    
    # Predict baseline YPLL (desert conditions)
    ypll_baseline = model.predict(baseline_df)[0]
    
    # Get oasis-level values for each determinant
    oasis_means = oases_df[model_predictors].mean()
    
    # Sequential decomposition: improve one determinant at a time
    # Profile 1: Improve only IV2 (Physical Environment)
    profile_iv2 = baseline_profile.copy()
    profile_iv2['physical_environment_score'] = oasis_means['physical_environment_score']
    profile_iv2_scaled = scaler.transform(profile_iv2[cols_to_scale])
    profile_iv2_df = pd.DataFrame(profile_iv2_scaled, columns=cols_to_scale)
    profile_iv2_df['census_division'] = deserts_df['census_division'].mode()[0]
    ypll_after_iv2 = model.predict(profile_iv2_df)[0]
    iv2_contribution = ypll_baseline - ypll_after_iv2
    
    # Profile 2: Improve IV2 + IV3 (Health Behaviors)
    profile_iv2_iv3 = profile_iv2.copy()
    profile_iv2_iv3['health_behaviors_score'] = oasis_means['health_behaviors_score']
    profile_iv2_iv3_scaled = scaler.transform(profile_iv2_iv3[cols_to_scale])
    profile_iv2_iv3_df = pd.DataFrame(profile_iv2_iv3_scaled, columns=cols_to_scale)
    profile_iv2_iv3_df['census_division'] = deserts_df['census_division'].mode()[0]
    ypll_after_iv2_iv3 = model.predict(profile_iv2_iv3_df)[0]
    iv3_contribution = ypll_after_iv2 - ypll_after_iv2_iv3
    
    # Profile 3: Improve IV2 + IV3 + IV4 (Socioeconomic)
    profile_iv2_iv3_iv4 = profile_iv2_iv3.copy()
    profile_iv2_iv3_iv4['social_economic_factors_score'] = oasis_means['social_economic_factors_score']
    profile_iv2_iv3_iv4_scaled = scaler.transform(profile_iv2_iv3_iv4[cols_to_scale])
    profile_iv2_iv3_iv4_df = pd.DataFrame(profile_iv2_iv3_iv4_scaled, columns=cols_to_scale)
    profile_iv2_iv3_iv4_df['census_division'] = deserts_df['census_division'].mode()[0]
    ypll_after_iv2_iv3_iv4 = model.predict(profile_iv2_iv3_iv4_df)[0]
    iv4_contribution = ypll_after_iv2_iv3 - ypll_after_iv2_iv3_iv4
    
    # Total SDOH contribution (sum of IV contributions)
    total_iv_contribution = iv2_contribution + iv3_contribution + iv4_contribution
    
    # Check for negative or problematic contributions - use absolute value approach
    # This can happen with suppressor variables or multicollinearity
    if iv2_contribution < 0 or iv3_contribution < 0 or total_iv_contribution <= 0:
        logging.warning("Negative contributions detected - using standardized coefficient approach instead")
        
        # Alternative: Use standardized coefficients directly
        coef_iv2 = abs(model.params.get('physical_environment_score', 0))
        coef_iv3 = abs(model.params.get('health_behaviors_score', 0))
        coef_iv4 = abs(model.params.get('social_economic_factors_score', 0))
        
        total_coef = coef_iv2 + coef_iv3 + coef_iv4
        
        if total_coef > 0:
            iv2_prop_raw = coef_iv2 / total_coef
            iv3_prop_raw = coef_iv3 / total_coef
            iv4_prop_raw = coef_iv4 / total_coef
        else:
            # Fallback to equal weights
            iv2_prop_raw = 0.333
            iv3_prop_raw = 0.333
            iv4_prop_raw = 0.334
        
        # Allocate 5% to IV1, adjust others proportionally
        iv1_prop = 0.05
        adjustment = 0.95
        iv2_prop = iv2_prop_raw * adjustment
        iv3_prop = iv3_prop_raw * adjustment
        iv4_prop = iv4_prop_raw * adjustment
        
    else:
        # IV1 (Public Health Policy/Medicaid) is residual - not directly in model
        # Allocate a small proportion as unmeasured/policy factors
        iv1_contribution = total_iv_contribution * 0.05  # 5% residual for policy/unmeasured
        
        # Adjust other IVs to account for IV1 residual
        adjustment_factor = 0.95  # Remaining 95% goes to measured IVs
        iv2_contribution *= adjustment_factor
        iv3_contribution *= adjustment_factor
        iv4_contribution *= adjustment_factor
        
        # Recalculate total
        total_iv_contribution = iv1_contribution + iv2_contribution + iv3_contribution + iv4_contribution
        
        # Convert to proportions
        iv1_prop = iv1_contribution / total_iv_contribution if total_iv_contribution > 0 else 0.05
        iv2_prop = iv2_contribution / total_iv_contribution if total_iv_contribution > 0 else 0.15
        iv3_prop = iv3_contribution / total_iv_contribution if total_iv_contribution > 0 else 0.32
        iv4_prop = iv4_contribution / total_iv_contribution if total_iv_contribution > 0 else 0.48
    
    # Ensure they sum to 1.0 (handle rounding)
    total_prop = iv1_prop + iv2_prop + iv3_prop + iv4_prop
    if total_prop > 0:
        iv1_prop /= total_prop
        iv2_prop /= total_prop
        iv3_prop /= total_prop
        iv4_prop /= total_prop
    
    logging.info(f"\nEmpirical IV Decomposition Results:")
    logging.info(f"  Baseline YPLL (Desert):           {ypll_baseline:,.0f}")
    logging.info(f"  After IV2 Improvement:            {ypll_after_iv2:,.0f} (Δ = {iv2_contribution:,.0f})")
    logging.info(f"  After IV2+IV3 Improvement:        {ypll_after_iv2_iv3:,.0f} (Δ = {iv3_contribution:,.0f})")
    logging.info(f"  After IV2+IV3+IV4 Improvement:    {ypll_after_iv2_iv3_iv4:,.0f} (Δ = {iv4_contribution:,.0f})")
    logging.info(f"\nProportional Contributions (of SDOH):")
    logging.info(f"  IV1 (Public Health Policy):       {iv1_prop*100:.1f}% (residual)")
    logging.info(f"  IV2 (Physical Environment):       {iv2_prop*100:.1f}%")
    logging.info(f"  IV3 (Health Behaviors):           {iv3_prop*100:.1f}%")
    logging.info(f"  IV4 (Socioeconomic Factors):      {iv4_prop*100:.1f}%")
    logging.info(f"  Total:                            {(iv1_prop+iv2_prop+iv3_prop+iv4_prop)*100:.1f}%")
    logging.info("="*52)
    
    return {
        'iv1': iv1_prop,
        'iv2': iv2_prop,
        'iv3': iv3_prop,
        'iv4': iv4_prop
    }

def calculate_shapley_iv_proportions(model, deserts_df, oases_df, scaler):
    """
    Calculates order-invariant IV proportions using Shapley value decomposition.
    
    Shapley values compute the average marginal contribution of each variable
    across all possible orderings (all 3! = 6 permutations for 3 variables),
    providing fair attribution that doesn't depend on the sequence in which
    variables are improved.
    
    Args:
        model: Fitted OLS model with standardized predictors
        deserts_df: DataFrame of desert counties (unscaled)
        oases_df: DataFrame of oasis counties (unscaled)
        scaler: StandardScaler used to standardize predictors
    
    Returns:
        dict: {'iv1': float, 'iv2': float, 'iv3': float, 'iv4': float}
    """
    import math
    from itertools import permutations
    
    logging.info("\n" + "="*20 + " SHAPLEY IV DECOMPOSITION " + "="*20)
    
    # Get model predictors and setup
    model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
    baseline_profile = deserts_df[model_predictors].mean().to_frame().T
    oasis_means = oases_df[model_predictors].mean()
    cols_to_scale = scaler.feature_names_in_
    census_div = deserts_df['census_division'].mode()[0]
    
    # Define the three measured IVs
    iv_vars = {
        'iv2': 'physical_environment_score',
        'iv3': 'health_behaviors_score',
        'iv4': 'social_economic_factors_score'
    }
    
    def predict(profile):
        """Helper to predict YPLL from unscaled profile"""
        scaled = scaler.transform(profile[cols_to_scale])
        df = pd.DataFrame(scaled, columns=cols_to_scale)
        df['census_division'] = census_div
        return float(model.predict(df)[0])
    
    # Compute Shapley values via full permutation averaging
    # For each permutation, track cumulative marginal contribution of each variable
    contrib = {k: 0.0 for k in iv_vars.keys()}
    perms = list(permutations(iv_vars.keys()))
    
    for perm in perms:
        prof = baseline_profile.copy()
        y_prev = predict(prof)
        for k in perm:
            # Improve variable k from desert to oasis level
            prof[iv_vars[k]] = oasis_means[iv_vars[k]]
            y_new = predict(prof)
            # Marginal contribution (reduction in YPLL)
            contrib[k] += (y_prev - y_new)
            y_prev = y_new
    
    # Average over all permutations to get Shapley values
    nfact = math.factorial(len(iv_vars))
    shapley_values = {k: v / nfact for k, v in contrib.items()}
    
    # Convert to proportions using max(0, v) to handle negative contributions
    # (negative means variable is harmful or suppressor; exclude from proportion)
    positive_shapley = {k: max(0.0, shapley_values[k]) for k in shapley_values}
    total_positive = sum(positive_shapley.values())
    
    if total_positive <= 0:
        # Fallback to equal weights if all negative
        logging.warning("All Shapley values negative or zero - using equal weights")
        positive_shapley = {k: 1.0 for k in positive_shapley}
        total_positive = 3.0
    
    # Allocate 5% to IV1 (policy residual), scale others to 95%
    iv1_prop = 0.05
    scale = 0.95
    iv2_prop = scale * (positive_shapley['iv2'] / total_positive)
    iv3_prop = scale * (positive_shapley['iv3'] / total_positive)
    iv4_prop = scale * (positive_shapley['iv4'] / total_positive)
    
    # Normalize to ensure sum = 1.0
    s = iv1_prop + iv2_prop + iv3_prop + iv4_prop
    iv1_prop /= s
    iv2_prop /= s
    iv3_prop /= s
    iv4_prop /= s
    
    logging.info(f"\nShapley Value Decomposition Results (via {nfact} permutations):")
    logging.info(f"  Raw Shapley Values (YPLL reduction, signed):")
    logging.info(f"    IV2 (Physical Environment):     {shapley_values['iv2']:,.2f}")
    logging.info(f"    IV3 (Health Behaviors):         {shapley_values['iv3']:,.2f}")
    logging.info(f"    IV4 (Socioeconomic Factors):    {shapley_values['iv4']:,.2f}")
    if any(v < 0 for v in shapley_values.values()):
        logging.info(f"  Note: Negative values excluded from proportion calculation (suppressor/harmful effects)")
    logging.info(f"\nProportional Contributions (nonnegative, of SDOH):")
    logging.info(f"  IV1 (Public Health Policy):       {iv1_prop*100:.1f}% (residual)")
    logging.info(f"  IV2 (Physical Environment):       {iv2_prop*100:.1f}%")
    logging.info(f"  IV3 (Health Behaviors):           {iv3_prop*100:.1f}%")
    logging.info(f"  IV4 (Socioeconomic Factors):      {iv4_prop*100:.1f}%")
    logging.info(f"  Total:                            {(iv1_prop+iv2_prop+iv3_prop+iv4_prop)*100:.1f}%")
    logging.info("="*52)
    
    # Export signed Shapley values for transparency
    shapley_export = pd.DataFrame({
        'variable': ['iv2', 'iv3', 'iv4'],
        'shapley_value_signed': [shapley_values['iv2'], shapley_values['iv3'], shapley_values['iv4']],
        'proportion_nonnegative': [iv2_prop * 0.95 / (1-0.05), iv3_prop * 0.95 / (1-0.05), iv4_prop * 0.95 / (1-0.05)]
    })
    shapley_export.to_csv(CSV_DIR / "shapley_values_signed.csv", index=False)
    logging.info(f"Exported signed Shapley values to {CSV_DIR / 'shapley_values_signed.csv'}")
    
    return {
        'iv1': iv1_prop,
        'iv2': iv2_prop,
        'iv3': iv3_prop,
        'iv4': iv4_prop,
        'shapley_signed': shapley_values  # Include signed values
    }

def bootstrap_counterfactual_decomposition(model, df_unscaled, scaler, n_bootstrap=1000, 
                                          desert_quantile=0.9, oasis_quantile=0.1,
                                          use_shapley=False, verbose=False):
    """
    Bootstrap confidence intervals for technology share and IV proportions.
    
    Args:
        model: Fitted OLS model
        df_unscaled: Full dataframe of counties (unscaled)
        scaler: StandardScaler
        n_bootstrap: Number of bootstrap iterations
        desert_quantile: Quantile for desert definition (default 0.9 = P90)
        oasis_quantile: Quantile for oasis definition (default 0.1 = P10)
        use_shapley: If True, use Shapley decomposition; otherwise sequential
        verbose: If True, log detailed progress; if False, only summary
    
    Returns:
        dict with 'tech_share', 'iv1', 'iv2', 'iv3', 'iv4' bootstrap distributions
    """
    if verbose:
        logging.info(f"\n" + "="*20 + f" BOOTSTRAP ANALYSIS (n={n_bootstrap}) " + "="*20)
    else:
        logging.info(f"\n" + "="*20 + f" BOOTSTRAP ANALYSIS (n={n_bootstrap}, verbose=False) " + "="*20)
    
    bootstrap_results = {
        'tech_share': [],
        'iv1': [],
        'iv2': [],
        'iv3': [],
        'iv4': []
    }
    
    n_counties = len(df_unscaled)
    
    for i in range(n_bootstrap):
        if verbose and i % 100 == 0:
            logging.info(f"  Bootstrap iteration {i}/{n_bootstrap}")
        
        # Resample counties with replacement
        boot_indices = np.random.choice(n_counties, size=n_counties, replace=True)
        boot_df = df_unscaled.iloc[boot_indices].reset_index(drop=True)
        
        # Identify desert/oasis in bootstrap sample
        desert_threshold = boot_df['county_avg_dist_to_ai'].quantile(desert_quantile)
        oasis_threshold = boot_df['county_avg_dist_to_ai'].quantile(oasis_quantile)
        deserts_boot = boot_df[boot_df['county_avg_dist_to_ai'] >= desert_threshold]
        oases_boot = boot_df[boot_df['county_avg_dist_to_ai'] <= oasis_threshold]
        
        if len(deserts_boot) < 5 or len(oases_boot) < 5:
            continue  # Skip if not enough counties
        
        # Calculate technology share
        ypll_desert = deserts_boot['ypll'].mean()
        ypll_oasis = oases_boot['ypll'].mean()
        gap = ypll_desert - ypll_oasis
        
        if gap <= 0:
            continue  # Skip invalid gaps
        
        # Counterfactual prediction
        model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
        baseline_profile = deserts_boot[model_predictors].mean().to_frame().T
        baseline_profile['county_avg_dist_to_ai'] = oases_boot['county_avg_dist_to_ai'].mean()
        
        cols_to_scale = scaler.feature_names_in_
        baseline_scaled = scaler.transform(baseline_profile[cols_to_scale])
        baseline_df = pd.DataFrame(baseline_scaled, columns=cols_to_scale)
        baseline_df['census_division'] = deserts_boot['census_division'].mode()[0] if len(deserts_boot['census_division'].mode()) > 0 else boot_df['census_division'].mode()[0]
        
        predicted_ypll = model.predict(baseline_df)[0]
        tech_contribution = ypll_desert - predicted_ypll
        tech_share = tech_contribution / gap if gap > 0 else 0
        
        bootstrap_results['tech_share'].append(tech_share)
        
        # Calculate IV proportions
        try:
            # Suppress logging during bootstrap iterations unless verbose
            if not verbose:
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
            
            if use_shapley:
                iv_props = calculate_shapley_iv_proportions(model, deserts_boot, oases_boot, scaler)
            else:
                iv_props = calculate_empirical_iv_proportions(model, deserts_boot, oases_boot, scaler)
            
            if not verbose:
                logging.getLogger().setLevel(original_level)
            
            bootstrap_results['iv1'].append(iv_props['iv1'])
            bootstrap_results['iv2'].append(iv_props['iv2'])
            bootstrap_results['iv3'].append(iv_props['iv3'])
            bootstrap_results['iv4'].append(iv_props['iv4'])
        except:
            # If calculation fails, skip this bootstrap iteration
            bootstrap_results['tech_share'].pop()
            if not verbose:
                logging.getLogger().setLevel(original_level)
            continue
    
    # Calculate confidence intervals
    ci_results = {}
    for key in bootstrap_results.keys():
        values = np.array(bootstrap_results[key])
        if len(values) > 0:
            ci_results[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'distribution': values
            }
        else:
            ci_results[key] = {
                'mean': np.nan,
                'median': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'distribution': np.array([])
            }
    
    logging.info(f"\nBootstrap Results (n={len(bootstrap_results['tech_share'])} successful iterations):")
    logging.info(f"  Technology Share:  {ci_results['tech_share']['mean']*100:.1f}% (95% CI: {ci_results['tech_share']['ci_lower']*100:.1f}%–{ci_results['tech_share']['ci_upper']*100:.1f}%)")
    logging.info(f"  IV1 (Policy):      {ci_results['iv1']['mean']*100:.1f}% (95% CI: {ci_results['iv1']['ci_lower']*100:.1f}%–{ci_results['iv1']['ci_upper']*100:.1f}%)")
    logging.info(f"  IV2 (Environment): {ci_results['iv2']['mean']*100:.1f}% (95% CI: {ci_results['iv2']['ci_lower']*100:.1f}%–{ci_results['iv2']['ci_upper']*100:.1f}%)")
    logging.info(f"  IV3 (Behaviors):   {ci_results['iv3']['mean']*100:.1f}% (95% CI: {ci_results['iv3']['ci_lower']*100:.1f}%–{ci_results['iv3']['ci_upper']*100:.1f}%)")
    logging.info(f"  IV4 (Socioeconomic): {ci_results['iv4']['mean']*100:.1f}% (95% CI: {ci_results['iv4']['ci_lower']*100:.1f}%–{ci_results['iv4']['ci_upper']*100:.1f}%)")
    logging.info("="*52)
    
    return ci_results

def sensitivity_analysis_desert_oasis_thresholds(model, df_unscaled, scaler, use_shapley=False):
    """
    Test sensitivity of decomposition to desert/oasis definitions.
    
    Args:
        model: Fitted OLS model
        df_unscaled: Full dataframe of counties
        scaler: StandardScaler
        use_shapley: If True, use Shapley decomposition
    
    Returns:
        DataFrame with results for different threshold combinations
    """
    logging.info("\n" + "="*20 + " SENSITIVITY: DESERT/OASIS THRESHOLDS " + "="*20)
    
    threshold_combinations = [
        (0.95, 0.05),  # Most extreme
        (0.90, 0.10),  # Default
        (0.80, 0.20),  # More inclusive
    ]
    
    results = []
    
    for desert_q, oasis_q in threshold_combinations:
        desert_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(desert_q)
        oasis_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(oasis_q)
        deserts_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] >= desert_threshold]
        oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] <= oasis_threshold]
        
        ypll_desert = deserts_df['ypll'].mean()
        ypll_oasis = oases_df['ypll'].mean()
        gap = ypll_desert - ypll_oasis
        
        # Counterfactual prediction
        model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
        baseline_profile = deserts_df[model_predictors].mean().to_frame().T
        baseline_profile['county_avg_dist_to_ai'] = oases_df['county_avg_dist_to_ai'].mean()
        
        cols_to_scale = scaler.feature_names_in_
        baseline_scaled = scaler.transform(baseline_profile[cols_to_scale])
        baseline_df = pd.DataFrame(baseline_scaled, columns=cols_to_scale)
        baseline_df['census_division'] = deserts_df['census_division'].mode()[0]
        
        predicted_ypll = model.predict(baseline_df)[0]
        tech_contribution = ypll_desert - predicted_ypll
        tech_share = (tech_contribution / gap * 100) if gap > 0 else 0
        
        # IV decomposition
        if use_shapley:
            iv_props = calculate_shapley_iv_proportions(model, deserts_df, oases_df, scaler)
        else:
            iv_props = calculate_empirical_iv_proportions(model, deserts_df, oases_df, scaler)
        
        results.append({
            'desert_quantile': f"P{int(desert_q*100)}",
            'oasis_quantile': f"P{int(oasis_q*100)}",
            'n_desert': len(deserts_df),
            'n_oasis': len(oases_df),
            'ypll_desert': ypll_desert,
            'ypll_oasis': ypll_oasis,
            'gap': gap,
            'tech_share_pct': tech_share,
            'sdoh_share_pct': 100 - tech_share,
            'iv1_pct': iv_props['iv1'] * 100,
            'iv2_pct': iv_props['iv2'] * 100,
            'iv3_pct': iv_props['iv3'] * 100,
            'iv4_pct': iv_props['iv4'] * 100
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_DIR / "sensitivity_desert_oasis_thresholds.csv", index=False)
    
    logging.info("\nSensitivity Results:")
    for _, row in results_df.iterrows():
        logging.info(f"  {row['desert_quantile']} vs {row['oasis_quantile']}: Tech={row['tech_share_pct']:.1f}%, "
                    f"IV4={row['iv4_pct']:.1f}%, IV3={row['iv3_pct']:.1f}%, IV2={row['iv2_pct']:.1f}%, IV1={row['iv1_pct']:.1f}%")
    logging.info("="*52)
    
    return results_df

def create_attribution_sankey_diagram(simulation_csv_path, output_dir, iv_proportions=None, use_empirical=True):
    """
    Creates a publication-worthy Sankey diagram showing the attribution of 
    preventable mortality reduction between Technology and Structural (SDOH) factors,
    with SDOH broken down into IV1-IV4 (CHR determinants).
    
    Args:
        simulation_csv_path: Path to what_if_simulation_results.csv
        output_dir: Directory to save the Sankey diagram
        iv_proportions: Dict with keys 'iv1', 'iv2', 'iv3', 'iv4' containing
                       empirical proportions (should sum to 1.0). If None, uses
                       illustrative defaults.
        use_empirical: If True and iv_proportions provided, labels as "Empirical".
                      If False, labels as "Illustrative allocation".
    
    Returns:
        Dictionary with breakdown proportions for transparency
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logging.warning("plotly not installed. Skipping Sankey diagram. Install with: pip install plotly")
        return
    
    logging.info("\n" + "="*60)
    logging.info("CREATING ATTRIBUTION SANKEY DIAGRAM")
    logging.info("="*60)
    
    # Load simulation results
    results = pd.read_csv(simulation_csv_path)
    results_dict = dict(zip(results['Metric'], results['Value']))
    
    # Extract key values
    mortality_gap = results_dict['Observed_Mortality_Gap']
    tech_contribution = results_dict['YPLL_Reduction_From_Technology']
    sdoh_contribution = results_dict['Percent_Gap_From_SDOH_Differences'] / 100 * mortality_gap
    tech_pct = results_dict['Percent_Gap_Explained_By_Technology']
    sdoh_pct = results_dict['Percent_Gap_From_SDOH_Differences']
    
    # Extract desert vs oasis comparison
    desert_ypll = results_dict['Avg_YPLL_Desert_Actual']
    oasis_ypll = results_dict['Avg_YPLL_Oasis_Actual']
    pct_difference = ((desert_ypll - oasis_ypll) / oasis_ypll) * 100
    
    logging.info(f"Total Mortality Gap: {mortality_gap:,.0f} YPLL")
    logging.info(f"  Desert Counties: {desert_ypll:,.0f} YPLL")
    logging.info(f"  Oasis Counties: {oasis_ypll:,.0f} YPLL")
    logging.info(f"  Percentage Difference: {pct_difference:.1f}% higher in deserts")
    logging.info(f"  Technology Attribution: {tech_contribution:,.0f} YPLL ({tech_pct:.1f}%)")
    logging.info(f"  Structural (SDOH) Attribution: {sdoh_contribution:,.0f} YPLL ({sdoh_pct:.1f}%)")
    
    # Determine IV breakdown: empirical or illustrative
    if iv_proportions is not None and use_empirical:
        # Use provided empirical proportions from regression model
        iv1_pct = iv_proportions.get('iv1', 0.03)
        iv2_pct = iv_proportions.get('iv2', 0.15)
        iv3_pct = iv_proportions.get('iv3', 0.32)
        iv4_pct = iv_proportions.get('iv4', 0.50)
        breakdown_label = "Shapley (model-implied)"
        logging.info(f"\n  Using SHAPLEY (MODEL-IMPLIED) IV breakdown from regression:")
    else:
        # Use illustrative defaults based on CHR framework
        iv1_pct = 0.03   # ~3% of SDOH (policy has indirect effects)
        iv2_pct = 0.15   # ~15% of SDOH (air quality, housing, food access)
        iv3_pct = 0.32   # ~32% of SDOH (behaviors are major driver)
        iv4_pct = 0.50   # ~50% of SDOH (SES is strongest predictor)
        breakdown_label = "Illustrative (CHR framework)"
        logging.info(f"\n  Using ILLUSTRATIVE IV breakdown (for demonstration):")
        logging.info(f"  NOTE: These proportions are based on typical CHR framework allocations,")
        logging.info(f"        not derived from your specific regression model.")
    
    logging.info(f"    IV1 (Public Health Policy): {iv1_pct*100:.1f}% of SDOH")
    logging.info(f"    IV2 (Physical Environment): {iv2_pct*100:.1f}% of SDOH")
    logging.info(f"    IV3 (Health Behaviors): {iv3_pct*100:.1f}% of SDOH")
    logging.info(f"    IV4 (Socioeconomic Factors): {iv4_pct*100:.1f}% of SDOH")
    
    # Break down SDOH into CHR determinants (IV1-IV4)
    iv1_contrib = sdoh_contribution * iv1_pct
    iv2_contrib = sdoh_contribution * iv2_pct
    iv3_contrib = sdoh_contribution * iv3_pct
    iv4_contrib = sdoh_contribution * iv4_pct
    
    # Define detailed Sankey structure
    # Nodes: 0=Mortality Gap, 1=Technology, 2=SDOH, 3=IV1, 4=IV2, 5=IV3, 6=IV4
    
    # Add breakdown source label to subtitle
    subtitle_text = f"AI Access vs. County Health Rankings (CHR) Determinants<br>" + \
                   f"<span style='font-size:10px; font-style:italic'>CHR breakdown: {breakdown_label}</span>"
    
    labels = [
        f"Mortality Gap: {mortality_gap:,.0f} YPLL<br>" +
        f"<span style='font-size:11px'>Desert: {desert_ypll:,.0f} | Oasis: {oasis_ypll:,.0f}<br>" +
        f"({pct_difference:.1f}% higher in deserts)</span>",
        f"AI Access<br>({tech_pct:.1f}%)",
        f"SDOH<br>({sdoh_pct:.1f}%)",
        f"IV1: Public Health<br>Policy ({iv1_pct*sdoh_pct:.1f}%)",
        f"IV2: Physical<br>Environment ({iv2_pct*sdoh_pct:.1f}%)",
        f"IV3: Health<br>Behaviors ({iv3_pct*sdoh_pct:.1f}%)",
        f"IV4: Socioeconomic<br>Factors ({iv4_pct*sdoh_pct:.1f}%)"
    ]
    
    # Define flows
    source = [0, 0, 2, 2, 2, 2]  # Mortality Gap → Tech/SDOH, then SDOH → IV1-IV4
    target = [1, 2, 3, 4, 5, 6]  # To Technology, SDOH, and IV breakdowns
    values = [tech_contribution, sdoh_contribution, iv1_contrib, iv2_contrib, iv3_contrib, iv4_contrib]
    
    # Color scheme (professional palette matching CHR framework)
    node_colors = [
        '#8B4513',  # Mortality Gap - brown (serious issue)
        '#3498db',  # Technology - blue (innovation/actionable)
        '#e74c3c',  # SDOH - red (structural challenges)
        '#95a5a6',  # IV1: Policy - gray (indirect/small effect)
        '#1abc9c',  # IV2: Environment - teal (physical factors)
        '#f39c12',  # IV3: Behaviors - orange (modifiable behaviors)
        '#9b59b6'   # IV4: Socioeconomic - purple (deep structural)
    ]
    
    link_colors = [
        'rgba(52, 152, 219, 0.4)',   # Technology flow - blue
        'rgba(231, 76, 60, 0.4)',    # SDOH flow - red
        'rgba(149, 165, 166, 0.4)',  # IV1 - gray
        'rgba(26, 188, 156, 0.4)',   # IV2 - teal
        'rgba(243, 156, 18, 0.4)',   # IV3 - orange
        'rgba(155, 89, 182, 0.4)'    # IV4 - purple
    ]
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=20,
            line=dict(color='white', width=2),
            label=labels,
            color=node_colors,
            customdata=[
                f"{mortality_gap:,.0f} YPLL",
                f"{tech_contribution:,.0f} YPLL",
                f"{sdoh_contribution:,.0f} YPLL",
                f"{iv1_contrib:,.0f} YPLL",
                f"{iv2_contrib:,.0f} YPLL",
                f"{iv3_contrib:,.0f} YPLL",
                f"{iv4_contrib:,.0f} YPLL"
            ],
            hovertemplate='%{label}<br>%{customdata}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors,
            customdata=[
                f"AI Access: {tech_contribution:,.0f} YPLL ({tech_pct:.1f}%)",
                f"SDOH Total: {sdoh_contribution:,.0f} YPLL ({sdoh_pct:.1f}%)",
                f"IV1 Policy: {iv1_contrib:,.0f} YPLL ({iv1_pct*sdoh_pct:.1f}%)",
                f"IV2 Environment: {iv2_contrib:,.0f} YPLL ({iv2_pct*sdoh_pct:.1f}%)",
                f"IV3 Behaviors: {iv3_contrib:,.0f} YPLL ({iv3_pct*sdoh_pct:.1f}%)",
                f"IV4 Socioeconomic: {iv4_contrib:,.0f} YPLL ({iv4_pct*sdoh_pct:.1f}%)"
            ],
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    # Update layout for publication quality
    fig.update_layout(
        title=dict(
            text=f"<b>Attribution of Preventable Mortality Gap</b><br>" +
                 f"<sub>{subtitle_text}</sub>",
            font=dict(size=22, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=13, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
        width=1300,
        margin=dict(l=20, r=20, t=120, b=140)
    )
    
    # Add context annotation below diagram (above key findings)
    fig.add_annotation(
        text=f"<b>Context:</b> AI Desert counties average {desert_ypll:,.0f} YPLL vs. {oasis_ypll:,.0f} YPLL in Oasis counties " +
             f"({pct_difference:.1f}% higher mortality)",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        xanchor='center', yanchor='bottom',
        showarrow=False,
        font=dict(size=12, color='#2c3e50', family='Arial, sans-serif'),
        align='center',
        bgcolor='rgba(236, 240, 241, 0.8)',
        bordercolor='#95a5a6',
        borderwidth=1,
        borderpad=6
    )
    
    # Add key findings annotation at bottom
    fig.add_annotation(
        text=f"<b>Key Finding:</b> {tech_pct:.1f}% of the {mortality_gap:,.0f} YPLL mortality gap could be addressed through improved AI access.<br>" +
             f"The remaining {sdoh_pct:.1f}% is driven by: Socioeconomic factors ({iv4_pct*sdoh_pct:.1f}%), Health behaviors ({iv3_pct*sdoh_pct:.1f}%), " +
             f"Physical environment ({iv2_pct*sdoh_pct:.1f}%), and Public health policy ({iv1_pct*sdoh_pct:.1f}%)",
        xref="paper", yref="paper",
        x=0.5, y=-0.16,
        xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(size=11, color='#34495e', family='Arial, sans-serif'),
        align='center'
    )
    
    # Save as HTML (interactive)
    html_path = output_dir / "attribution_sankey_diagram.html"
    fig.write_html(str(html_path))
    logging.info(f"✓ Saved interactive Sankey diagram: {html_path}")
    
    # Save as static PNG (for publications)
    try:
        png_path = output_dir / "attribution_sankey_diagram.png"
        fig.write_image(str(png_path), width=1500, height=750, scale=2)
        logging.info(f"✓ Saved static PNG: {png_path}")
    except Exception as e:
        logging.warning(f"Could not save PNG (requires kaleido): {e}")
        logging.info("  Install with: pip install kaleido")
    
    # Save as PDF (for publications)
    try:
        pdf_path = output_dir / "attribution_sankey_diagram.pdf"
        fig.write_image(str(pdf_path), width=1500, height=750)
        logging.info(f"✓ Saved PDF: {pdf_path}")
    except Exception as e:
        logging.warning(f"Could not save PDF (requires kaleido): {e}")
    
    # Create simple version (without IV breakdown)
    fig_simple = create_simple_sankey(mortality_gap, tech_contribution, sdoh_contribution,
                                      tech_pct, sdoh_pct, output_dir)
    
    logging.info("="*60)


def create_simple_sankey(mortality_gap, tech_contribution, sdoh_contribution,
                        tech_pct, sdoh_pct, output_dir):
    """
    Creates a simplified Sankey diagram with just Technology vs. SDOH (no IV breakdown).
    Useful for presentations or simpler narratives.
    """
    import plotly.graph_objects as go
    
    labels = [
        "Mortality Gap<br>(Desert vs. Oasis)",
        f"AI Access<br>({tech_pct:.1f}%)",
        f"Structural Factors<br>(SDOH {sdoh_pct:.1f}%)"
    ]
    
    source = [0, 0]
    target = [1, 2]
    values = [tech_contribution, sdoh_contribution]
    
    node_colors = [
        '#8B4513',  # Mortality Gap
        '#3498db',  # Technology
        '#e74c3c'   # SDOH
    ]
    
    link_colors = [
        'rgba(52, 152, 219, 0.4)',
        'rgba(231, 76, 60, 0.4)'
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=25,
            line=dict(color='white', width=2),
            label=labels,
            color=node_colors,
            customdata=[
                f"{mortality_gap:,.0f} YPLL",
                f"{tech_contribution:,.0f} YPLL",
                f"{sdoh_contribution:,.0f} YPLL"
            ],
            hovertemplate='%{label}<br>%{customdata}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors,
            customdata=[
                f"AI Access: {tech_contribution:,.0f} YPLL ({tech_pct:.1f}%)",
                f"SDOH: {sdoh_contribution:,.0f} YPLL ({sdoh_pct:.1f}%)"
            ],
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Attribution of Preventable Mortality Gap (Simplified)</b><br>" +
                 "<sub>AI Access vs. Social Determinants of Health</sub>",
            font=dict(size=20, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=14, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        width=1000,
        margin=dict(l=20, r=20, t=120, b=80)
    )
    
    fig.add_annotation(
        text=f"<b>Key Finding:</b> {tech_pct:.1f}% of the mortality gap could be addressed through<br>" +
             f"improved AI access, while {sdoh_pct:.1f}% requires structural interventions",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(size=12, color='#34495e'),
        align='center'
    )
    
    html_path = output_dir / "attribution_sankey_simple.html"
    fig.write_html(str(html_path))
    logging.info(f"✓ Saved simple Sankey diagram: {html_path}")
    
    return fig


def create_detailed_attribution_sankey(results_dict, mortality_gap, tech_contribution, 
                                       sdoh_contribution, tech_pct, sdoh_pct):
    """
    LEGACY: Creates a more detailed Sankey with multiple levels showing the flow from
    mortality gap through attribution to potential interventions.
    """
    import plotly.graph_objects as go
    
    # More granular breakdown
    # Assume SDOH can be further split into: Socioeconomic (40%), Behavioral (20%), Environment (13.7%)
    # Based on typical SDOH research proportions
    socioeconomic = sdoh_contribution * 0.543  # ~40% of SDOH
    behavioral = sdoh_contribution * 0.271      # ~20% of SDOH  
    environment = sdoh_contribution * 0.186     # ~13.7% of SDOH
    
    labels = [
        "Mortality Gap",                           # 0
        f"Technology<br>{tech_pct:.1f}%",         # 1
        f"SDOH<br>{sdoh_pct:.1f}%",               # 2
        "AI/Robotics<br>Intervention",            # 3
        "Socioeconomic<br>Policy",                # 4
        "Health<br>Behaviors",                    # 5
        "Physical<br>Environment"                 # 6
    ]
    
    source = [0, 0, 2, 2, 2, 1]  # From Mortality Gap and SDOH
    target = [1, 2, 4, 5, 6, 3]  # To Tech, SDOH breakdown, and interventions
    values = [
        tech_contribution,
        sdoh_contribution,
        socioeconomic,
        behavioral,
        environment,
        tech_contribution
    ]
    
    node_colors = [
        '#8B4513',  # Mortality Gap
        '#3498db',  # Technology
        '#e74c3c',  # SDOH
        '#2ecc71',  # AI Intervention
        '#9b59b6',  # Socioeconomic
        '#f39c12',  # Behavioral
        '#1abc9c'   # Environment
    ]
    
    link_colors = [
        'rgba(52, 152, 219, 0.4)',   # Tech flow
        'rgba(231, 76, 60, 0.4)',    # SDOH flow
        'rgba(155, 89, 182, 0.4)',   # Socioeconomic
        'rgba(243, 156, 18, 0.4)',   # Behavioral
        'rgba(26, 188, 156, 0.4)',   # Environment
        'rgba(46, 204, 113, 0.4)'    # To intervention
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=20,
            line=dict(color='white', width=2),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Detailed Attribution: Preventable Mortality to Intervention Pathways</b>",
            font=dict(size=20, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=12, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1200,
        margin=dict(l=20, r=20, t=100, b=60)
    )
    
    return fig


def generate_summary_visualizations(gdf):
    """Creates general-purpose visualizations for the final report."""
    logging.info("\n" + "="*20 + " GENERATING SUMMARY VISUALIZATIONS " + "="*20)
    sns.set_theme(style="whitegrid")

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping all map visualizations.")
    else:
        try:
            logging.info("Preparing basemap and data for map visualizations...")
            us_basemap = gpd.read_file(COUNTY_FILE)
            # Set CRS if missing (Census shapefiles are typically NAD83)
            if us_basemap.crs is None:
                logging.warning("County shapefile has no CRS defined. Setting to EPSG:4269 (NAD83).")
                us_basemap = us_basemap.set_crs("EPSG:4269")
            
            # Filter to contiguous US if STATEFP column exists
            if 'STATEFP' in us_basemap.columns:
                excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
                us_contig_basemap = us_basemap[~us_basemap['STATEFP'].isin(excluded_fips)]
            else:
                logging.warning("County shapefile missing STATEFP column. Using all geometries for basemap.")
                us_contig_basemap = us_basemap

            excluded_states = ['AK', 'HI', 'PR']
            gdf_contig_points = gdf[~gdf['state'].isin(excluded_states)].copy()

            logging.info("Generating map 1: All hospital tech types (Contiguous US).")
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # Ensure basemap is in AEA projection
            us_contig_basemap = us_contig_basemap.to_crs(AEA_CRS)
            us_contig_basemap.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)
            
            # Ensure points are in AEA projection
            gdf_contig_points = gdf_contig_points.to_crs(AEA_CRS)
            gdf_contig_points.plot(
                ax=ax, column='tech_type', categorical=True, legend=True, markersize=20,
                alpha=0.8, legend_kwds={'title': "Technology Type", 'loc': 'lower left'}
            )
            ax.set_title('Distribution of US Hospitals by AI and Robotics Adoption (Contiguous US)', fontsize=16)
            ax.set_axis_off()
            # Let matplotlib auto-scale based on the data extent in AEA projection
            fig_path = FIG_DIR / "hospital_distribution_contiguous_US.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved contiguous US map to {fig_path}")
            plt.close(fig)

        except Exception as e:
            logging.error(f"Could not generate distribution map(s): {e}", exc_info=True)

    sns.set_theme(style="whitegrid", palette="viridis")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=gdf, x='nearest_miles', bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Distances to Nearest Hospital (Cleaned Data)')
    ax.set_xlabel('Distance to Nearest Hospital (miles)'); ax.set_ylabel('Number of Hospitals')
    ax.axvline(gdf['nearest_miles'].median(), color='red', linestyle='--', label=f"Median: {gdf['nearest_miles'].median():.1f} mi")
    ax.legend()
    fig_path = FIG_DIR / "nearest_distance_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved distance histogram to {fig_path}")
    plt.close(fig)

# ======================= MAIN EXECUTION ==========================
def main():
    """Main function to orchestrate the analysis workflow."""
    setup_environment()
    engine = connect_to_db()

    # --- Load Data ---
    hospital_df = load_hospital_data(engine)
    pop_gdf = load_population_data(POP_FILE)

    # --- Preprocess and Clean ---
    hospital_gdf_preprocessed = preprocess_data(hospital_df)
    hospital_gdf_cleaned = calculate_proximity_metrics(hospital_gdf_preprocessed)

    # --- Run Core Analyses ---
    analyze_population_coverage(hospital_gdf_cleaned, pop_gdf)
    analyze_inequality_with_lorenz(hospital_gdf_cleaned, pop_gdf)
    test_sutva_assumptions(engine, hospital_gdf_cleaned, pop_gdf)
    model, df_for_simulation, scaler, deserts_df, oases_df = analyze_ypll_and_technology_access(engine, hospital_gdf_cleaned, pop_gdf)
    
    # --- NEW: Sensitivity Analysis for Exposure Construction ---
    sensitivity_analysis_exposure_construction(engine, hospital_gdf_cleaned, pop_gdf)
    
    model_adoption_drivers(hospital_gdf_cleaned)
    perform_hotspot_analysis(hospital_gdf_cleaned)

    # --- NEW: Run and Visualize k-NN Graph Analysis (CORRECTED LOGIC) ---
    logging.info("\n--- Preparing data for Contiguous US k-NN Graph ---")
    excluded_states = ['AK', 'HI', 'PR', 'GU', 'AS', 'VI', 'MP']
    hospital_gdf_contig = hospital_gdf_cleaned[~hospital_gdf_cleaned['state'].isin(excluded_states)].copy()
    logging.info(f"Filtered to {len(hospital_gdf_contig)} hospitals in the contiguous US for graph analysis.")

    K_VALUE = 5
    # Build the graph ONLY with contiguous US data to ensure all edges are internal
    knn_edges = create_knn_graph(hospital_gdf_contig, k=K_VALUE)
    
    if knn_edges: # Only plot if the graph was created successfully
        plot_knn_graph(
            gdf=hospital_gdf_contig, # Pass the filtered GDF
            edges=knn_edges,
            k=K_VALUE,
            filename=FIG_DIR / f"knn_graph_k{K_VALUE}_contiguous_us_corrected.png",
            title=f'k-Nearest Neighbor Graph of U.S. Hospitals (k={K_VALUE}, Contiguous US)'
        )

    # --- Existing final steps ---
    generate_summary_visualizations(hospital_gdf_cleaned)

    # --- Save Final Output ---
    try:
        output_path = BASE_DIR / "hospital_ai_robotics_enriched.parquet"
        final_df = pd.DataFrame(hospital_gdf_cleaned.drop(columns='geometry'))
        final_df['geometry_wkt'] = hospital_gdf_cleaned.geometry.to_wkt()
        final_df.to_parquet(output_path, index=False)
        logging.info(f"\nSuccessfully saved enriched data to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")

    # --- Create Diagnostic Summary of Publication Outputs ---
    logging.info("\n" + "="*60)
    logging.info("PUBLICATION OUTPUTS SUMMARY")
    logging.info("="*60)
    
    expected_outputs = [
        "hospital_summary_statistics.csv",
        "proximity_metrics_summary.csv",
        "population_coverage_analysis.csv",
        "county_population_weighted_distances.csv",
        "decile_analysis_all_access.csv",
        "decile_analysis_ai_access.csv",
        "decile_analysis_robo_access.csv",
        "absolute_gap_analysis_p90_p10.csv",
        "gini_coefficients.csv",
        "robustness_checks_inequality.csv",
        "sensitivity_exposure_thresholds_weights.csv",
        "dose_response_continuous_exposure.csv",
        "adoption_drivers_logistic_regression.csv",
        "sutva_violation_tests_summary.csv",
        "ypll_regression_coefficients.csv",
        "ypll_regression_summary.csv",
        "what_if_simulation_results.csv",
        "national_impact_estimate.csv"
    ]
    
    diagnostic_data = []
    for filename in expected_outputs:
        filepath = CSV_DIR / filename
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        diagnostic_data.append({
            'Filename': filename,
            'Created': 'Yes' if exists else 'No',
            'Size_Bytes': size
        })
    
    diagnostic_df = pd.DataFrame(diagnostic_data)
    diagnostic_df.to_csv(CSV_DIR / "_diagnostic_output_summary.csv", index=False)
    
    logging.info(f"\nCreated {diagnostic_df['Created'].value_counts().get('Yes', 0)} out of {len(expected_outputs)} expected output files")
    logging.info(f"Diagnostic summary saved to: {CSV_DIR / '_diagnostic_output_summary.csv'}")
    logging.info(f"All publication outputs are in: {CSV_DIR}")
    
    # --- Robustness Checks for Counterfactual Decomposition ---
    logging.info("\n" + "="*60)
    logging.info("COUNTERFACTUAL DECOMPOSITION ROBUSTNESS CHECKS")
    logging.info("="*60)
    
    try:
        # 1. Shapley decomposition (order-invariant)
        logging.info("\n[1/3] Computing Shapley value decomposition...")
        iv_props_shapley = calculate_shapley_iv_proportions(model, deserts_df, oases_df, scaler)
        
        # 2. Bootstrap confidence intervals
        logging.info("\n[2/3] Bootstrapping confidence intervals (n=1000, this may take a few minutes)...")
        bootstrap_ci = bootstrap_counterfactual_decomposition(
            model, df_for_simulation, scaler, n_bootstrap=1000, 
            desert_quantile=0.9, oasis_quantile=0.1, use_shapley=True
        )
        
        # Save bootstrap results
        bootstrap_summary = pd.DataFrame({
            'metric': ['tech_share', 'iv1', 'iv2', 'iv3', 'iv4'],
            'mean': [bootstrap_ci[k]['mean'] for k in ['tech_share', 'iv1', 'iv2', 'iv3', 'iv4']],
            'median': [bootstrap_ci[k]['median'] for k in ['tech_share', 'iv1', 'iv2', 'iv3', 'iv4']],
            'ci_lower': [bootstrap_ci[k]['ci_lower'] for k in ['tech_share', 'iv1', 'iv2', 'iv3', 'iv4']],
            'ci_upper': [bootstrap_ci[k]['ci_upper'] for k in ['tech_share', 'iv1', 'iv2', 'iv3', 'iv4']]
        })
        bootstrap_summary.to_csv(CSV_DIR / "bootstrap_confidence_intervals.csv", index=False)
        logging.info(f"Saved bootstrap results to {CSV_DIR / 'bootstrap_confidence_intervals.csv'}")
        
        # 3. Sensitivity to desert/oasis thresholds
        logging.info("\n[3/3] Testing sensitivity to desert/oasis definitions...")
        sensitivity_df = sensitivity_analysis_desert_oasis_thresholds(
            model, df_for_simulation, scaler, use_shapley=True
        )
        
        logging.info("\n" + "="*60)
        logging.info("ROBUSTNESS CHECKS COMPLETED")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error in robustness checks: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # --- Create Attribution Sankey Diagram ---
    logging.info("\n" + "="*20 + " CREATING SANKEY DIAGRAM " + "="*20)
    simulation_csv = CSV_DIR / "what_if_simulation_results.csv"
    if simulation_csv.exists():
        # Use Shapley decomposition for final diagram (order-invariant)
        try:
            logging.info("Using Shapley value decomposition for Sankey diagram...")
            iv_props = calculate_shapley_iv_proportions(model, deserts_df, oases_df, scaler)
            logging.info(f"Successfully calculated IV proportions: {iv_props}")
            logging.info("Creating Sankey diagram with Shapley proportions...")
            create_attribution_sankey_diagram(simulation_csv, FIG_DIR, 
                                            iv_proportions=iv_props, use_empirical=True)
            logging.info("Sankey diagram created successfully!")
        except Exception as e:
            logging.error(f"Failed to calculate Shapley IV proportions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            logging.info("Falling back to sequential decomposition...")
            try:
                iv_props = calculate_empirical_iv_proportions(model, deserts_df, oases_df, scaler)
                create_attribution_sankey_diagram(simulation_csv, FIG_DIR, 
                                                iv_proportions=iv_props, use_empirical=True)
            except Exception as e2:
                logging.error(f"Sequential decomposition also failed: {e2}")
                logging.info("Falling back to illustrative IV proportions")
                create_attribution_sankey_diagram(simulation_csv, FIG_DIR, 
                                                iv_proportions=None, use_empirical=False)
    else:
        logging.warning(f"Cannot create Sankey diagram: {simulation_csv} not found")
    
    logging.info("=" * 60)
    logging.info("RUN COMPLETED")
    logging.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An unhandled exception occurred during the main execution.")
        sys.exit(1)
