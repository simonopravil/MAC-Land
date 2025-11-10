"""
Accuracy-Conditional Probability Calculation and Export for LULC Classification

This script calculates accuracy-conditional probabilities for Land Use Land Cover (LULC) 
classifications across multiple datasets for specified Areas of Interest (AOIs) - Alps or 
Carpathians. It processes input datasets, computes class probabilities based on error matrices, 
and exports probability maps in chunks.
"""

# Standard library imports
import sys
import itertools

# Third-party imports
import ee
import geemap
import pandas as pd
import numpy as np


from Wrapper import getData

# ------------------------- Initialization and Configuration -------------------------


# Configuration parameters
VERSION = 'v7'
DATASET_NAMES = ['dw', 'esri', 'esa', 'elc', 'clc', 'glc']
PROB_BANDS = {
    'Alps': ['artificial', 'cropland', 'woodland', 'shrubland',
             'grassland', 'bareland', 'water', 'wetland', 'snowice'],
    'Carpathians': ['artificial', 'cropland', 'woodland', 'shrubland',
                    'grassland', 'bareland', 'water', 'wetland']
}
CHUNK_SIZE = 30000  # Number of rows to process per chunk for export


PROJECT_ID = 'ee-alpscarpathians'
ASSET_ID = f'projects/{PROJECT_ID}/assets/LULC/{AOI_NAME}/Con_AccCon',

def initialize_earth_engine():
    """Initialize Google Earth Engine with high-volume endpoint."""
    try:
        ee.Authenticate()
        ee.Initialize(
            opt_url='https://earthengine-highvolume.googleapis.com',
            project=PROJECT_ID
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Earth Engine: {str(e)}")

# Initialize Earth Engine
initialize_earth_engine()

# ------------------------- Utility Functions -------------------------
def load_spatial_data(aoi_name: str) -> tuple:
    """
    Load spatial data including AOI, training points, and match image.

    Args:
        aoi_name (str): Name of the area of interest ('Alps' or 'Carpathians').

    Returns:
        tuple: AOI, training data, and match image.
    """
    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph')
           .filter(ee.Filter.eq('m_range', aoi_name)))
    training = (ee.FeatureCollection(f'projects/ee-simonopravil/assets/LULC/{aoi_name}/Parametrisation_{aoi_name}_merged')
                .select('class')
                .filter(ee.Filter.neq('class', 0)))
    match_image = ee.Image(f'projects/ee-simonopravil/assets/LULC/{aoi_name}/sameMask')
    return aoi, training, match_image

def calculate_error_matrices(dataset: ee.Image, dataset_names: list, training: ee.FeatureCollection) -> list:
    """
    Calculate normalized error matrices for each dataset.

    Args:
        dataset (ee.Image): Input dataset image with multiple bands.
        dataset_names (list): List of dataset names (bands).
        training (ee.FeatureCollection): Training data for error matrix calculation.

    Returns:
        list: List of normalized error matrices.
    """
    samples = dataset.sampleRegions(training, scale=10, projection='EPSG:3035', tileScale=16)
    error_matrices = []
    for name in dataset_names:
        array_2d = np.array(samples.errorMatrix('class', name).getInfo())
        column_sums = np.sum(array_2d, axis=1)
        normalized_array = array_2d / column_sums
        normalized_array = np.nan_to_num(normalized_array, nan=0)
        error_matrices.append(normalized_array)
    return error_matrices

def calculate_class_probabilities(row: list, error_matrices: list, aoi_name: str) -> dict:
    """
    Calculate class probabilities based on error matrices for a given scenario.

    Args:
        row (list): List of class predictions from different datasets.
        error_matrices (list): List of normalized error matrices.
        aoi_name (str): Name of the area of interest ('Alps' or 'Carpathians').

    Returns:
        dict: Dictionary of class probabilities.
    """
    dataset_indices = list(range(len(DATASET_NAMES)))
    class_probabilities = {}

    for class_label in set(row):
        if aoi_name == 'Carpathians':
            class_label = 0 if class_label > 8 else class_label

        prob_sum = 0
        for dataset_index, class_index in zip(dataset_indices, row):
            if aoi_name == 'Carpathians':
                class_index = 0 if class_index > 8 else class_index

            if class_index == class_label:
                prob_sum += error_matrices[dataset_index][class_index, class_index]
            else:
                prob_sum -= error_matrices[dataset_index][class_index, class_label]

        class_probabilities[class_label] = (1 / len(DATASET_NAMES)) * prob_sum

    return class_probabilities

def concatenate_values(lst: list) -> int:
    """
    Concatenate a list of integers into a single integer.

    Args:
        lst (list): List of integers.

    Returns:
        int: Concatenated integer.
    """
    return int(''.join(map(str, lst)))

# ------------------------- Data Processing Functions -------------------------
def process_probabilities(aoi_name: str) -> tuple:
    """
    Process datasets to compute accuracy-conditional probabilities.

    Args:
        aoi_name (str): Name of the area of interest ('Alps' or 'Carpathians').

    Returns:
        tuple: DataFrame with scenarios and probabilities, composite image, and AOI.
    """
    # Load spatial data and dataset
    aoi, training, match_image = load_spatial_data(aoi_name)
    dataset = Harmonize_data(aoi_name, False).select(DATASET_NAMES)

    # Create a composite image encoding scenarios
    image = dataset.expression(
        'array[0]*100000 + array[1]*10000 + array[2]*1000 + array[3]*100 + array[4]*10 + array[5]',
        {'array': dataset}
    )

    # Compute frequency histogram of scenarios
    reduction = image.reduceRegion(ee.Reducer.frequencyHistogram(), aoi, maxPixels=1e10)
    keys = ee.Dictionary(reduction.get(image.bandNames().get(0))).keys().map(lambda f: ee.Feature(None, {'keys': f}))
    keys = ee.FeatureCollection(keys)

    # Calculate error matrices
    error_matrices = calculate_error_matrices(dataset, DATASET_NAMES, training)

    # Convert scenarios to DataFrame
    df = (ee.data.computeFeatures({'expression': keys, 'fileFormat': 'PANDAS_DATAFRAME'})
          .drop('geo', axis=1)
          .rename(columns={'keys': 'scenarios'}))
    df['scenarios'] = df['scenarios'].apply(lambda x: [int(digit) for digit in str(x)])

    # Calculate class probabilities for each scenario
    df['probability'] = df['scenarios'].apply(lambda row: calculate_class_probabilities(row, error_matrices, aoi_name))
    prob_df = df['probability'].apply(pd.Series)
    probability_df = pd.concat([df, prob_df], axis=1)

    # Add scenario number and clean up DataFrame
    probability_df['scenario_number'] = probability_df['scenarios'].apply(concatenate_values)
    probability_df = probability_df.drop(['scenarios', 'probability'], axis=1).fillna(0)

    # Adjust columns based on AOI
    if aoi_name == 'Alps':
        columns = ['scenario_number', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        probability_df = probability_df[columns]
        probability_df.columns = ['scenario_number', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:  # Carpathians
        columns = ['scenario_number', 0, 1, 2, 3, 4, 5, 6, 7, 8]
        probability_df = probability_df[columns]
        probability_df.columns = ['scenario_number', '0', '1', '2', '3', '4', '5', '6', '7', '8']

    return probability_df, image, aoi

def export_probabilities(probability_df: pd.DataFrame, image: ee.Image, aoi: ee.FeatureCollection,
                         aoi_name: str) -> None:
    """
    Export probability maps in chunks to Earth Engine assets.

    Args:
        probability_df (pd.DataFrame): DataFrame with scenarios and probabilities.
        image (ee.Image): Image with encoded scenarios.
        aoi (ee.FeatureCollection): Area of interest.
        aoi_name (str): Name of the area of interest ('Alps' or 'Carpathians').
    """
    total_rows = len(probability_df)
    for start_row in range(0, total_rows, CHUNK_SIZE):
        end_row = min(start_row + CHUNK_SIZE, total_rows)
        old_values = probability_df.iloc[start_row:end_row]['scenario_number'].tolist()
        prob_values = probability_df.iloc[start_row:end_row].drop(['scenario_number', '0'], axis=1).to_numpy()
        new_values = [ee.Array(i) for i in prob_values.tolist()]

        # Remap scenarios to probabilities
        acc_con = image.remap(old_values, new_values)
        acc_con_probabilities = (acc_con
                                 .arrayFlatten([PROB_BANDS[aoi_name]])
                                 .multiply(10000)
                                 .toUint16())

        # Generate labels
        label = (acc_con
                 .arrayArgmax()
                 .arrayProject([0])
                 .arrayFlatten([['label']])
                 .add(1)
                 .toUint16())

        # Combine probabilities and labels
        res_prob_img = acc_con_probabilities.addBands(label).clip(aoi)

        # Export to Earth Engine asset
        task = ee.batch.Export.image.toAsset(
            image=res_prob_img,
            description=f'Tuanmu_Probability_{start_row}',
            assetId=f'projects/ee-simonopravil/assets/LULC/{aoi_name}/AccCon_Probability_{start_row}',
            region=aoi.geometry(),
            scale=10,
            crs='EPSG:3035',
            maxPixels=1e10
        )
        task.start()
        print(f"Export task started for chunk {start_row} to {end_row} in {aoi_name}")

# ------------------------- Execution Logic -------------------------
for aoi_name in ['Alps', 'Carpathians']:
    print(f"Processing AOI: {aoi_name}")
    probability_df, image, aoi = process_probabilities(aoi_name)
    export_probabilities(probability_df, image, aoi, aoi_name)