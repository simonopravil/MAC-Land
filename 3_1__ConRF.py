"""
Land Use Land Cover (LULC) Classification using Random Forest on Google Earth Engine

This script performs LULC classification using a Random Forest classifier across multiple
datasets for specified Areas of Interest (AOIs) - Alps or Carpathians. It processes input
datasets, trains a classifier, and exports probability and label maps.
"""


import sys
import math


import ee
import geemap
import numpy as np
import pandas as pd


from Wrapper import getData



# Configuration parameters
AOI_NAME = 'Carpathians'  # Options: 'Carpathians', 'Alps'
VERSION = 'v7'
DATASET_NAMES = ['dw', 'esri', 'esa', 'elc', 'clc', 'glc']
PROB_BANDS = {
    'Carpathians': ['artificial', 'cropland', 'woodland', 'shrubland',
                    'grassland', 'bareland', 'water', 'wetland'],
    'Alps': ['artificial', 'cropland', 'woodland', 'shrubland',
             'grassland', 'bareland', 'water', 'wetland', 'snowice']
}
PROJECT_ID = 'ee-alpscarpathians'
ASSET_ID = f'projects/{PROJECT_ID}/assets/LULC/{AOI_NAME}/Con_RF',

# ------------------------- Initialization and Configuration -------------------------
def initialize_earth_engine():
    """Initialize Google Earth Engine with high-volume endpoint."""
    try:
        ee.Authenticate()
        ee.Initialize(
            project=PROJECT_ID
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Earth Engine: {str(e)}")

# Initialize Earth Engine
initialize_earth_engine()

# ------------------------- Data Loading Functions -------------------------
def load_spatial_data():
    """Load spatial data including AOI, grid, training points, and parameterization."""
    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph')
           .filter(ee.Filter.eq('m_range', AOI_NAME)))
    
    training = ee.FeatureCollection(f'projects/ee-simonopravil/assets/LULC/{AOI_NAME}/LUCAS')
    
    parametrisation = (ee.FeatureCollection(f'projects/ee-simonopravil/assets/LULC/{AOI_NAME}/Parametrisation_{AOI_NAME}_merged')
                       .select('class')
                       .filter(ee.Filter.neq('class', 0)))
    
    match_image = ee.Image(f'projects/ee-simonopravil/assets/LULC/{AOI_NAME}/sameMask')
    
    return aoi,  training, parametrisation, match_image

# Load spatial data
aoi, training, parametrisation, match_image = load_spatial_data()

# Load dataset using custom wrapper
dataset = Harmonize_data(AOI_NAME, False)

# ------------------------- Data Processing Functions -------------------------
def prepare_data(image, fscore, aoi_name):
    """
    Prepare data by applying F-scores to different land cover classes.
    
    Args:
        image: ee.Image, input image to process
        fscore: list, F-scores for different classes
        aoi_name: str, area of interest name ('Carpathians' or 'Alps')
    
    Returns:
        ee.Image, processed image with weighted bands
    """
    # Common classes for both AOIs
    bands = {
        'urban': (image.eq(1), fscore[0]),
        'crop': (image.eq(2), fscore[1]),
        'forest': (image.eq(3), fscore[2]),
        'shrubs': (image.eq(4), fscore[3]),
        'grass': (image.eq(5), fscore[4]),
        'bare': (image.eq(6), fscore[5]),
        'water': (image.eq(7), fscore[6]),
        'wetland': (image.eq(8), fscore[7])
    }
    
    # Process common bands
    processed_bands = []
    for name, (mask, score) in bands.items():
        band = (mask.selfMask()
                .remap([1], [score], 0)
                .rename(name)
                .unmask(0))
        processed_bands.append(band)
    
    # Add snow band for Alps
    if aoi_name == 'Alps':
        snow = (image.eq(9).selfMask()
                .remap([1], [fscore[8]], 0)
                .rename('snow')
                .unmask(0))
        processed_bands.append(snow)
    
    return ee.Image(processed_bands)

# ------------------------- Main Processing Pipeline -------------------------
def process_datasets():
    """Process all datasets and prepare predictor images."""
    image_list = []
    for name in DATASET_NAMES:
        image = dataset.select(name)
        f1_scores = dataset.sampleRegions(parametrisation, scale=10, 
                                        projection='EPSG:3035', tileScale=1)
        f1 = f1_scores.errorMatrix('class', name).fscore(1).getInfo()[1:]
        f1_cleaned = [0 if x == 'NaN' or (isinstance(x, float) and math.isnan(x)) else x for x in f1]
        expanded = prepare_data(image, f1_cleaned, AOI_NAME).float()
        image_list.append(expanded)
    
    return ee.ImageCollection.fromImages(image_list).toBands()

# Prepare predictors
predictors = process_datasets()

# Sample training data
training_samples = predictors.sampleRegions(
    collection=training,
    scale=10,
    projection='EPSG:3035',
    tileScale=16
)

# ------------------------- Classification and Export -------------------------
def train_and_classify():
    """Train Random Forest classifier and generate probability maps."""
    # Train classifier
    classifier = (ee.Classifier.smileRandomForest(
        numberOfTrees=500,
        seed=42
    ).train(
        features=training_samples,
        classProperty='class',
        inputProperties=predictors.bandNames()
    ).setOutputMode('MULTIPROBABILITY'))
    
    # Classify and generate probabilities
    classified = predictors.classify(classifier)
    probabilities = (classified.arrayFlatten([PROB_BANDS[AOI_NAME]])
                     .multiply(10000)
                     .toUint16())
    
    # Add snow band for Carpathians if needed
    if AOI_NAME == 'Carpathians':
        probabilities = probabilities.addBands(
            ee.Image(0).clip(aoi).rename('snowice'))
    
    # Generate labels
    label = (classified.arrayArgmax()
             .arrayProject([0])
             .arrayFlatten([['label']])
             .add(1)
             .toUint16())
    
    return probabilities.addBands(label).clip(aoi).unmask(0)

# Generate final image
result_image = train_and_classify()

# Export results
def export_results(image):
    """Export classification results to Earth Engine asset."""
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=f'ConRF_{AOI_NAME}',
        assetId = ASSET_ID,
        region=aoi.geometry(),
        scale=10,
        crs='EPSG:3035',
        maxPixels=1e10
    )
    task.start()
    print(f"Export task started for {AOI_NAME}")

# Execute export
export_results(result_image)
