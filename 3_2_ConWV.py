"""
Voting-Based LULC Classification and Export for Alps and Carpathians

This script performs Land Use Land Cover (LULC) classification using a voting-based approach 
across multiple datasets for specified Areas of Interest (AOIs) - Alps or Carpathians. It 
processes input datasets, applies F-scores for weighting, computes votes, and exports the 
resulting probability and label maps.
"""

import os
import sys
import math


import ee
import geemap
import numpy as np
import pandas as pd

from Wrapper import getData




# Configuration parameters
AOI_NAME = 'Carpathians'  # Options: 'Carpathians', 'Alps'
VERSION = 'v6'
DATASET_NAMES = ['dw', 'esri', 'esa', 'elc', 'clc', 'glc']
PROB_BANDS = {
    'Alps': ['artificial', 'cropland', 'woodland', 'shrubland',
             'grassland', 'bareland', 'water', 'wetland', 'snowice'],
    'Carpathians': ['artificial', 'cropland', 'woodland', 'shrubland',
                    'grassland', 'bareland', 'water', 'wetland']
}

PROJECT_ID = 'ee-alpscarpathians'
ASSET_ID = f'projects/{PROJECT_ID}/assets/LULC/{AOI_NAME}/Con_WV',

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
    """Load spatial data including AOI and parameterization data."""
    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph')
           .filter(ee.Filter.eq('m_range', AOI_NAME)))
    
    parametrisation = (ee.FeatureCollection(f'projects/ee-simonopravil/assets/LULC/{AOI_NAME}/Parametrisation_{AOI_NAME}_merged')
                       .select('class')
                       .filter(ee.Filter.neq('class', 0)))
    
    return aoi, parametrisation

# Load spatial data
aoi, parametrisation = load_spatial_data()

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

# Sample parameterization data and prepare weighted images
samples = dataset.sampleRegions(parametrisation, scale=10, projection='EPSG:3035', tileScale=1)
image_list = []
for name in DATASET_NAMES:
    image = dataset.select(name)
    f1_scores = samples.errorMatrix('class', name).fscore(1).getInfo()[1:]
    f1_cleaned = [0 if x == 'NaN' or (isinstance(x, float) and math.isnan(x)) else x for x in f1_scores]
    expanded = prepare_data(image, f1_cleaned, AOI_NAME).float()
    image_list.append(expanded)

# ------------------------- Voting and Classification -------------------------
# Create an ImageCollection from weighted images and compute votes
dataset_collection = ee.ImageCollection.fromImages(image_list)
votes = dataset_collection.sum().divide(len(DATASET_NAMES))
band_names = votes.bandNames()

# Rename bands and scale probabilities
votes = (votes
         .select(band_names, PROB_BANDS[AOI_NAME])
         .multiply(10000)
         .toUint16())

# Generate class labels from votes
label = (votes.toArray()
         .arrayArgmax()
         .arrayProject([0])
         .arrayFlatten([['label']])
         .add(1)
         .toUint16())

# Combine probabilities and labels
res_prob_img = votes.addBands(label)

# ------------------------- Export Results -------------------------
# Export the final image to Earth Engine asset
task = ee.batch.Export.image.toAsset(
    image=res_prob_img.clip(aoi),
    description=f'Votes_{AOI_NAME}',
    assetId=f'projects/ee-simonopravil/assets/LULC/{AOI_NAME}/Votes',
    region=aoi.geometry(),
    scale=10,
    crs='EPSG:3035',
    maxPixels=1e10
)
task.start()
print(f"Export task started for {AOI_NAME}")