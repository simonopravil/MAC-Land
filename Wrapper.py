"""
Utility Functions for Land Use Land Cover (LULC) Analysis with Google Earth Engine

This script provides helper functions for retrieving and processing LULC data, creating
validation datasets, generating confusion matrices, and calculating classification accuracy
metrics for specified Areas of Interest (AOIs).
"""

# Standard library imports
import math
import itertools

# Third-party imports
import ee
import pandas as pd
import numpy as np

# ------------------------- Initialization -------------------------
def initialize_earth_engine():
    """Initialize Google Earth Engine."""
    try:
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com',
                     project='ee-alpscarpathians')
    except Exception as e:
        raise Exception(f"Failed to initialize Earth Engine: {str(e)}")

# Uncomment to initialize Earth Engine if needed
# initialize_earth_engine()

# ------------------------- Data Retrieval Functions -------------------------
def get_data(aoi_name: str, return_as_list: bool = False) -> ee.Image or list:
    """
    Retrieves land cover data for a specified area of interest (AOI).

    Args:
        aoi_name (str): Name of the AOI (e.g., 'Carpathians', 'Alps').
        return_as_list (bool): If True, returns a list of images; if False, returns a composite image.

    Returns:
        ee.Image or list: Composite image or list of individual images containing land cover data.
    """
    # Define AOI
    aoi = (ee.FeatureCollection('projects/ee-simonopravil/assets/LULC/Alps_Carph')
           .filter(ee.Filter.eq('m_range', aoi_name)))

    # Dataset configurations with remapping rules
    datasets = {
        'elc': {'path': 'projects/ee-simonopravil/assets/LULC/ELC', 'bands': ['b1'], 'remap': None},
        'clc': {'path': 'projects/ee-simonopravil/assets/LULC/CLC',
                'remap': ([1, 7, 2, 3, 4, 5, 6, 9, 10, 11], [1, 2, 3, 3, 3, 4, 5, 6, 7, 9])},
        'glc': {'path': 'projects/ee-simonopravil/assets/LULC/GLC',
                'remap': ([62, 73, 75, 82, 83, 103, 102, 121, 162, 105, 106, 123], [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9])},
        'dw': {'path': 'projects/ee-simonopravil/assets/LULC/DW',
               'remap': ([6, 4, 1, 5, 2, 7, 0, 3, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9])},
        'esri': {'path': 'projects/ee-simonopravil/assets/LULC/ESRI',
                 'remap': ([7, 5, 2, 6, 3, 8, 1, 4, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9])},
        'esa': {'path': 'projects/ee-simonopravil/assets/LULC/ESA',
                'remap': ([50, 40, 10, 20, 30, 60, 100, 80, 90, 95, 70], [1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9])}
    }

    images = []
    for name, config in datasets.items():
        image = ee.Image(config['path'])
        band_name = 'label' if return_as_list else name

        if config['remap']:
            image = (image.remap(*config['remap'], 0)
                     .select(['remapped'], [band_name]))
        else:
            image = image.select(config['bands'], [band_name])

        image = (image.clip(aoi)
                 .toInt()
                 .set({'name': name}))
        images.append(image)

    return images if return_as_list else ee.Image(images)

# ------------------------- LUCAS Data Processing -------------------------
def create_lucas(lucas_points: ee.FeatureCollection, lucas_poly: ee.FeatureCollection,
                 aoi: ee.FeatureCollection) -> ee.FeatureCollection:
    """
    Process and filter LUCAS points and polygons to create a validation dataset.

    Args:
        lucas_points (ee.FeatureCollection): LUCAS point data.
        lucas_poly (ee.FeatureCollection): LUCAS polygon data.
        aoi (ee.FeatureCollection): Area of interest to filter data.

    Returns:
        ee.FeatureCollection: Processed and filtered LUCAS data.
    """
    def map_lc1_to_numeric(feature):
        """Map LUCAS lc1 values to numeric class values."""
        lc1 = ee.String(feature.get("lc1"))
        numeric_value = ee.Algorithms.If(
            lc1.compareTo('G50').eq(0),
            9,
            ee.Dictionary({
                "A": 1, "B": 2, "C": 3, "D": 4,
                "E": 5, "F": 6, "G": 7, "H": 8
            }).get(lc1.slice(0, 1), 0)
        )
        return feature.set("class", numeric_value)

    def lucas_filter(lucas):
        """Apply filters to LUCAS data."""
        filters = [
            ee.Filter.neq('lc1', 'F40'),  # Exclude other bare land
            ee.Filter.equals('lc1_perc', "> 75 %"),
            ee.Filter.inList('parcel_area_ha', ["1 - 10 ha", "> 10 ha"])
        ]
        filter_office_pi = ee.Filter.eq('obs_type', "In office PI")
        filter_gps_prec = ee.Filter.Or(ee.Filter.lt('gps_prec', 15), filter_office_pi)
        filter_obs_dist = ee.Filter.Or(ee.Filter.lt('obs_dist', 15), filter_office_pi)
        combined_filter = ee.Filter.And(*filters, filter_gps_prec, filter_obs_dist)
        return lucas.filter(combined_filter)

    def centroids(feature):
        """Convert polygons to centroids."""
        return feature.centroid(maxError=1, proj='EPSG:3035')

    # Process points and polygons
    points = lucas_points.map(map_lc1_to_numeric)
    polygons = lucas_poly.map(map_lc1_to_numeric)
    lucas_points_filtered = lucas_filter(points)
    lucas_poly_filtered = (polygons
                          .map(centroids)
                          .filter(ee.Filter.eq('copernicus_cleaned', True)))

    # Merge and finalize
    return (lucas_points_filtered
            .merge(lucas_poly_filtered)
            .select(['id', 'lc1', 'class'])
            .randomColumn(seed=3)
            .filterBounds(aoi.geometry()))

# ------------------------- Voting and Classification Functions -------------------------
def create_votes(ds_list: list, class_num: int, number_of_leftouts: int) -> ee.Image:
    """
    Generate votes for a specific class across dataset combinations.

    Args:
        ds_list (list): List of dataset images.
        class_num (int): Class number to create votes for.
        number_of_leftouts (int): Number of datasets to exclude in combinations.

    Returns:
        ee.Image: Image with voting results for each combination.
    """
    def mask_class(map_class):
        def wrap(image):
            mask = image.eq(map_class)
            return image.updateMask(mask).unmask(0).copyProperties(image)
        return wrap

    def votes_by_class(image_collection, map_class, num_datasets):
        masked_collection = image_collection.map(mask_class(map_class))
        votes = masked_collection.reduce(ee.Reducer.sum()).divide(map_class).divide(num_datasets)
        return votes

    final_image = None
    combinations = list(itertools.combinations(ds_list, number_of_leftouts))

    for selected_datasets in combinations:
        num_datasets = len(selected_datasets)
        subset_ds = ee.ImageCollection.fromImages(selected_datasets)
        image_names = subset_ds.aggregate_array('name').getInfo()
        new_name = '_'.join(image_names)

        votes = votes_by_class(subset_ds, class_num, num_datasets).select(['label_sum'], [new_name])
        final_image = votes if final_image is None else ee.Image([final_image, votes])

    return final_image

# ------------------------- Accuracy Assessment Functions -------------------------
def calculate_class_proportions(image: ee.Image, band: str,
                                aoi: ee.FeatureCollection) -> pd.DataFrame:
    """
    Calculate class proportions for a given band in an image.

    Args:
        image (ee.Image): Input image.
        band (str): Band name to analyze.
        aoi (ee.FeatureCollection): Area of interest.

    Returns:
        pd.DataFrame: Class proportions as weights (Wi).
    """
    area_image = ee.Image.pixelArea().addBands(image.select(band))
    areas = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName=band),
        geometry=aoi.geometry(),
        scale=10,
        maxPixels=1e12
    )

    def format_area(item):
        area_dict = ee.Dictionary(item)
        class_number = ee.Number(area_dict.get(band)).format()
        area = ee.Number(area_dict.get('sum')).round()
        return ee.List([class_number, area])

    class_areas = ee.List(areas.get('groups'))
    class_area_lists = class_areas.map(format_area)
    result = ee.Dictionary(class_area_lists.flatten())
    total_sum = result.values().reduce('sum')
    class_proportions = result.map(lambda key, value: ee.Number(value).divide(total_sum))
    weights = pd.DataFrame(class_proportions.getInfo(), index=['Wi']).transpose()
    weights.index = weights.index.astype(int)
    return weights

def create_confusion_matrix(validation_data: ee.FeatureCollection, band: str) -> pd.DataFrame:
    """
    Create a confusion matrix from validation data.

    Args:
        validation_data (ee.FeatureCollection): Validation data.
        band (str): Band to evaluate.

    Returns:
        pd.DataFrame: Confusion matrix.
    """
    ee_cm = validation_data.errorMatrix('class', band)
    cm = pd.DataFrame(ee_cm.getInfo())[1:].drop(0, axis=1)
    return cm

def calculate_proportion_confusion_matrix(confusion_matrix: pd.DataFrame,
                                          weights: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the proportional confusion matrix using class weights.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.
        weights (pd.DataFrame): Class weights (Wi).

    Returns:
        pd.DataFrame: Proportional confusion matrix.
    """
    new_cm = confusion_matrix.copy(deep=True)
    new_cm.loc['Total', :] = new_cm.sum(axis=0)
    new_cm.loc[:, 'Total'] = new_cm.sum(axis=1)

    class_names = new_cm.columns[:-1]
    new_values = {}
    for class_name in class_names:
        try:
            wi = weights.loc[class_name, 'Wi']
            nij_values = new_cm.loc[class_name, class_names]
            ni = new_cm.loc[class_name, 'Total']
            new_row = wi * nij_values / ni
            new_values[class_name] = new_row
        except KeyError:
            if class_name in [8, 9]:
                new_values[class_name] = pd.Series(0, index=range(1, 10 if class_name == 9 else 9))
                continue
            else:
                raise

    return pd.DataFrame(new_values).transpose()

def overall_accuracy(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall accuracy from a confusion matrix.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.

    Returns:
        pd.DataFrame: Overall accuracy.
    """
    num_samples = confusion_matrix.sum().sum()
    correct_predictions = np.nan_to_num(np.diag(confusion_matrix)).sum()
    oa = correct_predictions / num_samples
    return pd.DataFrame([oa], index=[1], columns=['OA'])

def user_accuracy(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate user accuracy per class.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.

    Returns:
        pd.DataFrame: User accuracy per class.
    """
    row_sums = confusion_matrix.sum(axis=1)
    correct_predictions = np.diag(confusion_matrix)
    user_acc = correct_predictions / row_sums
    return pd.DataFrame(user_acc)

def producer_accuracy(confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate producer accuracy per class.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.

    Returns:
        pd.DataFrame: Producer accuracy per class.
    """
    column_sums = confusion_matrix.sum(axis=0)
    correct_predictions = np.diag(confusion_matrix)
    producer_acc = correct_predictions / column_sums
    return pd.DataFrame(producer_acc)

def oa_standard_error(confusion_matrix: pd.DataFrame, weights: pd.DataFrame,
                      user_accuracy_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """
    Calculate standard error for overall accuracy.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.
        weights (pd.DataFrame): Class weights.
        user_accuracy_df (pd.DataFrame): User accuracy per class.
        confidence (float): Confidence level for standard error calculation.

    Returns:
        pd.DataFrame: Standard error of overall accuracy.
    """
    user_accuracy_df = user_accuracy_df.rename(columns={0: 'UserAccuracy'})
    sample_counts = pd.DataFrame(confusion_matrix.sum(axis=1)).rename(columns={0: 'NumSamples'})
    var_components = (weights['Wi'] ** 2) * (user_accuracy_df['UserAccuracy'] * (1 - user_accuracy_df['UserAccuracy'])) / (sample_counts['NumSamples'] - 1)
    overall_variance = var_components.sum()
    error = confidence * np.sqrt(overall_variance)
    return pd.DataFrame([error], index=[1], columns=['OA_StdError'])

def ua_standard_error(confusion_matrix: pd.DataFrame, user_accuracy_df: pd.DataFrame,
                      confidence: float) -> pd.DataFrame:
    """
    Calculate standard error for user accuracy per class.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.
        user_accuracy_df (pd.DataFrame): User accuracy per class.
        confidence (float): Confidence level for standard error calculation.

    Returns:
        pd.DataFrame: Standard error of user accuracy per class.
    """
    user_accuracy_df = user_accuracy_df.rename(columns={0: 'UserAccuracy'})
    sample_counts = pd.DataFrame(confusion_matrix.sum(axis=1)).rename(columns={0: 'NumSamples'})
    errors = {}
    for class_name in sample_counts.index:
        sample_sum = sample_counts.loc[class_name, 'NumSamples']
        user_acc = user_accuracy_df.loc[class_name, 'UserAccuracy']
        error = confidence * np.sqrt(user_acc * (1 - user_acc) / (sample_sum - 1))
        errors[class_name] = error
    return pd.DataFrame.from_dict(errors, orient='index', columns=['UserAccuracy_StdError'])

def pa_standard_error(confusion_matrix: pd.DataFrame, weights: pd.DataFrame,
                      user_accuracy_df: pd.DataFrame, producer_accuracy_df: pd.DataFrame,
                      confidence: float) -> pd.DataFrame:
    """
    Calculate standard error for producer accuracy per class.

    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix.
        weights (pd.DataFrame): Class weights.
        user_accuracy_df (pd.DataFrame): User accuracy per class.
        producer_accuracy_df (pd.DataFrame): Producer accuracy per class.
        confidence (float): Confidence level for standard error calculation.

    Returns:
        pd.DataFrame: Standard error of producer accuracy per class.
    """
    user_acc = user_accuracy_df.to_numpy()
    producer_acc = producer_accuracy_df.to_numpy()
    proportion_cm = calculate_proportion_confusion_matrix(confusion_matrix, weights)
    N_j = proportion_cm.sum(axis=0).to_numpy()
    Ni_ = proportion_cm.sum(axis=1).to_numpy()
    nij_matrix = confusion_matrix.to_numpy()
    nj_ = np.sum(nij_matrix, axis=1)

    variance_per_class = np.zeros(len(producer_acc))
    stderror_per_class = np.zeros(len(producer_acc))

    for j in range(len(producer_acc)):
        inner_sum = 0
        for i in range(len(producer_acc)):
            if i != j and nj_[i] > 1:
                inner_sum += (Ni_[i] ** 2) * (nij_matrix[i, j] / nj_[i]) * (1 - nij_matrix[i, j] / nj_[i]) / (nj_[i] - 1)

        term1 = 0
        if nj_[j] > 1:
            term1 = (Ni_[j] ** 2) * ((1 - producer_acc[j]) ** 2) * user_acc[j] * (1 - user_acc[j]) / (nj_[j] - 1)
        term2 = (producer_acc[j] ** 2) * inner_sum

        if N_j[j] > 0:
            variance_per_class[j] = (1 / (N_j[j] ** 2)) * (term1 + term2)
            stderror_per_class[j] = confidence * np.sqrt(variance_per_class[j])

    return pd.DataFrame(stderror_per_class, columns=['ProdAccuracy_StdError'])

def calculate_confusion_matrix_stats(image: ee.Image, validation_data: ee.FeatureCollection,
                                     band: str, aoi: ee.FeatureCollection,
                                     confidence: float) -> pd.DataFrame:
    """
    Calculate comprehensive statistics from a confusion matrix.

    Args:
        image (ee.Image): Classified image.
        validation_data (ee.FeatureCollection): Validation data.
        band (str): Band to evaluate.
        aoi (ee.FeatureCollection): Area of interest.
        confidence (float): Confidence level for standard error calculations.

    Returns:
        pd.DataFrame: Combined statistics including accuracies and standard errors.
    """
    weights = calculate_class_proportions(image, band, aoi)
    confusion_matrix = create_confusion_matrix(validation_data, band)
    prop_confusion_matrix = calculate_proportion_confusion_matrix(confusion_matrix, weights)

    overall_acc_df = overall_accuracy(prop_confusion_matrix)
    user_acc_df = user_accuracy(prop_confusion_matrix)
    producer_acc_df = producer_accuracy(prop_confusion_matrix)

    oa_se = oa_standard_error(confusion_matrix, weights, user_acc_df, confidence)
    ua_se = ua_standard_error(confusion_matrix, user_acc_df, confidence)
    pa_se = pa_standard_error(confusion_matrix, weights, user_acc_df, producer_acc_df, confidence)

    stats = {
        'Error_Matrix': confusion_matrix,
        'Proportional_Error_Matrix': prop_confusion_matrix,
        'U': user_acc_df.rename(columns={0: 'User_Accuracy'}),
        'UA_Se': ua_se,
        'P': producer_acc_df.rename(columns={0: 'Prod_Accuracy'}),
        'PA_Se': pa_se,
        'O': overall_acc_df,
        'OA_Se': oa_se
    }

    return pd.concat(list(stats.values()), axis=1)