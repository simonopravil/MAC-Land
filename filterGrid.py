import rasterio
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

def calculate_proportion_for_cell(args):
    """
    Calculate the proportion of class 5 for a single grid cell
    """
    cell_geometry, raster_path, cell_id = args
    
    # Open raster for each process
    with rasterio.open(raster_path) as raster:
        # Get bounds of the grid cell
        minx, miny, maxx, maxy = cell_geometry.bounds
        
        # Get the window in raster coordinates
        window = raster.window(minx, miny, maxx, maxy)
        
        # Read the data for this window
        data = raster.read(1, window=window)
        
        if data.size == 0:
            return cell_id, 0.0
        
        # Calculate proportion of class 5
        total_pixels = data.size
        class_5_pixels = np.sum(data == 5)
        proportion = class_5_pixels / total_pixels
        
        return cell_id, proportion

def parallel_process_grid(raster_path, grid, threshold, n_processes=None):
    """
    Process the grid in parallel to calculate class 5 proportions
    """
    if n_processes is None:
        n_processes = cpu_count() - 1  # Leave one core free
    
    # Prepare arguments for parallel processing
    args_list = [(row.geometry, raster_path, idx) for idx, row in grid.iterrows()]
    
    # Create pool and map the calculation function
    with Pool(n_processes) as pool:
        results = pool.map(calculate_proportion_for_cell, args_list)
    
    # Convert results to dictionary
    proportions = dict(results)
    
    # Add proportions to grid
    grid['class_5_proportion'] = grid.index.map(proportions)
    
    # Filter based on threshold
    filtered_grid = grid[grid['class_5_proportion'] >= threshold]
    
    return filtered_grid

def main(raster_path, grid_path, output_path, threshold=0.5):
    """
    Main function to orchestrate the analysis
    """
    # Load grid
    print("Loading grid data...")
    grid = gpd.read_file(grid_path)
    
    # Process grid in parallel
    print(f"Processing grid with {cpu_count()-1} processes...")
    filtered_grid = parallel_process_grid(raster_path, grid, threshold)
    
    # Save results
    print("Saving results...")
    filtered_grid.to_file(output_path)
    
    print(f"Analysis complete. Found {len(filtered_grid)} cells above threshold.")
    return filtered_grid

if __name__ == "__main__":
    # Example usage
    raster_path = "RandomForest_labels.tif" 
    grid_path = "grid_2_5km.shp"
    output_path = "grid_2_5km_filtered_0.2.shp"
    threshold = 0.2  #  ``
    
    filtered_grid = main(raster_path, grid_path, output_path, threshold)