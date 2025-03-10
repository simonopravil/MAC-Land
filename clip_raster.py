import rasterio
import geopandas as gpd
from rasterio.mask import mask
import os
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

def clip_raster_by_cell(args):
    """
    Clip raster by a single grid cell and save to file
    """
    cell_geometry, raster_path, output_dir, cell_id, id_column_name = args
    
    try:
        # Create output filename using the specified ID column
        output_path = os.path.join(output_dir, f"{cell_id}.tif")
        
        # Skip if output already exists
        if os.path.exists(output_path):
            return cell_id, True
        
        # Open the raster
        with rasterio.open(raster_path) as src:
            # Perform the clip
            out_image, out_transform = mask(src, 
                                          [cell_geometry],
                                          crop=True,
                                          all_touched=True)
            
            # Skip empty rasters
            if out_image.sum() == 0:
                return cell_id, False
            
            # Copy the metadata
            out_meta = src.meta.copy()
            
            # Update metadata for the clipped raster
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"  # Add compression to reduce file size
            })
            
            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
                
        return cell_id, True
        
    except Exception as e:
        print(f"Error processing cell {cell_id}: {str(e)}")
        return cell_id, False

def parallel_clip_raster(raster_path, grid, output_dir, id_column_name='id', n_processes=None):
    """
    Clip raster in parallel using grid cells
    """
    if n_processes is None:
        n_processes = cpu_count() - 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the specified ID column exists
    if id_column_name not in grid.columns:
        raise ValueError(f"ID column '{id_column_name}' not found in the grid file.")
    
    # Prepare arguments for parallel processing
    args_list = [(row.geometry, raster_path, output_dir, str(row[id_column_name]), id_column_name) 
                 for idx, row in grid.iterrows()]
    
    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(clip_raster_by_cell, args_list)
    
    # Count successful clips
    successful = sum(1 for _, success in results if success)
    return successful

def main(raster_path, grid_path, output_dir, id_column_name='id'):
    """
    Main function to orchestrate the clipping process
    """
    # Load grid
    print("Loading grid data...")
    grid = gpd.read_file(grid_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process grid in parallel
    print(f"Processing grid with {cpu_count()-1} processes...")
    successful = parallel_clip_raster(raster_path, grid, output_dir, id_column_name)
    
    print(f"Clipping complete. Successfully processed {successful} out of {len(grid)} cells.")
    return successful

if __name__ == "__main__":
    # Example usage
    raster_path = "RandomForest_labels.tif"
    grid_path = "grid_2_5km_filtered_0.2.shp"
    output_dir = "ls25"
    
    # Specify the ID column name from your grid file
    id_column_name = "grid_id"  # Change this to match your actual ID column name
    
    successful = main(raster_path, grid_path, output_dir, id_column_name)