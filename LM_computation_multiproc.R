# -------------------------------
# Landscape Metrics Analysis Script with furrr and progressr (Optimized)
# -------------------------------

# Import necessary libraries
library(landscapemetrics)  # For calculating landscape metrics
library(terra)             # For raster image processing
library(dplyr)             # For data manipulation
library(furrr)             # For parallel processing with futures
library(progressr)         # For progress bar support

# Define paths
input_folder <- "D:/DATA/LULC_Final/Labels/RF/ls"
output_file  <- "D:/DATA/LULC_Final/Labels/RF/Landscpaes_Results_with_Clumpy_core.csv"

# Define the class you want to compute specific metrics for (e.g., class value 1 for grasslands)
class_value <- 5  # Replace with the class value of interest

# Define the edge buffer distance for core area (e.g., 30 meters)
edge_depth <- 5  # Replace with the desired edge buffer distance

metrics_all_classes <- c(
  "lsm_c_pland",  # Mean patch area
  "lsm_l_shdi",     # Shannon's diversity index (landscape-level)
  "lsm_l_prd"       # Patch richness density (landscape-level)
)

# Define metrics to calculate for ONLY the specified class
metrics_specific_class <- c(
  "lsm_c_lpi",      # Largest patch index
  "lsm_c_clumpy",   # Clumpiness index
  "lsm_c_pafrac",   # Perimeter-area fractal dimension
  "lsm_c_core_mn",   # Mean core area (requires edge_depth)
  "lsm_c_pd",       # Patch density
  "lsm_c_ed",       # Edge density
  "lsm_c_enn_mn"    # Mean Euclidean nearest-neighbor distance
)

# Combine all metrics for reporting purposes
metrics <- c(metrics_all_classes, metrics_specific_class)

# List image files from the input folder (adjust the pattern as needed)
image_files <- list.files(input_folder, pattern = "\\.(tif|jpg|png)$", full.names = TRUE)

# Sort numerically based on filename (without extension)
image_files <- image_files[order(as.numeric(tools::file_path_sans_ext(basename(image_files))))]

# Define a function to process a single image chip.
process_image <- function(file_path) {
  # Load the image as a raster object
  img <- rast(file_path)
  
  # Calculate metrics for ALL classes
  df_all_classes <- calculate_lsm(
    img,
    what = metrics_all_classes,
    progress = FALSE
  )
  
  # Calculate metrics for ONLY the specified class
  df_specific_class <- calculate_lsm(
    img,
    what = metrics_specific_class,
    edge_depth = edge_depth,
    progress = FALSE
  )
  
  # Combine the results into a single data frame
  df_metrics <- bind_rows(df_all_classes, df_specific_class)
  
  # Add a Patch_ID column (using the file name)
  df_metrics <- df_metrics %>% mutate(Patch_ID = tools::file_path_sans_ext(basename(file_path)))
  
  return(df_metrics)
}

# -------------------------------
# Setup Parallel Processing with furrr and progressr (Optimized)
# -------------------------------

# Determine the number of cores to use (use all available cores)
num_workers <- parallel::detectCores()

# Set up a multisession plan using furrr (works well on Windows and Unix alike)
plan(multisession, workers = num_workers)

# Set up progressr to report progress in the console
handlers(global = TRUE)
handlers("txtprogressbar")  # you can also choose "progress" for a different style

# Pre-load necessary libraries and data in each worker
future::plan(future::multisession, workers = num_workers)
future::supportsMulticore()  # Ensure multicore support is available

# Wrap the parallel mapping call within a with_progress block to show a progress bar.
# future_map() from furrr will distribute the tasks among workers.
result_list <- with_progress({
  p <- progressor(along = image_files)
  furrr::future_map(image_files, function(file_path) {
    # Pre-load libraries in each worker
    library(landscapemetrics)
    library(terra)
    library(dplyr)
    
    res <- process_image(file_path)
    p()  # update the progress bar
    res
  }, .options = furrr_options(seed = TRUE))  # Ensure reproducibility
})

# Combine all the individual dataframes into one big dataframe
combined_results <- dplyr::bind_rows(result_list)

# Save the combined results to the output CSV file
write.csv(combined_results, output_file, row.names = FALSE)

# Print a completion message
cat("Landscape metrics analysis complete. Results saved to:", output_file, "\n")