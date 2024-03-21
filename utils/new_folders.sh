#!/bin/bash

# Set the source directory (modify if needed)
source_dir="../augmented_directory"

# Create the destination directory (modify if needed)
# dest_dir="Original"
# dest_dir="../augmented_directory_files_not_used"
# dest_dir="../saved_originals_for_testing"
# dest_dir="../balanced_directory"
dest_dir="../train_directory"

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' does not exist."
  exit 1
fi

# Check if destination directory exists and warn if it does
if [ -d "$dest_dir" ]; then
  echo "Warning: Destination directory '$dest_dir' already exists. Existing files may be overwritten."
fi

# Get subdirectories in the source directory
for subdir in "$source_dir"/*; do
  # Check if it's a directory (optional)
  echo $subdir
  if [ -d "$subdir" ]; then
    # Extract the subdirectory name, handling potential special characters
    dir_name="${subdir##*/}"
    echo $dir_name
    # Escape any remaining special characters in the directory name (optional)
    # dir_name="${dir_name//\//\\}"
   
    # Create the directory in the destination folder
    mkdir -p "$dest_dir/$dir_name"
  fi
done

echo "Created folders in '$dest_dir' with the same names as subdirectories in '$source_dir'."