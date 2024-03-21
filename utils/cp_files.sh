#!/bin/bash

# Source directory (change this to your actual directory)
source_dir=$1

# Destination directory (change this to your actual directory)
dest_dir=$2

subsubdir_name=$3

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$dest_dir" ]; then
  mkdir -p "$dest_dir"
fi

# Loop through subdirectories in the source directory
for subdir in "$source_dir"/*; do
  # Check if it's a directory
  if [ -d "$subdir" ]; then
    # Get the subdirectory name
    subdir_name=$(basename "$subdir")
    
    # Create the corresponding subdirectory in the destination directory
    dest_subdir="$dest_dir/$subdir_name"
    mkdir -p "$dest_subdir"
    src_subdir="$subdir/$subsubdir_name"
    
    # Copy all files from the source subdirectory to the destination subdirectory
    cp -r "$src_subdir/"* "$dest_subdir/"
  fi
done

echo "Files copied successfully!"