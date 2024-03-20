#!/bin/bash

# Define source and destination directories (update these paths)
source_dir="$1"
dest_dir="$2"

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' does not exist."
  exit 1
fi

# Check if destination directory exists (and create if not)
if [ ! -d "$dest_dir" ]; then
  mkdir -p "$dest_dir"
fi

# Loop through subdirectories in the source directory
for subdir in "$source_dir"/*; do
  # Check if it's a directory (skip files)
  if [ -d "$subdir" ]; then
    # Extract subdirectory name (remove leading and trailing slashes)
    subdir_name="${subdir##*/}"
    subdir_name="${subdir_name%%/}"

    # Create new directory in destination with the subdirectory name
    new_dir="$dest_dir/$subdir_name"
    mkdir -p "$new_dir"

    echo "Created directory: $new_dir"
    # Find JPG files, shuffle, and select 15 while preserving filenames with spaces:
    all_jpgs=$(find "$subdir" -type f -name "*.JPG")

    while IFS= read -r jpg; do
      shuffled_jpgs+=("$jpg")  # Add each file to an array to preserve spaces
    done <<< "$(sort -R <<< "$all_jpgs")"
    shuffled_outputs=("${shuffled_jpgs[@]:0:15}")  # Select first 15 elements

    # Move the 15 randomly chosen JPGs:
    for jpg in "${shuffled_outputs[@]}"; do
      # echo "Moving: $jpg"  # Uncomment to see which files are being moved
      mv "$jpg" "$new_dir"

    # reinitialize shuffled_jpgs variable
    unset shuffled_jpgs
done
  fi
done