"""
Sync Previous Module Files

This script helps you sync files from your previous module to the current module.
It copies files specified in 'files_to_sync.txt' from the source directory to the destination directory.

Usage: python sync_previous_module.py <source_directory> <destination_directory>

Examples:
    python sync_previous_module.py ./my-awesome-module-1 ./my-awesome-module-2
    python sync_previous_module.py ~/assignments/Module-1-unicorn_ninja ~/assignments/Module-2-unicorn_ninja
"""
import os
import shutil
import sys

def print_usage():
    """Print usage information and examples."""
    print(__doc__)

def read_files_to_sync():
    """Read the list of files to sync from files_to_sync.txt"""
    try:
        with open("files_to_sync.txt", "r") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print("Error: files_to_sync.txt not found!")
        sys.exit(1)

def sync_files(source, dest, files_to_move):
    """Copy files from source to destination directory."""
    if not os.path.exists(source):
        print(f"Error: Source directory '{source}' does not exist!")
        sys.exit(1)

    if not os.path.exists(dest):
        print(f"Error: Destination directory '{dest}' does not exist!")
        sys.exit(1)

    copied_files = 0
    for file in files_to_move:
        source_path = os.path.join(source, file)
        dest_path = os.path.join(dest, file)

        if not os.path.exists(source_path):
            print(f"Warning: File '{file}' not found in source directory, skipping")
            continue

        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(source_path, dest_path)
            print(f"Copied: {file}")
            copied_files += 1
        except Exception as e:
            print(f"Error copying '{file}': {e}")

    print(f"Finished copying {copied_files} files")

def main():
    if len(sys.argv) != 3:
        print("Error: Invalid number of arguments!")
        print_usage()
        sys.exit(1)

    source = sys.argv[1]
    dest = sys.argv[2]
    files_to_move = read_files_to_sync()

    sync_files(source, dest, files_to_move)

if __name__ == "__main__":
    main()
