import h5py
import os
import shutil
import numpy as np
from src.utils.logger_functions import console


def find_unmatched_h5_files(folder_path: str, threshold=0.9, delete=False) -> list:
    """
    Find .h5 files in the specified folder that do not have matching .pkl files, indicating that there was an issue
    in the processing of the file.

    :param folder_path: (str) The path to the folder containing the files.
    :param threshold: (float) The threshold percentage to warn about potential processing issues.
    :param delete: (bool) Whether to prompt the user to delete unmatched .h5 files.
    :return: (list) A list of unmatched .h5 file names.
    """

    # Lists to store the base names of files without extensions
    pkl_files_base = set()
    h5_files_base = set()

    # List to store the unmatched .h5 files
    unmatched_h5_files = []

    # Iterate through files in the directory
    for filename in os.listdir(folder_path):
        # Check for .pkl files and add the base name to pkl_files_base
        if filename.endswith('.pkl'):
            base_name = os.path.splitext(filename)[0]
            if base_name.endswith("_processed"):
                base_name = base_name[:-len("_processed")]
            pkl_files_base.add(base_name)

        # Check for .h5 files and add the base name to h5_files_base
        elif filename.endswith('.h5'):
            base_name = os.path.splitext(filename)[0]
            if base_name.startswith("Sorter_"):
                base_name = base_name[len("Sorter_"):]
            h5_files_base.add(base_name)

    # Find .h5 files that do not have a matching .pkl file
    for base_name in h5_files_base:
        if base_name not in pkl_files_base:
            unmatched_h5_files.append(f"{base_name}.h5")

    # Check if the percentage of unmatched files is above the threshold, indicating that there was a processing issue.
    if len(unmatched_h5_files) / len(h5_files_base) >= threshold:
        console.warning(f"More than 90% of files were not properly processed.\n"
                        f"Found {len(unmatched_h5_files)} unmatched files out of {len(h5_files_base)} .h5 files.\n"
                        f"Delete the files manually after checking the naming convention.")

    else:
        console.info(f"Found {len(unmatched_h5_files)} unmatched files out of {len(h5_files_base)} .h5 files.")

        if delete:
            # Ask the user if the files should be deleted.
            if unmatched_h5_files:
                user_input = input("Do you want to delete the directories named with these unmatched .h5 files? "
                                   "(yes/no): ").strip().lower()

                if user_input == 'yes':
                    delete_files_in_list(folder_path=folder_path, list_of_files=unmatched_h5_files)
                    print("Directories deleted.")
                else:
                    print("No directories were deleted.")
            else:
                print("No unmatched .h5 files found.")

    return unmatched_h5_files


def delete_files_in_list(folder_path: str, list_of_files: list):
    """
    Delete the directories listed in the list_of_files from the specified folder_path.

    :param folder_path: (str) The path to the folder containing the directories.
    :param list_of_files: (list) A list of directory names to delete.
    """

    # Delete the directories named in the list of files
    for file in list_of_files:
        dir_to_delete = os.path.join(folder_path, "Sorter_"+file)
        if os.path.isdir(dir_to_delete):
            console.info(f"Deleting directory: {dir_to_delete}")
            shutil.rmtree(dir_to_delete)


def split_hdf5_file(input_path: str, filename: str, num_splits=5):
    """
    Split an HDF5 file into multiple subfiles by dividing the 'sig' dataset (raw traces).

    :param input_path: (str) The path to the folder containing the input HDF5 file.
    :param filename: (str) The name of the HDF5 file to split.
    :param num_splits: (int) The number of subfiles to create.
    """

    console.info(f"Starting splitting of file {filename} into {num_splits} subfiles.")
    try:
        # Open the original HDF5 file in read mode
        with h5py.File(os.path.join(input_path, filename), "r") as file:
            # Extract the raw signal
            raw_data = file.get("sig")

            # Determine the frame range and split points
            max_frame = raw_data.shape[1]
            split_frames = np.linspace(0, max_frame, num_splits + 1)

            # Split the spike times into different sequences
            spike_splits = []
            for i in range(num_splits):
                start_frame = int(split_frames[i])
                end_frame = int(split_frames[i + 1])

                # Split the raw data
                split_raw_data = raw_data[:, start_frame:end_frame]

                # Create new HDF5 files
                output_file = f"{filename[:-3]}_part{i}.h5"
                shutil.copy(os.path.join(input_path, filename), os.path.join(input_path, output_file))

                with h5py.File(os.path.join(input_path, output_file), 'r+') as out_file:
                    # Replace the raw trace with the split raw trace
                    del out_file['sig']
                    out_file.create_dataset('sig', data=split_raw_data)

                console.info(f"Subfile {i} of file {filename} saved successfully.")

            console.info(f"Splitting of file {filename} completed successfully.")

    except Exception as e:
        console.error(f"An error occurred while loading data: {str(e)}")
        return []  # Return empty list to indicate an error
