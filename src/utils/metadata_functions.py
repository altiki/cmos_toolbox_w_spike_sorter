import os
import pandas as pd
from src.utils.logger_functions import console


def load_metadata_as_dataframe(file_path: str, auto_create_metadata=True):
    """
    Load metadata as a Pandas DataFrame, considering the first row as column names.

    :param div_gbm_addition: DIV that GBM was added to the coculture.
    :param auto_create_metadata: Whether a metadata.xlsx file should be automatically created if none is found.
    :param file_path: (str) The path to the metadata.
    :return df: (pd.DataFrame) A DataFrame containing the data from the metadata with column names from the first row.
    """
    try:
        # Load the Excel file into a DataFrame, considering the first row as column names
        #df = pd.read_excel(os.path.join(file_path, "metadata.xlsx"))
        df = pd.read_csv(os.path.join(file_path, "metadata.csv"))
    except FileNotFoundError:
        if auto_create_metadata:
            console.info(f"The file '{file_path}' does not exist. Creating metadata now...")

            # List all HDF5 files in the specified folder
            hdf5_files = [f for f in os.listdir(file_path) if f.endswith(".h5")]

            # Initialize an empty DataFrame with the desired columns
            metadata_df = pd.DataFrame(columns=["Filename", "DIV", "Chip_ID", "Network_ID"])

            # Iterate through each HDF5 file and extract metadata
            for hdf5_file in hdf5_files:
                filename_parts = hdf5_file.split("_")
                print(filename_parts)
                #if filename_parts[0] == '.':
                #    continue

                # Extract metadata from the filename
                div = int(filename_parts[2].replace("DIV", ""))
                chip_id = int(filename_parts[0].replace("ID", ""))
                network_id = filename_parts[1]
                #network_id = int(filename_parts[1].replace("N", ""))
		        
                #cell_type = filename_parts[6].replace(".raw.h5", "")

                # Append metadata to the DataFrame
                file_df = pd.DataFrame({
                    "Filename": [hdf5_file],
                    "DIV": [div],
                    "Chip_ID": [chip_id],
                    "Network_ID": [network_id]
                    #"Cell_Type": [cell_type]
                })

                metadata_df = pd.concat((metadata_df, file_df), ignore_index=True)

            # Save the metadata DataFrame to an Excel file
            metadata_df.to_csv(file_path + "metadata.csv", index=False)

            console.info(f"Metadata successfully created and saved. \n First 5 rows: \n {metadata_df.head()}")
            return metadata_df

        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    else:
        console.info(f"Metadata successfully loaded. \n First 5 rows: \n {df.head()}")
        return df

