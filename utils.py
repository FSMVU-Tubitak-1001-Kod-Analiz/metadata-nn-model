import pandas as pd
import os
import glob

def merge_csv_files_in_directory(directory="inputs"):
    # Get the absolute path of the directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, directory)
    
    # Search for all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print("No CSV files found in the directory!")
        return

    # Tüm CSV dosyalarını birleştir
    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df