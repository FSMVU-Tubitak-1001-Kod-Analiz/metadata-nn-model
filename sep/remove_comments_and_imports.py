import pandas as pd
import re
import os
from utils import clean_java_code



def main():
    """
    Main function to read, process, and write the CSV file.
    """
    input_csv_path = 'merged.csv'
    output_csv_path = 'merged_no_license.csv'
    column_to_process = 'content_before'

    # --- 1. Read the CSV file ---
    print(f"Reading data from '{input_csv_path}'...")
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file '{input_csv_path}' not found.")
        print("Please create it with a column named 'content_before' containing Java code.")
        return

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if column_to_process not in df.columns:
        print(f"Error: Column '{column_to_process}' not found in the CSV file.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # --- 2. Process the column and 3. Overwrite it ---
    print(f"Processing the '{column_to_process}' column...")
    
    # The .apply() method executes our function on each row of the specified column
    df[column_to_process] = df[column_to_process].apply(clean_java_code)
    
    print("Processing complete.")

    # --- 4. Export the data to a new CSV file ---
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully exported the processed data to '{output_csv_path}'.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()