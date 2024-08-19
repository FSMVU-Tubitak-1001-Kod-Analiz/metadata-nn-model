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

    # Birleştirilmiş dosyayı kaydet
    merged_df.to_csv('merged.csv', index=False)
    print(f"Files have been merged and saved as merged.csv")


    return merged_df

# Standardizes labels and removes rows without matching keywords
def standardize_labels(dataframe, keywords):
    def replace_label(label, keywords):
        # Normalize the label to lowercase
        label_lower = label.lower()

        for keyword in keywords:
            # Check if the keyword is in the label (case insensitive)
            if keyword.lower() in label_lower:
                return keyword  # Replace the label with the keyword
        
        return None  # Return None if no keywords match

    # Apply the replace_label function to each row in the 'labels' column
    dataframe['labels'] = dataframe['labels'].apply(lambda label: replace_label(label, keywords))

    # Remove rows where the label is None (i.e., no keyword matched)
    dataframe = dataframe.dropna(subset=['labels'])

    return dataframe