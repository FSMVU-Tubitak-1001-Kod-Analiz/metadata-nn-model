import pandas as pd
import os
import glob

def merge_csv_files_in_directory(directory="inputs"):
    # Get the absolute path of the directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(script_dir, directory)
    
    # Initialize a list to hold all found CSV files
    csv_files = []
    
    # Recursively search for all CSV files in the specified directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        csv_files.extend(glob.glob(os.path.join(root, "*.csv")))

    if not csv_files:
        print("No CSV files found in the directory or its subdirectories!")
        return

    # Read all CSV files and merge them
    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged file
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

# Does string processing on a patch value and turns it into before and after code
def extract_code_from_patch(patch):
    before_lines = []
    after_lines = []

    for line in patch.splitlines():
        if line.startswith('@@'):
            # Split the line at the second '@@' and keep only the code part
            code_part = line.split('@@', 2)[-1].strip()
            if code_part:
                before_lines.append(code_part)
                after_lines.append(code_part)
        elif line.startswith('-'):
            # Line removed in the after version, add to before version
            before_lines.append(line[1:])
        elif line.startswith('+'):
            # Line added in the after version, add to after version
            after_lines.append(line[1:])
        else:
            # Line unchanged, add to both versions
            before_lines.append(line)
            after_lines.append(line)

    before_code = '\n'.join(before_lines)
    after_code = '\n'.join(after_lines)

    return before_code, after_code