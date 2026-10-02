import pandas as pd
import torch
import os
import glob
import re
import tqdm
import numpy as np
import gc

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
    dataframes = [pd.read_csv(file) for file in tqdm.tqdm_notebook(csv_files)]
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    merged_df['index'] = range(len(merged_df))

    # Save the merged file
    merged_df.to_csv('merged.csv', index=False)
    print(f"Files have been merged and saved as merged.csv")

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


# Load all embeddings from the specified directory
def load_embeddings_from_directory(directory_path):

    embeddings = []
    files = os.listdir(directory_path)

    # Sort the files based on the number in the filename
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # List all files in the directory
    for file_name in tqdm.tqdm_notebook(files):
        print(file_name)
        file_path = os.path.join(directory_path, file_name)

        # Check if it's a file and ends with .pt
        if os.path.isfile(file_path) and file_path.endswith('.pt'):
            try:
                # Load the tensor from file
                embedding = torch.load(file_path)
                print(embedding.shape)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    return embeddings

# Load embedding and label from the specified directory using ID number
def load_embedding_from_directory(directory_path, id_number):

    # Build the search pattern
    pattern = os.path.join(directory_path, f"embedding_{id_number}.pt")

    # Find files matching the pattern
    matching_files = glob.glob(pattern)

    if matching_files:
        # Assuming there's only one matching file, load the first match
        file_to_load = matching_files[0]

        embedding = torch.load(file_to_load, map_location='cpu')
    else:
        print(f"No file found for ID {id_number}")

    return embedding

def prepare_dataframe(file_path, shouldBalance, should_merge_csv, columns_to_keep):
    if should_merge_csv:
        merge_csv_files_in_directory()
   
    df = pd.read_csv(file_path)

    # Remove duplicates
    df = df.drop_duplicates(keep='first')

    # Drop unnecessary columns
    columns_to_drop = [
        "pr_title", "class_names", "commit_sha", "commit_message", 
        "file_path", "additions", "deletions", "file_name", 
        "content_before", "content_after", "patch", 
        "created_at", "closed_at", "status"
    ]
    
    # Exclude columns to keep from the drop list
    columns_to_drop = [col for col in columns_to_drop if col not in columns_to_keep]
    
    df.drop(columns=columns_to_drop, inplace=True, axis=1)

    # Standardize labels
    labels = ['bug', 'enhancement', 'performance', 'refactor', 'clean']
    df = standardize_labels(df, labels)

    # Count the occurrences of each unique value in the 'labels' column
    label_counts = df['labels'].value_counts()
    print(label_counts)

    ### Limit df size for testing purposes ###
    label_enh = df[df['labels'] == 'enhancement']
    label_bug = df[df['labels'] == 'bug']
    label_cln = df[df['labels'] == 'clean']
    
    if shouldBalance:
        df = filterForEqualLabelCount(df)
    else:
        sampled_enh = label_enh
        sampled_bug = label_bug
        sampled_cln = label_cln

    df = pd.concat([sampled_enh, sampled_bug, sampled_cln])

    # Filter NaN rows based on the "patch" column (if it's kept)
    if 'patch' in df.columns:
        df = df.dropna(subset=['patch'])

    # Display DataFrame with a scrollable table (optional)
    # display(HTML(df.to_html(max_rows=None, max_cols=None)))

    # Count the occurrences of each unique value in the 'labels' column
    label_counts = df['labels'].value_counts()
    print(label_counts)

    return df


def load_embeddings_from_directory_for_df(directory_path, df):

    df['embedding'] = [None] * len(df)

    for index in tqdm.tqdm_notebook(df['index'].values):
        embedding = load_embedding_from_directory(directory_path=directory_path, id_number=index)
        embedding_flatten = torch.flatten(embedding)
        
        embedding.to('cpu')
        embedding_flatten.to('cpu')
        
        idx = df.loc[df['index'] == index].index
        df.at[idx[0], 'embedding']= embedding_flatten
        
    return df



def load_and_concat_embeddings_from_directories(directory_path_1, directory_path_2, df):
   
    # Initialize the 'embedding' column with None values
    df['embedding'] = [None] * len(df)
    
    for index in tqdm.tqdm_notebook(df['index'].values):
        # Load embeddings from the first directory
        embedding_1 = load_embedding_from_directory(directory_path=directory_path_1, id_number=index)
        embedding_1_flatten = torch.flatten(embedding_1)
        
        # Load embeddings from the second directory
        embedding_2 = load_embedding_from_directory(directory_path=directory_path_2, id_number=index)
        embedding_2_flatten = torch.flatten(embedding_2)
        
        # Move to CPU and concatenate embeddings
        embedding_1_flatten.to('cpu')
        embedding_2_flatten.to('cpu')
        concatenated_embedding = torch.cat((embedding_1_flatten, embedding_2_flatten), dim=0)
        
        # Assign concatenated embedding to the DataFrame
        idx = df.loc[df['index'] == index].index
        df.at[idx[0], 'embedding'] = concatenated_embedding
        
    return df


def filterRepoName(df,nameToFilter):
    df = df[df['full_repo_name'].str.contains(nameToFilter, case=False, na=False)]
    return df

def filterForEqualLabelCount(df):     
    
    labels = ['bug', 'enhancement', 'performance', 'refactor', 'clean']
    df = standardize_labels(df, labels)

    # Test için veri setini dengeleme
    label_enh = df[df['labels'] == 'enhancement']
    label_bug = df[df['labels'] == 'bug']
    label_cln = df[df['labels'] == 'clean']

    min_samples = min(len(label_enh), len(label_bug), len(label_cln))
    print (min_samples)
    sampled_enh = label_enh.sample(n=min_samples, random_state=1)
    sampled_bug = label_bug.sample(n=min_samples, random_state=1)
    sampled_cln = label_cln.sample(n=min_samples, random_state=1)
    df = pd.concat([sampled_enh, sampled_bug, sampled_cln])
    return df
