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

def prepare_dataframe(file_path):
    df = pd.read_csv(file_path)
    # df = merge_csv_files_in_directory()

    #remove duplicates
    df = df.drop_duplicates(keep='first')

    df.drop("pr_title",inplace = True,axis=1)
    df.drop("class_names",inplace = True,axis=1)
    df.drop("commit_sha",inplace = True,axis=1)
    df.drop("commit_message",inplace = True,axis=1)
    df.drop("file_path",inplace = True,axis=1)
    df.drop("additions",inplace = True,axis=1)
    df.drop("deletions",inplace = True,axis=1)
    df.drop("file_name",inplace = True,axis=1)
    # df.drop("full_repo_name",inplace = True,axis=1)

    df.drop("content_before",inplace = True,axis=1)
    df.drop("content_after",inplace = True,axis=1)

    # df.drop("patch",inplace = True,axis=1)

    df.drop("created_at",inplace = True,axis=1)
    df.drop("closed_at",inplace = True,axis=1)
    df.drop("status",inplace = True,axis=1)

    # Standardize labels
    labels = ['bug', 'enhancement', 'performance', 'refactor', 'clean']
    df = standardize_labels(df, labels)

    # Count the occurrences of each unique value in the 'lables' column
    label_counts = df['labels'].value_counts()
    print(label_counts)

    ### Limit df size for testing purposes ###
    label_enh = df[df['labels'] == 'enhancement']
    label_bug = df[df['labels'] == 'bug']
    label_cln = df[df['labels'] == 'clean']
    # label_prf = df[df['labels'] == 'performance']
    # label_ref = df[df['labels'] == 'refactor']
    sampled_enh = label_enh.sample(n=15300, random_state=1)
    sampled_bug = label_bug.sample(n=15300, random_state=1)
    sampled_cln = label_cln.sample(n=15300, random_state=1)
    # sampled_prf = label_prf.sample(n=3200, random_state=1)
    # sampled_ref = label_ref.sample(n=3200, random_state=1)
    df = pd.concat([sampled_enh, sampled_bug, sampled_cln])

    # # Filter NaN rows
    # df = df.dropna(subset=['content_before', 'content_after'])
    df = df.dropna(subset=['patch'])

    # Display DataFrame with a scrollable table
    #display(HTML(df.to_html(max_rows=None, max_cols=None)))

    # Count the occurrences of each unique value in the 'lables' column
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

