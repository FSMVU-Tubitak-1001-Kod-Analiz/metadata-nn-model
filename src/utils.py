import pandas as pd
import torch
import os
import glob
import re
import tqdm
import numpy as np
import gc
from sep.sep_adder_file_reader import SepAdderFileReader
import multiprocessing
import concurrent.futures

def _process_single_item_with_jar(args):
    """Helper function to process a single item using SepAdderFileReader in a separate process."""
    path_to_jar, item_value = args # Receive path and value from the task tuple

    # --- CRITICAL ---
    # Create a NEW instance of SepAdderFileReader IN THIS PROCESS.
    # Do NOT try to pass the SepAdderFileReader instance created in the main process.
    # Resource-holding objects (like file handles, process connections) are often not picklable.
    try:
        sep_adder = SepAdderFileReader(path_to_jar)
        # Handle potential nulls just in case, although filtered earlier
        if pd.notnull(item_value):
             return sep_adder.build_input(item_value)
        return item_value # Return as is if null
    except Exception as e:
        # Log the error or handle it as needed. Returning None or the original value might be options.
        print(f"Error processing item: {e}")
        return item_value # Return original value or None on error


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
    # 1. Create a mapping of lowercase keywords to the EXACT original keyword
    # Example: {'bug': 'bug', 'enhancement': 'enhancement'}
    keyword_map = {k.lower(): k for k in keywords}
    
    # 2. Create a regex pattern: "bug|enhancement|performance|refactor|clean"
    pattern = '|'.join(map(re.escape, keyword_map.keys()))
    
    # 3. Extract ONLY the matched word (case-insensitive)
    # ">Bug" -> "Bug"
    # "performance-issue" -> "performance"
    extracted_labels = dataframe['labels'].str.extract(f'(?i)({pattern})')[0]
    
    # 4. Lowercase the match, then map it to your exact original keyword list
    dataframe['labels'] = extracted_labels.str.lower().map(keyword_map)
    
    # 5. Drop rows where nothing matched
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

    return before_code#, after_code


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
        "pr_title", "class_names", "commit_sha", 
        "file_path", "additions", "deletions", "file_name", 
        "content_before", "content_after",
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
    #label_cln = df[df['labels'] == 'clean']

    #label_ref = df[df['labels'] == 'refactor']
    #label_per = df[df['labels'] == 'performance']
    
    if shouldBalance:
        df = filterForEqualLabelCount(df)
    else:
        sampled_enh = label_enh
        sampled_bug = label_bug
        #sampled_cln = label_cln

        #sampled_ref = label_ref
        #sampled_per = label_per

    df = pd.concat([sampled_enh, sampled_bug])

    # Filter NaN rows based on the "patch" column (if it's kept)
    if 'patch' in df.columns:
        df = df.dropna(subset=['patch'])

    # Display DataFrame with a scrollable table (optional)
    # display(HTML(df.to_html(max_rows=None, max_cols=None)))

    # Count the occurrences of each unique value in the 'labels' column
    label_counts = df['labels'].value_counts()
    # print(label_counts)

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
    
    labels = ['bug', 'enhancement']#, 'performance', 'refactor', 'clean']
    df = standardize_labels(df, labels)
    print(f"After standardization, unique labels: {df['labels'].unique()}")
    # Test için veri setini dengeleme
    label_enh = df[df['labels'] == 'enhancement']
    label_bug = df[df['labels'] == 'bug']
    #label_cln = df[df['labels'] == 'clean']

    min_samples = min(len(label_enh), len(label_bug))#, len(label_cln))
    print (min_samples)
    sampled_enh = label_enh.sample(n=min_samples, random_state=1)
    sampled_bug = label_bug.sample(n=min_samples, random_state=1)
    #sampled_cln = label_cln.sample(n=min_samples, random_state=1)
    df = pd.concat([sampled_enh, sampled_bug])#, sampled_cln])
    return df

def addSepToDf_multiprocess(df, path_to_jar, max_workers=None):
    """
    Processes a dataframe using SepAdderFileReader with multiprocessing.

    Args:
        df (pd.DataFrame): The dataframe to process.
        path_to_jar (str): Path to the SepAdder JAR file.
        max_workers (int, optional): Maximum number of worker processes.
                                     Defaults to the number of CPU cores.

    Returns:
        pd.DataFrame: The modified dataframe with separators added.
    """
    # Check for the presence of the relevant columns
    columns_to_process = []
    if 'content_before' in df.columns:
        columns_to_process.append('content_before')
    if 'patch_before' in df.columns:
        columns_to_process.append('patch_before')

    # If no relevant columns are found, return the dataframe as is
    if not columns_to_process:
        print("No relevant columns ('content_before' or 'patch_before') found in the dataframe.")
        return df

    # Determine the number of worker processes
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    print(f"Using {max_workers} worker processes.")


    for column in columns_to_process:
        print(f"Processing column: {column}")

        # Prepare tasks for the pool. Each task is a tuple (path_to_jar, item_value)
        # We also keep track of the original DataFrame index
        tasks_with_indices = [(index, value)
                              for index, value in df[column].items()
                              if pd.notnull(value)] # Filter out nulls before submitting

        # If no non-null data in this column, skip
        if not tasks_with_indices:
             print(f"No non-null data found in column '{column}'. Skipping.")
             continue

        # Create a list of arguments for the worker function: [(path_to_jar, value), ...]
        task_args = [(path_to_jar, value) for index, value in tasks_with_indices]
        original_indices = [index for index, value in tasks_with_indices]

        results = {} # Store results indexed by original DataFrame index

        # Use ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and map futures back to original indices
            future_to_index = {
                executor.submit(_process_single_item_with_jar, task_args[i]): original_indices[i]
                for i in range(len(task_args))
            }

            # Collect results as they complete, using tqdm for progress
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc=f"Processing '{column}'"):
                original_index = future_to_index[future]
                try:
                    # Get the result from the future
                    result = future.result()
                    results[original_index] = result
                except Exception as exc:
                    # Handle exceptions from the worker process
                    print(f'Row {original_index} for column "{column}" generated an exception: {exc}')
                    # Optionally: Decide what to put in the DataFrame if an error occurs
                    # For now, we just print the error and don't add this index to 'results',
                    # meaning the original value in the DataFrame remains.

        # Update the DataFrame with the results gathered
        for index, result in results.items():
            df.loc[index, column] = result # Use .loc for index-based assignment

    return df


def add_sep_to_subset(full_code_with_sep: str, subset_code_without_sep: str) -> str:
    """
    Adds <sep> tags to a subset of code based on a fully tagged version.

    This function works by creating a map of which code lines from the full version
    have a <sep> tag. It then iterates through the subset, and for each line,
    it checks the map to see if a tag should be appended.

    This approach is robust against differences in indentation and works even if
    the subset contains non-contiguous lines from the original.

    Args:
        full_code_with_sep: A string containing the full Java class, where each
                            statement/line is followed by a <sep> tag.
        subset_code_without_sep: A string containing a subset of the Java class
                                 code, without any <sep> tags.

    Returns:
        A string containing the subset code with <sep> tags added in the
        appropriate places.
    """
    
    # 1. Create a mapping from each clean code line to whether it has a <sep> tag.
    # We use a dictionary for fast lookups. The key will be the line of code
    # with leading/trailing whitespace removed, and the value will be True.
    sep_map = {}
    
    # .splitlines() is used to handle different line endings correctly (e.g., \n, \r\n)
    for line in full_code_with_sep.splitlines():
        if '<sep>' in line:
            # Get the pure code by removing the tag and stripping whitespace
            clean_line = line.replace('<sep>', '').strip()
            # If the line is not empty, add it to our map
            if clean_line:
                sep_map[clean_line] = True

    # 2. Reconstruct the subset by iterating through its lines and adding tags
    #    where appropriate based on our map.
    output_lines = []
    for line in subset_code_without_sep.splitlines():
        # Strip the line to create a key for looking up in our map
        lookup_key = line.strip()
        
        # Check if the stripped version of this line exists in our map.
        # .get() is used to safely handle lines that might not be in the map.
        if sep_map.get(lookup_key):
            # If it exists, append the original line (to preserve indentation)
            # followed by the tag.
            output_lines.append("<sep>" + line.strip())
        else:
            # Otherwise, just append the original line as is.
            output_lines.append(line.strip())
            
    # 3. Join the processed lines back into a single string.
    return "\n".join(output_lines)


def clean_java_code(java_code: str) -> str:
    """
    Removes comments and import statements from a string of Java code.

    Args:
        java_code (str): The Java code as a single string.

    Returns:
        str: The cleaned Java code.
    """
    if not isinstance(java_code, str):
        return "" # Return empty string for non-string inputs (like empty cells)

    # 1. Remove multi-line comments (/* ... */) and Javadoc (/** ... */)
    # The re.DOTALL flag allows '.' to match newlines
    # cleaned_code = re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)
    cleaned_code = re.sub(r'/\*.*?\*/', '', java_code, count=1, flags=re.DOTALL) # Limit to license comments only

    # 2. Remove single-line comments (// ...)
    # cleaned_code = re.sub(r'//.*', '', cleaned_code)

    # 3. Remove import statements
    # The re.MULTILINE flag allows '^' to match the start of each line
    cleaned_code = re.sub(r'^\s*import\s+.*;', '', cleaned_code, flags=re.MULTILINE)

    # 4. Remove package statements
    cleaned_code = re.sub(r'^\s*package\s+.*;', '', cleaned_code, flags=re.MULTILINE)

    # 5. Remove excessive blank lines resulting from the removals
    # This collapses multiple newlines into a single one for cleaner output
    cleaned_code = re.sub(r'\n\s*\n', '\n', cleaned_code)
    
    # 6. Remove leading/trailing whitespace
    cleaned_code = cleaned_code.strip()

    return cleaned_code


    # ADD SEP WITH STRING PROCESSING ONLY (for functions)

def add_sep_to_java_statements(java_function_body: str) -> str:
    """
    Inserts a '<sep>' tag after each complete statement in a Java function body.

    This function uses heuristics to identify statements, focusing on semicolons
    and closing braces that are not part of a control structure's header.

    Args:
        java_function_body: A string containing the body of a Java function.

    Returns:
        A new string with '<sep>' tags inserted after each statement.
    """
    # 1. First, remove all block comments to simplify processing.
    code = re.sub(r'/\*.*?\*/', '', java_function_body, flags=re.DOTALL)

    lines = code.split('\n')
    processed_code = []
    
    # Regex to identify lines that start control structures (if, for, while, etc.)
    # These lines should not have a <sep> added even if they end with ')'
    control_flow_regex = re.compile(
        r'^\s*(if|for|while|switch|try|catch|finally)\s*\(.*\)\s*\{?$'
    )

    in_control_flow_header = False
    paren_depth = 0

    for line in lines:
        # Strip leading/trailing whitespace to analyze the line's structure
        stripped_line = line.strip()

        # Ignore empty lines
        if not stripped_line:
            processed_code.append(line)
            continue
        
        # Remove any line comments for analysis
        line_without_comments = re.sub(r'//.*$', '', line)
        stripped_line_without_comments = line_without_comments.strip()

        # This logic handles multi-line for/if/while headers.
        # It waits until the parentheses are fully closed.
        if control_flow_regex.match(stripped_line_without_comments) or in_control_flow_header:
            in_control_flow_header = True
            paren_depth += line_without_comments.count('(')
            paren_depth -= line_without_comments.count(')')
            if paren_depth <= 0:
                in_control_flow_header = False
                paren_depth = 0 # Reset
            
            # Add the line and move on, as it's part of a header.
            processed_code.append(line)
            continue

        # If a line ends with a semicolon, it's a statement.
        if stripped_line_without_comments.endswith(';'):
            processed_code.append(line + ' <sep>')
        # If a line ends with a closing brace, it terminates a block statement.
        elif stripped_line_without_comments.endswith('}'):
            processed_code.append(line + ' <sep>')
        # Handle `};` for do-while loops
        elif stripped_line_without_comments.endswith('};'):
            processed_code.append(line + ' <sep>')
        else:
            # It's a line that doesn't end a statement (e.g., part of a long expression)
            processed_code.append(line)

    return '\n'.join(processed_code)