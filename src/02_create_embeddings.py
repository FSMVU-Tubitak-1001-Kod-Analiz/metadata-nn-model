"""
02_create_embeddings.py
========================
Converted from: embedding_creator_properCB.ipynb

Loads a prepared CSV (see 01_prepare_csv.py), runs a HuggingFace encoder
(default: microsoft/codebert-base) over one text column, and writes one
`.pt` tensor file per row to EMBEDDING_DIR, named `embedding_<index>.pt`.

These per-row `.pt` files are what `utils.load_embedding_from_directory` /
`load_embeddings_from_directory_for_df` / `load_and_concat_embeddings_from_directories`
read back in during training (03a / 03b / 04).

To embed a different column (e.g. `patch_before` instead of `content_before`),
or use CLS-token extraction instead of pooler output, just edit the CONFIG
block below and re-run -- the two "code" pipelines used by the training
scripts (content_before and patch_before) are each produced by one run of
this script with EMBEDDING_DIRECTORY / COLUMN changed accordingly.
"""

import os

import pandas as pd
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# ---------------------------------------------------------------------------
# Configuration -- edit these per embedding run
# ---------------------------------------------------------------------------
INPUT_CSV = "new_csv_files_for_rewrite/merged_2label_no_license_no_imports_balanced_14768.csv"
COLUMN = "content_before"  # e.g. "content_before" or "patch_before"
MODEL_NAME = "microsoft/codebert-base"

# Naming this directory with "CLS" in it switches EXTRACT_CLS on below --
# that's how the original notebook signaled custom (non-pooler) models.
EMBEDDING_DIRECTORY = "embeddings/CB_pooler_embeddings_content_before_2label_no_license_no_imports_balanced_14768"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512


def main():
    df = pd.read_csv(INPUT_CSV)

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    os.makedirs(EMBEDDING_DIRECTORY, exist_ok=True)

    index_column_name = "index"
    hidden_size = model.config.hidden_size
    extract_cls = "CLS" in EMBEDDING_DIRECTORY

    print(f"\nProcessing column '{COLUMN}' in batches of {BATCH_SIZE}...")
    if extract_cls:
        print("[INFO] Mode: Custom model. Extracting [CLS] token's last hidden state.")
    else:
        print("[INFO] Mode: Standard model. Extracting pooler output.")

    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm.tqdm(range(num_batches), desc=f"Generating Embeddings for '{COLUMN}'"):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, len(df))

        batch_df = df.iloc[start_index:end_index]
        current_batch_size = len(batch_df)

        texts_batch = batch_df[COLUMN].tolist()
        index_values_batch = batch_df[index_column_name].tolist()

        valid_texts_batch = []
        valid_indices_in_batch = []

        for j, text in enumerate(texts_batch):
            original_df_index = df.index[start_index + j]
            is_invalid = False
            reason = ""

            if pd.isna(text):
                is_invalid = True
                reason = "content is NaN/None"
            elif not isinstance(text, str) or not text.strip():
                is_invalid = True
                reason = "content is an empty or whitespace-only string"

            if is_invalid:
                print(f"\n[INFO] Skipping invalid row. DataFrame Index: {original_df_index}, Reason: {reason}")
            else:
                valid_texts_batch.append(text)
                valid_indices_in_batch.append(j)

        full_batch_embeddings = torch.full(
            (current_batch_size, hidden_size), float("nan"), dtype=torch.float32, device=DEVICE
        )

        if valid_texts_batch:
            try:
                tokens = tokenizer(
                    valid_texts_batch,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_SEQ_LENGTH,
                    return_tensors="pt",
                )
                tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = model(**tokens)

                    if extract_cls:
                        # Custom models without a pooler: use the [CLS] token's last hidden state.
                        last_hidden_states = outputs.last_hidden_state
                        sentence_embeddings_batch = last_hidden_states[:, 0, :]
                    else:
                        # Standard models (e.g. codebert-base, roberta-base): use the pooler output.
                        if isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions):
                            sentence_embeddings_batch = outputs.pooler_output
                        else:
                            last_hidden_state = outputs.last_hidden_state
                            sentence_embeddings_batch = model.pooler(last_hidden_state)

                full_batch_embeddings[valid_indices_in_batch, :] = sentence_embeddings_batch

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nCUDA Out of Memory on batch starting at index {start_index}. "
                          f"Sub-batch size: {len(valid_texts_batch)}.")
                    print("Consider reducing BATCH_SIZE or MAX_SEQ_LENGTH.")
                else:
                    print(f"\nRuntime Error on batch starting at index {start_index}: {e}")
            except Exception as e:
                print(f"\nAn error occurred on batch starting at index {start_index}: {e}")

        for j in range(current_batch_size):
            single_embedding_vector = full_batch_embeddings[j, :]
            index_value_for_filename = index_values_batch[j]
            save_path = os.path.join(EMBEDDING_DIRECTORY, f"embedding_{index_value_for_filename}.pt")
            torch.save(single_embedding_vector.cpu(), save_path)

    print("\nEmbedding generation finished.")


if __name__ == "__main__":
    main()
