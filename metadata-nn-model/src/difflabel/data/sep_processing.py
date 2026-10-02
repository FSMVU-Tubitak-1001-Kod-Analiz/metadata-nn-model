"""Optional ``<sep>``-tagging of code columns via the external SepAdder JAR.

This depends on a ``sep`` package (``sep.sep_adder_file_reader.SepAdderFileReader``)
that is *not* part of this repository.  It is imported lazily so the rest of
the project works without it.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


def _get_reader_cls():
    try:
        from sep.sep_adder_file_reader import SepAdderFileReader
    except ImportError as exc:  # pragma: no cover - depends on external package
        raise ImportError(
            "SepAdderFileReader is required for <sep> tagging but the external `sep` package "
            "is not installed. Put it on PYTHONPATH or skip this step."
        ) from exc
    return SepAdderFileReader


def _process_single_item_with_jar(args):
    """Worker: builds the SepAdder input for one value, inside its own process.

    A fresh reader is created per call because handles to external processes
    generally can't be pickled across process boundaries.
    """
    path_to_jar, item_value = args
    try:
        reader = _get_reader_cls()(path_to_jar)
        if pd.notnull(item_value):
            return reader.build_input(item_value)
        return item_value
    except Exception as exc:
        print(f"Error processing item: {exc}")
        return item_value


def add_sep_to_df_multiprocess(df: pd.DataFrame, path_to_jar: str, max_workers: Optional[int] = None) -> pd.DataFrame:
    """Runs SepAdder over ``content_before`` / ``patch_before`` using a process pool."""
    columns = [c for c in ("content_before", "patch_before") if c in df.columns]
    if not columns:
        print("No relevant columns ('content_before' or 'patch_before') found in the dataframe.")
        return df

    max_workers = max_workers or multiprocessing.cpu_count()
    print(f"Using {max_workers} worker processes.")

    for column in columns:
        print(f"Processing column: {column}")
        items = [(idx, value) for idx, value in df[column].items() if pd.notnull(value)]
        if not items:
            print(f"No non-null data found in column '{column}'. Skipping.")
            continue

        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_process_single_item_with_jar, (path_to_jar, value)): idx for idx, value in items
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc=f"Processing '{column}'"
            ):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f'Row {idx} for column "{column}" generated an exception: {exc}')

        for idx, result in results.items():
            df.loc[idx, column] = result

    return df
