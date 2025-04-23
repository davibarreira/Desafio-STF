"""
This script processes the raw legal text dataset, cleans the text, and saves the processed data.
"""

import os
from pathlib import Path

import pandas as pd

from .cleaner_batch import clean_df_column


def process_dataset(
    input_path: str = "data/1_raw/dataset_desafio_ramo_direito (1).parquet",
    output_path: str = "data/2_pro/cleaned_dataset.parquet",
):
    """
    Process the dataset from input_path and save to output_path.

    Args:
        input_path: Path to the input parquet file
        output_path: Path to save the processed parquet file
    """
    # Ensure output directory exists
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Read the raw dataset
    print(f"Reading dataset from {input_path}")
    df = pd.read_parquet(input_path)
    text_column = "texto_bruto"

    # Clean the text column
    print(f"Cleaning text in column '{text_column}'")
    df_clean = clean_df_column(
        df,
        src_col=text_column,
        dest_col="clean_text",
        batch_size=10,  # Adjust based on available memory
        n_process=None,  # Will use max available cores - 1
    )

    # Keep only the required columns
    result_df = df_clean[["clean_text", "ramo_direito"]]

    # Save the processed dataset
    print(f"Saving processed dataset to {output_path}")
    result_df.to_parquet(output_path, index=False)
    print("Processing completed successfully!")


def main():
    """Main function that processes the dataset with default paths."""
    process_dataset()


if __name__ == "__main__":
    main()
