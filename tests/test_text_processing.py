import os
from pathlib import Path

import pandas as pd
import pytest

from app.text_processing.cleaner import clean_text, normalize_text
from app.text_processing.cleaner_batch import clean_df_column
from app.text_processing.text_processing import process_dataset


def test_normalize_text():
    """Test text normalization"""
    assert normalize_text("café!@#$%") == "cafe"
    assert normalize_text("  café  com  leite  ") == "cafe com leite"


def test_clean_text():
    """Test text cleaning"""
    assert clean_text("O café é bom") == "cafe"  # 'bom' is a stopword
    assert clean_text("O e a ou mas") == ""
    assert clean_text("Processo judicial importante") == "processo judicial importante"


def test_clean_df_column():
    """Test dataframe cleaning"""
    df = pd.DataFrame(
        {
            "texto_bruto": [
                "O café é bom",
                "O e a ou mas",
                "Processo judicial importante",
            ],
            "ramo_direito": ["civil", "civil", "civil"],
        }
    )

    df_clean = clean_df_column(df, batch_size=2, n_process=1)

    assert "texto_limpo" in df_clean.columns
    assert df_clean["texto_limpo"].iloc[0] == "cafe"
    assert df_clean["texto_limpo"].iloc[1] == ""
    assert df_clean["texto_limpo"].iloc[2] == "processo judicial importante"


def test_process_dataset(tmp_path):
    """Test the dataset processing function"""
    # Create test directories
    raw_dir = tmp_path / "data" / "1_raw"
    pro_dir = tmp_path / "data" / "2_pro"
    raw_dir.mkdir(parents=True)
    pro_dir.mkdir(parents=True)

    # Create a small test parquet file
    test_df = pd.DataFrame(
        {
            "texto_bruto": ["O café é bom", "Processo judicial importante"],
            "ramo_direito": ["civil", "penal"],
        }
    )
    input_path = raw_dir / "test_dataset.parquet"
    output_path = pro_dir / "cleaned_dataset.parquet"
    test_df.to_parquet(input_path)

    # Process the test dataset
    process_dataset(str(input_path), str(output_path))

    # Verify the output file exists and has the correct content
    assert output_path.exists()

    result_df = pd.read_parquet(output_path)
    assert "clean_text" in result_df.columns
    assert "ramo_direito" in result_df.columns
    assert len(result_df) == 2
    assert result_df["clean_text"].iloc[0] == "cafe"
    assert result_df["clean_text"].iloc[1] == "processo judicial importante"
