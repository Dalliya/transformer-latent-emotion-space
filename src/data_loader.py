import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_and_sample_data(sample_frac: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads GoEmotions and IMDB datasets from HuggingFace, processes multi-label 
    entries, and performs a memory-efficient stratified subsampling.
    
    Args:
        sample_frac: Fraction of the dataset to retain (default 10%).
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple containing downsampled GoEmotions and IMDB DataFrames.
    """
    # Load data using HuggingFace datasets (memory-mapped Arrow format)
    goemotions_ds = load_dataset("go_emotions", split="train")
    imdb_ds = load_dataset("imdb", split="train")
    
    # Convert to Pandas for easier tabular manipulation
    df_go = goemotions_ds.to_pandas()
    df_imdb = imdb_ds.to_pandas()

    # GoEmotions contains multiple labels per text. 
    # For a standard classifier evaluation, we extract the primary label.
    # If the labels list is empty, assign 27 (Neutral emotion class).
    df_go['label'] = df_go['labels'].apply(lambda x: x[0] if len(x) > 0 else 27)

    # Perform stratified sampling to maintain exact class proportions
    _, go_sample = train_test_split(
        df_go, 
        test_size=sample_frac, 
        stratify=df_go['label'], 
        random_state=random_state
    )
    
    _, imdb_sample = train_test_split(
        df_imdb, 
        test_size=sample_frac, 
        stratify=df_imdb['label'], 
        random_state=random_state
    )

    # Save to processed data folder (optional but recommended for caching)
    go_sample.to_csv("data/processed/goemotions_sampled.csv", index=False)
    imdb_sample.to_csv("data/processed/imdb_sampled.csv", index=False)

    return go_sample.reset_index(drop=True), imdb_sample.reset_index(drop=True)

if __name__ == "__main__":
    # Test execution
    print("Loading and stratifying datasets...")
    go_df, imdb_df = load_and_sample_data(sample_frac=0.1)
    
    print(f"GoEmotions sample shape: {go_df.shape}")
    print(f"IMDB sample shape: {imdb_df.shape}")
    
    # Verify class balance for GoEmotions
    print("\nGoEmotions Class Balance (Top 5):")
    print(go_df['label'].value_counts(normalize=True).head(5))