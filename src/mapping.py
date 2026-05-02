import pandas as pd
import time
from functools import wraps

# 1. Profiling Decorator
def timeit(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[{func.__name__}] Execution time: {execution_time:.6f} seconds")
        return result
    return timeit_wrapper

# 2. Define the Mapping Strategy (28 classes to 2 classes)
# IMDB labels: 1 = Positive, 0 = Negative
# GoEmotions mapping logic:
# Positive emotions -> 1
# Negative & Neutral/Ambiguous emotions -> 0
EMOTION_TO_POLARITY = {
    0: 1,   # admiration -> Positive
    1: 1,   # amusement -> Positive
    2: 0,   # anger -> Negative
    3: 0,   # annoyance -> Negative
    4: 1,   # approval -> Positive
    5: 0,   # confusion -> Negative (Ambiguous)
    6: 1,   # curiosity -> Positive
    7: 1,   # caring -> Positive
    8: 1,   # desire -> Positive
    9: 0,   # disappointment -> Negative
    10: 0,  # disapproval -> Negative
    11: 1,  # excitement -> Positive
    12: 0,  # disgust -> Negative
    13: 0,  # embarrassment -> Negative
    14: 0,  # fear -> Negative
    15: 1,  # gratitude -> Positive
    16: 0,  # grief -> Negative
    17: 1,  # joy -> Positive
    18: 1,  # love -> Positive
    19: 0,  # nervousness -> Negative
    20: 1,  # optimism -> Positive
    21: 1,  # pride -> Positive
    22: 0,  # remorse -> Negative
    23: 1,  # relief -> Positive
    24: 0,  # realization -> Negative (Ambiguous)
    25: 0,  # sadness -> Negative
    26: 1,  # surprise -> Positive
    27: 0   # neutral -> Negative (Baseline)
}

# 3. Mapping Methods
@timeit
def map_with_loop(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Maps values using a standard Python for-loop (Inefficient)."""
    mapped_values = []
    for val in df[source_col]:
        mapped_values.append(EMOTION_TO_POLARITY.get(val, 0))
    df[target_col] = mapped_values
    return df

@timeit
def map_with_apply(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Maps values using pandas .apply() method (Optimized)."""
    df[target_col] = df[source_col].apply(lambda x: EMOTION_TO_POLARITY.get(x, 0))
    return df

if __name__ == "__main__":
    print("Loading IMDB predictions...")
    imdb_df = pd.read_csv("data/processed/imdb_predictions.csv")

    print("\n--- Performance Profiling ---")
    
    # Test Loop Performance
    imdb_df = map_with_loop(imdb_df.copy(), 'predicted_28_classes', 'mapped_label_loop')
    
    # Test Apply Performance
    imdb_df = map_with_apply(imdb_df.copy(), 'predicted_28_classes', 'mapped_label_apply')

    # Assign final mapped column
    imdb_df['predicted_label'] = imdb_df['mapped_label_apply']
    
    # Drop temporary columns
    imdb_df.drop(columns=['mapped_label_loop', 'mapped_label_apply'], inplace=True)

    # Save transformed data
    imdb_df.to_csv("data/processed/imdb_mapped.csv", index=False)
    print("\nMapping complete. Transformed IMDB dataset saved to data/processed/imdb_mapped.csv")

    # Display correlation between Ground Truth and our mapped predictions
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(imdb_df['label'], imdb_df['predicted_label'])
    print(f"\nZero-Shot Transfer Accuracy (28->2 Mapping) vs IMDB Ground Truth: {acc:.2%}")