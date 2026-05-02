import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import List

def run_inference(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Runs batch inference using BERT on Apple Silicon (MPS) or CPU.
    
    Args:
        texts: List of text strings to classify.
        batch_size: Number of texts per batch (keep low for 8GB RAM).
        
    Returns:
        NumPy array of predicted class indices.
    """
    # Hardware acceleration for Apple M-series chips
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "logasanjeev/bert-emotion-classifier"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    
    # Set model to evaluation mode (disables dropout, saves memory)
    model.eval()

    predictions = []

    # Process in batches to prevent OutOfMemory errors
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting Batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize texts
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, # Restrict length to save memory
            return_tensors="pt"
        ).to(device)

        with torch.no_grad(): # Disable gradient calculation
            outputs = model(**inputs)
            logits = outputs.logits
            # Get the index of the highest probability
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)

    return np.array(predictions)

if __name__ == "__main__":
    # 1. Load the processed data
    print("Loading processed datasets...")
    go_df = pd.read_csv("data/processed/goemotions_sampled.csv")
    imdb_df = pd.read_csv("data/processed/imdb_sampled.csv")

    # 2. Run inference on GoEmotions
    print("\n--- Running Inference on GoEmotions ---")
    go_texts = go_df['text'].tolist()
    go_df['predicted_label'] = run_inference(go_texts, batch_size=32)

    # 3. Evaluate reproducibility (GoEmotions)
    print("\nModel Evaluation on GoEmotions Subset:")
    # We use zero_division=0 to suppress warnings if some classes were not predicted
    report = classification_report(go_df['label'], go_df['predicted_label'], zero_division=0)
    print(report)

    # 4. Run inference on IMDB
    print("\n--- Running Inference on IMDB ---")
    imdb_texts = imdb_df['text'].tolist()
    # The model predicts 28 classes for IMDB texts as well
    imdb_df['predicted_28_classes'] = run_inference(imdb_texts, batch_size=32)

    # 5. Save the results
    go_df.to_csv("data/processed/goemotions_predictions.csv", index=False)
    imdb_df.to_csv("data/processed/imdb_predictions.csv", index=False)
    print("\nPredictions saved to data/processed/")