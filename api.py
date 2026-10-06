from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModel

# Initialize FastAPI server
app = FastAPI(
    title="Latent Space Diagnostic Engine API",
    description="Extracts 768-d embeddings from the 12th hidden layer ([CLS]) of BERT.",
    version="1.0.0"
)

# Load model and tokenizer (using the same model from your inference pipeline)
MODEL_NAME = "logasanjeev/bert-emotion-classifier"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# output_hidden_states=True is critical to extract the 12th layer representations
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
model.eval()

class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text for latent embedding extraction")

class EmbeddingResponse(BaseModel):
    status: str
    vector_dimension: int
    cls_token_preview: List[float]

@app.post("/v1/extract", response_model=EmbeddingResponse, status_code=status.HTTP_200_OK)
def extract_embedding(payload: TextPayload) -> EmbeddingResponse:
    cleaned_text = payload.text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    # Tokenize input restricting length to save memory, identical to batch logic
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    # Disable gradients to prevent RAM overflow
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the 12th hidden layer (index -1) and isolate the [CLS] token (index 0)
        cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze()

    vector_values = cls_embedding.cpu().tolist()

    return EmbeddingResponse(
        status="success",
        vector_dimension=len(vector_values),
        cls_token_preview=[round(val, 4) for val in vector_values[:5]]
    )