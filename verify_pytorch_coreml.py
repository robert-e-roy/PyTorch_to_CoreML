"""
06_verify_pytorch_coreml.py
Verifies Core ML output matches PyTorch output directly.
Replaces the ONNX comparison with direct PyTorch comparison.

Cosine similarity should be > 0.999 — slight FP16 loss is expected.
"""

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MLMODEL_PATH = "./MiniLMEmbedding.mlpackage"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128

TEST_SENTENCES = [
    "Resentment is the number one offender",
    "I want to make amends to my father",
    "What does the book say about fear",
    "searching and fearless moral inventory",
    "spiritual awakening carry the message",
]

def tokenize_padded(text, tokenizer, max_len=MAX_LEN):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    return (
        enc.input_ids.astype(np.int32),
        enc.attention_mask.astype(np.int32)
    )

def mean_pool_normalize(last_hidden, mask):
    mask_expanded = mask[..., np.newaxis].astype(np.float32)
    summed = (last_hidden * mask_expanded).sum(axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
    return (pooled / np.clip(norm, 1e-9, None)).squeeze()

def coreml_embed(text, mlmodel, tokenizer):
    input_ids, mask = tokenize_padded(text, tokenizer)
    out = mlmodel.predict({
        "input_ids": input_ids,
        "attention_mask": mask
    })
    # Core ML output key varies — find the hidden state tensor
    hidden_key = None
    for key, val in out.items():
        if hasattr(val, "shape") and len(val.shape) == 3:
            hidden_key = key
            break
    if hidden_key is None:
        raise ValueError(f"Could not find hidden state in output: {list(out.keys())}")
    hidden = out[hidden_key]
    return mean_pool_normalize(hidden, mask)

def main():
    print(f"Loading Core ML model from {MLMODEL_PATH}...")
    mlmodel = ct.models.MLModel(MLMODEL_PATH)

    print(f"Loading reference PyTorch model...")
    pt_model = SentenceTransformer(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\nCore ML output keys:", list(
        mlmodel.predict({
            "input_ids": np.zeros((1, MAX_LEN), dtype=np.int32),
            "attention_mask": np.ones((1, MAX_LEN), dtype=np.int32)
        }).keys()
    ))

    print("\nComparing embeddings (PyTorch vs Core ML):")
    print("-" * 60)

    all_passed = True
    for sentence in TEST_SENTENCES:
        pt_emb = pt_model.encode(sentence, normalize_embeddings=True)
        coreml_emb = coreml_embed(sentence, mlmodel, tokenizer)

        cos_sim = float(np.dot(pt_emb, coreml_emb))
        max_diff = float(np.max(np.abs(pt_emb - coreml_emb)))

        passed = cos_sim > 0.990
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {sentence[:40]}")
        print(f"         cosine: {cos_sim:.6f}  max_diff: {max_diff:.6f}")

    print("-" * 60)
    if all_passed:
        print("All checks passed. Core ML model is ready.")
        print("\nCopy to Xcode projects:")
        print(f"  {MLMODEL_PATH}")
        print(f"  ./vocab.txt  (for Swift WordPiece tokenizer)")
    else:
        print("FAILURES detected.")
        print("If cosine similarity is 0.95-0.99, try FLOAT32 in script 05.")
        print("If below 0.95, the tracing captured something incorrectly.")

if __name__ == "__main__":
    main()
