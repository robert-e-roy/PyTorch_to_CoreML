"""
05_pytorch_to_coreml.py
Direct PyTorch to Core ML conversion for MiniLM-L6-v2.
Skips the ONNX intermediate step entirely.

Output:./MiniLMEmbedding.mlpackage

coremltools 8 removed direct ONNX support — PyTorch is now
the primary conversion path. This is simpler and more reliable.
"""

import coremltools as ct
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging

# Configure logging for professional output tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COREML_OUTPUT = "./MiniLMEmbedding.mlpackage"
VOCAB_OUTPUT = "./vocab.txt"
MAX_SEQUENCE_LENGTH = 128

def save_vocabulary(tokenizer: AutoTokenizer, output_path: str) -> None:
    """
    Extracts the vocabulary from the tokenizer and saves it in token-id order.
    This is essential for implementing WordPiece tokenization on the Swift side.
    """
    logger.info("Saving vocabulary for Swift tokenizer...")
    vocab = tokenizer.get_vocab()
    # Sort by ID to ensure the index matches the token
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    with open(output_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(f"{token}\n")
    
    logger.info(f"Wrote {output_path} ({len(vocab)} tokens)")

def convert_model() -> None:
    """
    Handles the loading, tracing, and conversion of the PyTorch model to Core ML.
    """
    logger.info(f"Loading {MODEL_ID} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # torchscript=True ensures the model is compatible with torch.jit.trace
    model = AutoModel.from_pretrained(MODEL_ID, torchscript=True)
    model.eval()

    save_vocabulary(tokenizer, VOCAB_OUTPUT)

    # Trace the model with example inputs to define the computation graph
    logger.info("Tracing PyTorch model...")
    example_ids = torch.zeros(1, MAX_SEQUENCE_LENGTH, dtype=torch.long)
    example_mask = torch.ones(1, MAX_SEQUENCE_LENGTH, dtype=torch.long)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (example_ids, example_mask),
            strict=False
        )

    logger.info("Converting to Core ML (this may take a few minutes)...")
    
    # Convert using the mlprogram format for modern iOS/macOS targets
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, MAX_SEQUENCE_LENGTH),
                dtype=np.int32
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, MAX_SEQUENCE_LENGTH),
                dtype=np.int32
            ),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
    )

    # Metadata for model clarity in Xcode
    mlmodel.author = "MiniLM-L6-v2 via PyTorch -> Core ML"
    mlmodel.short_description = (
        "Sentence embedding model for RAG. "
        "Input: token ids and attention mask (max 128 tokens). "
        "Output: last hidden state — apply mean pooling and "
        "L2 norm in Swift to get final 384-dim embedding."
    )
    mlmodel.input_description["input_ids"] = "Token IDs from WordPiece tokenization, padded to 128"
    mlmodel.input_description["attention_mask"] = "1 for real tokens, 0 for padding"

    mlmodel.save(COREML_OUTPUT)
    
    # Calculate final package size
    total_size = sum(
        f.stat().st_size
        for f in Path(COREML_OUTPUT).rglob("*")
        if f.is_file()
    )
    
    logger.info(f"Successfully saved {COREML_OUTPUT} ({total_size / 1_048_576:.1f} MB)")
    logger.info(f"Deployment target: iOS 17 / macOS 14 | Precision: FP16")

def main():
    try:
        convert_model()
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()

