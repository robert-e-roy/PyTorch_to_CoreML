PyTorch to Core ML Converter (MiniLM-L6-v2)

This script provides a streamlined pipeline to convert the `sentence-transformers/all-MiniLM-L6-v2` model from PyTorch directly to a Core ML `.mlpackage`. It bypasses the ONNX intermediate format, leveraging the native `coremltools` PyTorch integration for better reliability and performance.

🚀 Features

• Direct Conversion: Uses `torch.jit.trace` for high-fidelity conversion.
• Optimized for Apple Silicon: Targets `mlprogram` with `FLOAT16` precision and `ComputeUnit.ALL` (CPU, GPU, and Neural Engine).
• Swift-Ready Vocab: Automatically exports a `vocab.txt` file sorted by token ID, enabling easy WordPiece tokenization implementation in Swift.
• Modern Target: Optimized for iOS 17+ and macOS 14+.

🛠 Prerequisites

System Requirements

• macOS (Apple Silicon recommended)
• Python 3.9+

Dependencies

Install the required libraries via pip: pip install torch coremltools transformers numpy

📖 Usage

Run the script from your terminal: python pytorch_to_coreml.py

Outputs

• `MiniLMEmbedding.mlpackage`: The compiled Core ML model.
• `vocab.txt`: The vocabulary file required for the tokenizer.

⚙ Integration Notes for Swift

The model outputs the last hidden state. To obtain the final sentence embedding (384-dimensional vector) in your iOS/macOS app, you must implement the following post-processing steps in Swift:

1. Mean Pooling: Average the output vectors based on the `attention_mask`.
2. L2 Normalization: Normalize the resulting vector to unit length.

📈 Performance

• Precision: FP16
• Max Sequence Length: 128 tokens
• Target Hardware: Apple Neural Engine (ANE) / GPU:
