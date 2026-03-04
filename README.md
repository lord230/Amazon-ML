# Amazon ML Challenge 2025: Smart Product Pricing Solution

An end-to-end multi-modal deep learning framework designed to predict product prices from textual descriptions and images. This project achieves robust price prediction by fusing semantic text embeddings from a Transformer with visual features from a Convolutional Neural Network.

## Architecture

The model (`MiniLMEfficientNetModel`) uses a dual-branch architecture:
1. **Text Branch:** `MiniLM-L6-H384-uncased` encodes the product's catalog content (title, description).
2. **Image Branch:** `EfficientNet-B0` (pretrained on ImageNet) extracts visual features from the product images.
3. **Fusion Head:** Both embeddings are projected to 256 dimensions, concatenated, and passed through a 3-layer Multilayer Perceptron (MLP) with ReLU/GELU activation and Batch Normalization.

## Features

- **Multi-modal Fusion:** Leverages both text and image data for accurate price prediction.
- **Log-Price Target:** Predicts the `log1p` of the price to handle the highly skewed price distribution natively.
- **Mixed Precision Training:** Uses FP16 AutoCast for fast and memory-efficient training.
- **Differential Learning Rates:** Fine-tuned encoder pathways with separate learning rates for the text model, image model, and the fusion head.
- **On-the-fly Image Download:** Missing images are smoothly downloaded and processed during dataset instantiation dynamically.

## File Structure

- `model.py`: Defines the `MiniLMEfficientNetModel` PyTorch architecture.
- `dataset.py`: Contains the `ProductPriceDataset` class for handling data loading, tokenizer integration, image transforms, and on-the-fly downloading.
- `train.py`: The complete training loop featuring gradient accumulation, `AdamW`, `CosineAnnealingWarmRestarts` scheduler, and SMAPE/SmoothL1 loss tracking.
- `test_m.py`: Inference script to generate price predictions on test CSV datasets.
- `documentation.md`: Detailed documentation explaining methodology and results from the team.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (CUDA supported)
- Transformers (`huggingface`)
- torchvision, pandas, numpy, Pillow, requests, scikit-learn, tqdm

### Setup & Directory Structure
Ensure your training data `train.csv` is placed in the `data/` folder and test data `test.csv` in `dataset/`:
```text
Amazon-ML/
├── data/
│   ├── train.csv
│   └── images/      # Images will be downloaded here automatically
├── dataset/
│   ├── test.csv     # Test data for inference
│   └── test_out_1.csv # Generated predictions
```

### Training
To train the model from scratch or resume from a checkpoint:
```bash
python train.py
```
Checkpoints will be saved in the `checkpoints_minilm_effnet/` directory, named with the best validation SMAPE score (e.g., `best_fullunfreeze_20.68.pt`).

### Inference
To run predictions on your test dataset:
```bash
python test_m.py
```
This will output `dataset/test_out_1.csv` containing the `sample_id` and the `predicted_price`.

## Performance
- **Loss Function:** Smooth L1 Loss (with Differentiable SMAPE optimization)
- **Validation SMAPE:** 20.68%
- **Training SMAPE:** 18.59%
