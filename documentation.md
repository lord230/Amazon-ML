# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Activations  
**Team Members:** Amit Verma, Abantika Biswas, Aishik Maiti, Yugal Tyagi  
**Submission Date:** October 13, 2025  

---

## 1. Executive Summary

Our solution presents a multi-modal deep learning framework designed to predict product prices by jointly analyzing textual product descriptions and visual image features.  
By fine-tuning pre-trained models for both modalities — MiniLM for text and EfficientNet-B0 for images — and combining their embeddings through a unified fusion network, the model effectively captures semantic, contextual, and visual attributes influencing market value.  

The final architecture was trained end-to-end with log-transformed targets and differential learning rates, achieving stable optimization and competitive performance.  

---

## 2. Methodology Overview

### 2.1 Problem Understanding
The challenge was framed as a regression task where the goal was to predict the continuous variable price from mixed-modality data.  

Observations and handling:  
- **Highly skewed price distribution** → Applied `log1p` transformation to stabilize learning and reduce outlier sensitivity.  
- **Multi-modal data** → Combined textual and visual features to capture comprehensive product information.  
- **Missing or invalid inputs** → Implemented fallbacks for missing text and invalid images during preprocessing.  

### 2.2 Core Strategy
We trained a **single, multi-modal neural network** that jointly learns from both text and image data.  

### Model Overview

| Component | Description |
|------------|--------------|
| Model Type | End-to-end Multi-modal Regression |
| Training Objective | Predict log-transformed price values |
| Key Techniques | Log-price training, mixed precision, discriminative learning rates, cosine warm restarts |


This setup yielded a well-regularized model that converged smoothly and generalized effectively.

---

## 3. Model Architecture

### 3.1 Architecture Summary
The model comprises three main modules:  

1. **Text Encoder** (MiniLM) – Extracts semantic embeddings from the concatenated title, description, and item quantity fields.  
2. **Image Encoder** (EfficientNet-B0) – Extracts visual embeddings from product images.  
3. **Fusion Head** – Projects both embeddings into a 256-dimensional space each, concatenates them, and passes through a **3-layer MLP** for regression.  

### 3.2 Detailed Configuration

### Model Specifications

| Module | Specification |
|---------|---------------|
| Text Encoder | "nreimers/MiniLM-L6-H384-uncased" |
| Text Embedding Size | 384 |
| Image Encoder | "EfficientNet-B0 (pretrained on IMAGENET1K_V1)" |
| Image Embedding Size | 1280 |
| Fusion Layers | [512 → 256 → 64 → 1] (GELU + BatchNorm) |
| Dropout | 0.25 |
| Loss Function | Smooth L1 Loss → Differentiable SMAPE (fine-tuning) |
| Learning Rate Scheduler | Cosine Annealing Warm Restarts |
| Training Regime | Mixed Precision (FP16) + Gradient Accumulation |


This configuration balances computational efficiency with expressive power, leveraging both modalities effectively.

---

## 4. Training Strategy

### 4.1 Training Phases
1. Phase 1 – Stable Fine-tuning:  
   - All layers unfrozen.  
   - Used SmoothL1Loss on log-transformed prices for stable gradient flow.  

2. Phase 2 – Metric-based Optimization: 
   - Switched to a differentiable **SMAPE loss** to directly optimize the leaderboard metric.  
   - Applied cosine restarts with short cycles to maintain active learning dynamics.  

3. **Optimization Details:**  
   - Optimizer: `AdamW`  
   - Batch size: Variable with gradient accumulation  
   - Differential LRs:  
     - Text encoder: 5e-6  
     - Image encoder: 7e-6  
     - Fusion layers: 1e-5  

---

## 5. Model Performance

### 5.1 Validation Results

| Metric |                         | Best Score |
|--------|                         |-------------|
| SMAPE (Validation)               | 20.68% |
| Training SMAPE (Final Epoch)     | 18.59% |
| Loss Function                    | Smooth L1(β = 0.5)|

These results were obtained after 20 epochs of full fine-tuning with cosine warm restarts.

---

## 6. Implementation Highlights

- Data Normalization: All text and image data were standardized to ensure consistent input scale.  
- Robustness Handling: Broken images and incomplete descriptions were automatically substituted with default placeholders.  
- **Reproducibility:** Fixed random seeds, deterministic settings, and consistent preprocessing pipelines ensured reproducible results.  
- **Performance Optimization:** Mixed precision training and discriminative learning rates significantly reduced training time while maintaining convergence stability.  

---

## 7. Conclusion

This project demonstrates that multi-modal deep learning effectively models price dynamics when structured and unstructured data are jointly leveraged.  
By combining **MiniLM** for textual understanding and **EfficientNet** for visual analysis within a unified regression framework, we achieved strong predictive performance without overfitting.  

Key insights: 
- Log-transforming target values improves convergence stability.  
- Differential learning rates enable efficient fine-tuning of large pre-trained encoders.  
- Mixed precision and cosine restarts accelerate training while preserving generalization.

---

## Appendix

### A. Code and Resources
Drive Link - https://drive.google.com/drive/folders/1Y5vBSuu23JifNhZWdhHO16feQQtgGZ6_?usp=sharing


