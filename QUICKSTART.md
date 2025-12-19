# Quick Start Guide

This guide will help you set up and run the Automated Tablet Defect Detection System.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space
- (Optional) GPU with CUDA for faster training

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including PyTorch, Streamlit, and scikit-learn.

### 2. Verify Dataset

Ensure the capsule dataset is in the correct location:

```
capsule/
├── train/good/      # Should contain 219 images
├── test/            # Test images (good + defects)
└── ground_truth/    # Defect masks
```

### 3. Train the Model

```bash
python train.py
```

**What happens:**
- Loads 219 normal tablet images
- Extracts features using ResNet-18
- Fits PaDiM Gaussian distributions
- Saves model to `models/padim_model.pkl`

**Expected time:** 2-3 minutes on CPU

### 4. Evaluate Performance

```bash
python evaluate.py
```

**What happens:**
- Tests on all defect types
- Computes ROC-AUC, precision, recall
- Saves results to `results/`
- Generates ROC curve plot

### 5. Run Streamlit App

```bash
streamlit run app.py
```

**What happens:**
- Opens web interface at http://localhost:8501
- Upload images for real-time inspection
- View anomaly heatmaps
- Download results

### 6. (Optional) Test Single Image

```bash
python inference.py capsule/test/crack/000.png --threshold 0.5
```

## Troubleshooting

### "Model not found" error
- Make sure you ran `python train.py` first
- Check that `models/padim_model.pkl` exists

### "No images found" error
- Verify dataset is in `capsule/` directory
- Check folder structure matches expected layout

### Slow inference
- First run downloads ResNet weights (~45MB)
- Subsequent runs are faster
- Consider using GPU if available

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check Python version: `python --version` (should be 3.8+)

## Usage Examples

### Basic Inference
```bash
python inference.py path/to/image.png
```

### Custom Threshold
```bash
python inference.py path/to/image.png --threshold 0.3
```

### Streamlit with Custom Config
```bash
streamlit run app.py --server.port 8080
```

## Next Steps

1. **Fine-tune threshold** based on your quality requirements
2. **Deploy to cloud** (Streamlit Cloud, Render, HF Spaces)
3. **Integrate with production** pipeline via REST API
4. **Expand dataset** with your own tablet images

## Support

- Check `README.md` for detailed documentation
- Report issues on GitHub
- Contact: your.email@example.com

---

**Happy defect detection! 🔍**
