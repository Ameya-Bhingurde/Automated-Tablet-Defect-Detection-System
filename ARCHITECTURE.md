# 🏗️ System Architecture & Technical Details

## Overview

This document explains the technical architecture, model training/inference pipeline, and deployment strategy for the **Automated Tablet Defect Detection System**.

---

## 🎯 Architecture Type

### **Current: Streamlit Web Application**

- **Framework**: Streamlit (Python-based web framework)
- **Type**: Monolithic web app (Frontend + Backend combined)
- **No REST API**: Direct Python function calls
- **Deployment**: Single web service

### **Why Streamlit vs FastAPI?**

| Aspect | Streamlit (Current) | FastAPI (Alternative) |
|--------|-------------------|---------------------|
| **Use Case** | Interactive demo, portfolio | Production API, mobile apps |
| **Setup** | Simple (1 file = UI + logic) | Complex (separate frontend) |
| **UI** | Built-in components | Requires separate React/Vue |
| **API Endpoints** | No | Yes (`/predict`, `/health`) |
| **Best For** | Demos, prototypes, dashboards | Microservices, mobile backends |
| **Learning Curve** | Easy | Moderate |

**Your choice**: Streamlit is perfect for a portfolio demo!

---

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER ACTIONS                     │
│                                                     │
│  1. Uploads Image  →  2. Clicks Predict            │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND (app.py)            │
│  • File uploader component                          │
│  • Display controls (sliders, buttons)              │
│  • Result visualization                             │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           PREPROCESSING (app.py)                    │
│  • preprocess_image()                               │
│  • Resize to 224×224                                │
│  • Normalize (ImageNet stats)                       │
│  • Convert to tensor                                │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│       FEATURE EXTRACTION (src/feature_extractor.py) │
│  • ResNet-18 backbone (pretrained)                  │
│  • Extract from layers: layer1, layer2, layer3      │
│  • Multi-scale embeddings: [B, 448, 56, 56]        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│         PADIM MODEL (src/padim.py)                  │
│  • Compute Mahalanobis distance                     │
│  • Generate anomaly map [H, W]                      │
│  • Calculate image-level anomaly score              │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│       VISUALIZATION (src/visualize.py)              │
│  • Apply colormap to anomaly map                    │
│  • Overlay heatmap on original image                │
│  • Create defect localization view                  │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│          STREAMLIT FRONTEND (Display)               │
│  • Show prediction (Normal/Defective)               │
│  • Display anomaly score & confidence               │
│  • Render heatmap overlay                           │
│  • Provide download button                          │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 Model Training Pipeline

### **When Does Training Happen?**

1. **Local Development**: Run `python train.py` manually
2. **Render Deployment**: Automatically on first deployment (model not in Git)

### **Training Process (train.py)**

```python
def train_padim():
    """
    1. Load Data
       - Read 219 normal images from capsule/train/good/
       - Each image: 224×224 RGB
    
    2. Extract Features
       - Use ResNet-18 (pretrained on ImageNet)
       - Extract from 3 layers: layer1, layer2, layer3
       - Concatenate → 448-dim embedding per pixel
       - Reduce to 100-dim using random projection
    
    3. Fit Gaussian Distribution
       - For each of 56×56 spatial locations:
         • Compute mean μ ∈ ℝ^100
         • Compute covariance Σ ∈ ℝ^(100×100)
       - Model: N(μ, Σ) per location
    
    4. Save Model
       - Serialize to models/padim_model.pkl (~252MB)
       - Contains: means, covariances, random projection matrix
    """
```

**Time**: ~2-3 minutes on CPU

**Memory**: ~1GB during training, 252MB saved model

---

## 🔍 Inference Pipeline

### **When Does Inference Happen?**

Every time a user uploads an image in the Streamlit app.

### **Inference Process (app.py)**

```python
def predict_defect(image, padim_model, extractor, device):
    """
    1. Preprocess Image
       - Resize to 224×224
       - Normalize with ImageNet stats
       - Convert to tensor [1, 3, 224, 224]
    
    2. Extract Embeddings
       - Pass through ResNet-18
       - Get multi-scale features [1, 448, 56, 56]
       - Reduce to [1, 100, 56, 56]
    
    3. Compute Anomaly Scores
       - For each of 56×56 locations:
         • Calculate Mahalanobis distance:
           M(x) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]
       - Apply Gaussian smoothing (σ=4)
       - Anomaly map: [56, 56]
    
    4. Generate Results
       - Image score = max(anomaly_map)
       - Heatmap = resize(anomaly_map, 224×224)
       - Decision = score > threshold
    
    Return: (anomaly_score, anomaly_map)
    """
```

**Time**: <0.5s per image on CPU

**Memory**: ~500MB during inference

---

## 💾 Model Persistence

### **Model Storage**

```
models/
└── padim_model.pkl  (252MB)
    ├── means: np.array [56, 56, 100]
    ├── covariances: np.array [56, 56, 100, 100]
    └── projection: scipy.sparse matrix [448 → 100]
```

### **Why Not in Git?**

- ❌ **Too large**: 252MB exceeds GitHub's 100MB file size limit
- ❌ **Binary file**: Not diff-friendly
- ✅ **Solution**: Train automatically on first deployment

### **Loading Strategy**

```python
@st.cache_resource  # ← Streamlit caching decorator
def load_model():
    """
    - Loads once per Streamlit session
    - Cached in memory
    - Shared across all users
    - GPU/CPU auto-detection
    """
```

**Caching Benefits:**
- ✅ Load once, use many times
- ✅ Fast subsequent predictions
- ✅ Shared across concurrent users

---

## 🚀 Deployment Architecture (Render)

### **Deployment Flow**

```
GitHub Push
    │
    ▼
Render Auto-Deploy Triggered
    │
    ├─→ 1. Clone repo
    │
    ├─→ 2. Install dependencies (requirements.txt)
    │      • torch (CPU-only)
    │      • torchvision
    │      • streamlit
    │      • opencv-python-headless
    │
    ├─→ 3. Start Streamlit app
    │      • streamlit run app.py --server.port=10000
    │
    ├─→ 4. Auto-train model (first time only)
    │      • Detected: models/padim_model.pkl missing
    │      • Run: train_padim()
    │      • Save model to disk
    │
    └─→ 5. App ready! 🎉
           • URL: https://tablet-defect-detection.onrender.com
```

### **Environment**

| Resource | Render Free Tier | Your App Needs |
|----------|-----------------|----------------|
| **RAM** | 512 MB | ~500MB (tight fit!) |
| **CPU** | Shared | CPU-only PyTorch |
| **Storage** | 10 GB | ~500MB (dataset + model) |
| **Port** | 10000 | Configured ✅ |
| **Auto-sleep** | 15 min idle | Yes (wakes on request) |

---

## 🔄 State Management

### **Streamlit Session State**

```python
# Model loaded ONCE per session (cached)
@st.cache_resource
def load_model():
    # Expensive operation, cached in memory
    return model, extractor, device

# Called on every user interaction
def main():
    # UI re-renders on every interaction
    # But cached functions don't re-run!
    model = load_model()  # ← Fast (from cache)
    
    if uploaded_file:
        result = predict(image)  # ← New prediction
```

**Key Points:**
- ✅ Model loads once, stays in memory
- ✅ Predictions are computed fresh each time
- ✅ Multiple users share the same model instance

---

## 📡 Alternative: FastAPI Architecture (If You Want API)

If you want to add a REST API later, here's the structure:

```python
# api.py (NEW FILE)
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint for image prediction
    
    Request:
        POST /predict
        Body: multipart/form-data with image file
    
    Response:
        {
            "is_defective": true,
            "anomaly_score": 0.857,
            "confidence": 0.92,
            "heatmap_base64": "iVBORw0KG..."
        }
    """
    image = Image.open(io.BytesIO(await file.read()))
    result = predict_defect(image, model, extractor, device)
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
```

**When to use FastAPI:**
- 🔧 Building a mobile app (React Native, Flutter)
- 🔧 Integrating with other services
- 🔧 Need authentication/rate limiting
- 🔧 Production-grade API

**When to stick with Streamlit:**
- ✅ Demo/portfolio project (your case!)
- ✅ Internal tools/dashboards
- ✅ Quick prototypes
- ✅ Non-technical users

---

## 🎓 Interview Talking Points

### **"How does your model training work?"**

> "I use the PaDiM algorithm which is unsupervised. During training, I extract multi-scale features from 219 normal tablet images using a pretrained ResNet-18 backbone. For each spatial location, I fit a multivariate Gaussian distribution. The model learns what 'normal' looks like without needing labeled defect data."

### **"How do you handle inference?"**

> "For inference, I extract features from the input image and compute the Mahalanobis distance from the learned distribution at each pixel. This gives me both an image-level anomaly score and a pixel-level heatmap showing where defects are located."

### **"Why Streamlit and not FastAPI?"**

> "For a portfolio demo, Streamlit is ideal because it provides both the frontend and backend in a single Python file. It's perfect for showcasing the model to recruiters. If I were building a production system for integration with mobile apps or other services, I'd use FastAPI to create RESTful endpoints."

### **"How did you handle the model file in deployment?"**

> "The trained model is 252MB, which exceeds GitHub's file size limits. Instead of committing it, I designed the app to automatically train the model on first deployment. This takes 2-3 minutes but only happens once. Subsequent restarts reuse the saved model."

---

## 📚 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | PyTorch 2.0+ | Model framework |
| **Backbone** | ResNet-18 (pretrained) | Feature extraction |
| **Algorithm** | PaDiM | Anomaly detection |
| **Frontend** | Streamlit | Web UI |
| **Visualization** | OpenCV, Matplotlib | Heatmaps |
| **Deployment** | Render.com | Cloud hosting |
| **Version Control** | Git/GitHub | Code management |

---

## ✅ Summary

**Your System:**
- ✅ **Trained Model**: PaDiM trained on 219 normal images
- ✅ **Inference**: Real-time (<0.5s per image)
- ✅ **No API**: Streamlit web app (perfect for demos)
- ✅ **Auto-Training**: On first Render deployment
- ✅ **Deployment**: Single web service, auto-deploys on git push

**Perfect for**: Portfolio, demos, interviews, learning
**Production upgrade path**: Add FastAPI layer if needed later

---

**Questions? Check the code comments or ask during interviews!** 🚀
