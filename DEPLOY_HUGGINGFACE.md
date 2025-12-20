# 🤗 Hugging Face Spaces Deployment Guide

## 📋 Overview

Hugging Face Spaces is **perfect** for your Streamlit ML demo:
- ✅ **16GB RAM** (vs 512MB on Render)
- ✅ **Free forever**
- ✅ **No sleep** (always fast)
- ✅ **Built for ML demos**

---

## 🚀 Deployment Steps

### **Method 1: Direct GitHub Integration (Recommended)**

1. **Create Space**
   - Go to: https://huggingface.co/new-space
   - Space name: `tablet-defect-detection`
   - SDK: **Streamlit**
   - Visibility: **Public**
   - Click "Create Space"

2. **Connect GitHub**
   - In your Space, go to "Settings"
   - Find "Repository"
   - Click "Link to GitHub"
   - Select: `Ameya-Bhingurde/Automated-Tablet-Defect-Detection-System`
   - Branch: `main`
   - Click "Link"

3. **Configure**
   - Hugging Face auto-detects `app.py` and `requirements.txt`
   - That's it! It deploys automatically

---

### **Method 2: Manual File Upload**

If GitHub linking doesn't work:

1. **Clone Your Repo Locally** (you already have it)

2. **Copy HF_README.md**
   ```bash
   copy HF_README.md README.md
   ```

3. **Upload to Space**
   - In your Space, click "Files" tab
   - Click "Add file" → "Upload files"
   - Upload these files:
     - `app.py`
     - `config.py`
     - `requirements.txt`
     - `README.md` (the HF version)
     - `train.py`
     - `inference.py`
     - `evaluate.py`
     - Entire `src/` folder
     - Entire `capsule/` folder (dataset)
     - `models/.gitkeep` (empty folder placeholder)
     - `results/.gitkeep`

4. **Commit**
   - Write commit message: "Initial deployment"
   - Click "Commit to main"
   - **Space auto-deploys!**

---

## ⚙️ What Happens Automatically

When you deploy, Hugging Face:

1. ✅ Detects Streamlit SDK from README
2. ✅ Installs `requirements.txt`
3. ✅ Runs `app.py`
4. ✅ Your app trains the model on first load
5. ✅ Serves at: `https://huggingface.co/spaces/YOUR_USERNAME/tablet-defect-detection`

---

## 🎯 Advantages Over Render

| Feature | Hugging Face | Render Free |
|---------|--------------|-------------|
| **RAM** | 16GB | 512MB |
| **Sleep** | Never | After 15min |
| **ML Focus** | Yes | No |
| **Community** | ML researchers | General devs |
| **Visibility** | High (HF community) | Low |

---

## 📱 Your Live Space URL

After deployment:
```
https://huggingface.co/spaces/YOUR_USERNAME/tablet-defect-detection
```

**Perfect for:**
- ✅ Resume/portfolio
- ✅ Sharing with recruiters
- ✅ ML community visibility
- ✅ Always-on demos

---

## 🔧 Troubleshooting

### **Issue: Model training times out**

**Solution:** Pre-train locally and upload model:
```bash
# Local
python train.py

# Then upload models/padim_model.pkl to Space
# (Use Git LFS for large files)
```

### **Issue: Dataset too large**

Dataset (capsule/) is ~100MB - should be fine.

If issues, use Git LFS:
```bash
git lfs install
git lfs track "*.png"
git lfs track "models/*.pkl"
```

### **Issue: Slow first load**

Normal! Model trains on first visit (2-3 min). After that, instant.

---

## 🎓 Pro Tips

1. **Pin Your Space** - Makes it show up first on your profile
2. **Add Tags** - `computer-vision`, `anomaly-detection`, `streamlit`
3. **Write Good README** - Shows on Space homepage (use HF_README.md)
4. **Share Link** - Better than Render for ML community

---

## ✅ Deployment Checklist

- [ ] Create account on Hugging Face
- [ ] Create new Space (Streamlit SDK)
- [ ] Link GitHub repo OR upload files manually
- [ ] Wait for build (~5-8 minutes)
- [ ] Test upload functionality
- [ ] Add to resume/portfolio!

---

**Ready to deploy? Let's do it!** 🚀
