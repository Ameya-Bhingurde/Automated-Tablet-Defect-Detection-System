# 🚀 Deployment Guide - Render.com

This guide explains how to deploy the **Automated Tablet Defect Detection System** to Render.com's free tier.

---

## 📋 Prerequisites

1. **GitHub Repository**: Your code is already pushed to GitHub ✅
2. **Render Account**: Create a free account at [render.com](https://render.com)

---

## 🎯 Deployment Steps

### **Step 1: Create Render Account**

1. Go to [render.com](https://render.com)
2. Click **"Get Started"**
3. Sign up with GitHub (recommended for easy integration)

### **Step 2: Create New Web Service**

1. From your Render Dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub repository:
   - Click **"Connect account"** (if first time)
   - Select **"Ameya-Bhingurde/Automated-Tablet-Defect-Detection-System"**
   - Click **"Connect"**

### **Step 3: Configure Web Service**

Fill in the following settings:

| Field | Value |
|-------|-------|
| **Name** | `tablet-defect-detection` (or your choice) |
| **Region** | Choose closest to you |
| **Branch** | `main` |
| **Root Directory** | Leave blank |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=10000 --server.address=0.0.0.0` |
| **Instance Type** | `Free` |

### **Step 4: Environment Variables** (Optional)

If needed, add these environment variables:
- `PYTHON_VERSION`: `3.10`
- `PORT`: `10000`

### **Step 5: Deploy!**

1. Click **"Create Web Service"**
2. Render will:
   - Clone your repository
   - Install dependencies (~5-10 minutes)
   - Start the Streamlit app
   - **Train the model automatically** on first run (~3 minutes)

3. Monitor the deployment logs for progress

---

## ⏱️ First Deployment Timeline

- **Total time**: ~8-12 minutes
- **Dependency installation**: 5-10 minutes
- **Model training** (automatic): 2-3 minutes
- **App startup**: ~30 seconds

⚠️ **Note**: The model trains automatically on first deployment because the trained model file (252MB) is not in Git. Subsequent restarts will reuse the trained model.

---

## 🌐 Access Your App

Once deployed, Render will provide a URL like:
```
https://tablet-defect-detection-xxxx.onrender.com
```

- **First load**: May take 30-60 seconds (cold start)
- **Subsequent loads**: Instant
- **Auto-sleep**: After 15 minutes of inactivity (free tier)
- **Wake-up time**: ~30 seconds

---

## 🔧 Troubleshooting

### **Issue: Build Fails**

**Solution**: Check logs for specific error. Common fixes:
```bash
# If memory issues, try lighter dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### **Issue: Model Training Times Out**

**Solution**: Free tier has 512MB RAM limit. The app is optimized but if it fails:
1. Reduce `BATCH_SIZE` in `config.py` to 16
2. Set `REDUCE_DIM` to 50 instead of 100

### **Issue: App Crashes on Inference**

**Solution**: Check deployment logs. Usually means:
- Model not trained properly
- Memory limit exceeded
- Try reducing image upload size

### **Issue: Cold Starts**

**Explanation**: Free tier apps sleep after 15 minutes of inactivity.
**Solution**: 
- First request after sleep takes ~30s
- Upgrade to paid tier ($7/month) for always-on service
- Or use a free uptime monitor (e.g., UptimeRobot)

---

## 🔄 Continuous Deployment

Your app is now connected to GitHub!

**To update your app:**
1. Make changes locally
2. Commit: `git commit -m "your message"`
3. Push: `git push origin main`
4. Render **automatically redeploys** 🎉

---

## 📊 Monitoring

View real-time logs in Render Dashboard:
- Go to your service
- Click **"Logs"** tab
- See live application output

---

## 💰 Free Tier Limits

| Resource | Limit |
|----------|-------|
| **RAM** | 512 MB |
| **CPU** | Shared |
| **Build minutes** | 400 min/month |
| **Bandwidth** | 100 GB/month |
| **Uptime** | Sleeps after 15 min inactivity |
| **Instances** | 1 free web service |

✅ **Sufficient for**: Demo, portfolio, low-traffic testing
❌ **Not suitable for**: Production, high-traffic, real-time needs

---

## 🔒 Security Best Practices

1. **Environment Variables**: Use Render's environment variables for any secrets
2. **HTTPS**: Render provides free SSL/TLS certificates
3. **Rate Limiting**: Consider adding rate limiting for production

---

## 🚀 Alternative: Railway.app

If you prefer Railway:

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Railway auto-detects Python
4. Add start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy!

**Railway Free Tier**: $5 credit/month, faster but limited

---

## 📈 Upgrade Options

### **Render Paid Plans**

| Plan | Price | RAM | Features |
|------|-------|-----|----------|
| **Starter** | $7/month | 512 MB | Always-on, no sleep |
| **Standard** | $25/month | 2 GB | Better performance |
| **Pro** | $85/month | 4 GB | Production-ready |

### **When to Upgrade?**

Upgrade if you need:
- ✅ 24/7 uptime (no sleep)
- ✅ Faster response times
- ✅ More concurrent users
- ✅ Higher memory limits

---

## 🎓 Learning Resources

- [Render Docs](https://render.com/docs)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [Troubleshooting Guide](https://render.com/docs/troubleshooting-deploys)

---

## ✅ Deployment Checklist

- [ ] GitHub repo pushed and updated
- [ ] Render account created
- [ ] Web service configured
- [ ] Build and start commands set correctly
- [ ] Deployment successful
- [ ] Model trained automatically
- [ ] App accessible via public URL
- [ ] Test upload functionality
- [ ] Share URL in resume/portfolio! 🎉

---

**Built with ❤️ for advancing quality control in pharmaceutical manufacturing**

**Your Live App**: `https://[your-app-name].onrender.com`
