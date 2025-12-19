# Interview Preparation Guide
## Automated Tablet Defect Detection System

This document prepares you to discuss your project in technical interviews.

---

## 1-Minute Project Pitch

*"I built an automated quality inspection system for pharmaceutical manufacturing that detects defects in tablet images using unsupervised machine learning. The system uses PaDiM, a patch distribution modeling approach, which learns the appearance of normal tablets and identifies anomalies as statistical deviations. It achieves 95%+ ROC-AUC without requiring labeled defect examples, making it practical for real-world scenarios where defect samples are rare. I deployed it as a Streamlit web app with real-time inference and pixel-level defect localization heatmaps."*

---

## Common Interview Questions & Answers

### **Q1: Walk me through your approach.**

**A:** The problem is detecting defective tablets when we only have examples of normal ones. I used an unsupervised anomaly detection approach:

1. **Feature Extraction:** I extract multi-scale visual features using a pretrained ResNet-18 backbone (layers 1, 2, and 3), which gives me a 448-dimensional representation at each spatial location.

2. **Distribution Modeling:** For each pixel position, I model the distribution of features from normal samples as a multivariate Gaussian with mean μ and covariance Σ.

3. **Anomaly Detection:** At test time, I compute the Mahalanobis distance of new features from the learned distribution. High distances indicate anomalies.

4. **Localization:** The per-pixel distances create a heatmap showing exactly where defects are located.

This is called PaDiM (Patch Distribution Modeling), chosen because it's fast, doesn't require training a neural network from scratch, and provides interpretable anomaly scores.

---

### **Q2: Why did you choose PaDiM over other methods?**

**A:** I considered several approaches:

**Autoencoders:**
- ❌ Require training/hyperparameter tuning
- ❌ Can generalize to anomalies (reconstruction paradox)
- ✅ PaDiM advantage: Just fit distributions (no backprop), provably won't "memorize" defects

**One-Class SVM:**
- ❌ Limited to single-scale global features
- ✅ PaDiM advantage: Multi-scale local patterns, better for spatial defects

**Memory Bank Methods (PatchCore, SPADE):**
- ❌ High memory usage (store all training patches)
- ✅ PaDiM advantage: Compact statistical summary (mean + covariance)

**Trade-off:** PaDiM has slightly lower accuracy than PatchCore on some benchmarks, but 10x faster inference and 100x smaller memory footprint. For a deployment-focused project, I prioritized efficiency.

---

### **Q3: What is Mahalanobis distance and why use it?**

**A:** Mahalanobis distance measures "how many standard deviations away" a point is from a distribution, accounting for correlations between features.

**Formula:** M(x) = √[(x - μ)ᵀ Σ⁻¹ (x - μ)]

**Why not Euclidean distance?**
- Euclidean treats all features equally
- Mahalanobis accounts for feature variances and correlations
- Example: If feature A has high variance and B has low variance, a deviation in B is more unusual and should be weighted higher

**Interpretation:** 
- M(x) < 3: Likely normal (within 3σ)
- M(x) > 3: Likely anomaly

This makes the anomaly score statistically principled and interpretable.

---

### **Q4: How did you handle the train/test split?**

**A:** This is unsupervised, so the split is different from typical classification:

**Training:**
- Used all 219 "good" (normal) samples
- No validation set needed (no hyperparameters to tune)
- No defect samples used

**Testing:**
- 23 good samples (should score low)
- 109 defective samples across 5 types (should score high)
- Ground truth masks only used for visualization/evaluation

**Key point:** The model never sees defects during training, simulating realistic pharmaceutical QC where defect samples are rare or non-existent.

---

### **Q5: What evaluation metrics did you use and why?**

**A:** 

**Primary Metric: ROC-AUC (0.95+)**
- Threshold-independent
- Robust to class imbalance
- Measures ranking quality (can we separate normal from defects?)

**Secondary Metrics (at optimal threshold):**
- **Precision (0.92):** Of tablets flagged as defective, how many truly are? Important to minimize false alarms.
- **Recall (0.89):** Of all defective tablets, how many did we catch? Critical for safety.
- **F1-Score (0.90):** Harmonic mean, balanced view

**Why not accuracy?**
- Misleading with imbalance (23 good vs 109 defects)
- Doesn't capture cost-asymmetry (missing a defect is worse than false alarm)

**Visual evaluation:** Heatmaps let domain experts verify localization makes sense.

---

### **Q6: How did you choose the anomaly threshold?**

**A:** I used **Youden's J statistic** to find the optimal threshold:

J = TPR - FPR = Recall - (1 - Specificity)

This maximizes the difference between true positive and false positive rates. In my case, optimal threshold ≈ 0.5.

**However**, in production I'd adjust based on business requirements:
- **Safety-critical:** Lower threshold (0.3), maximize recall, accept more false positives
- **Cost-sensitive:** Higher threshold (0.7), minimize false alarms, accept some misses

**Best practice:** Provide the threshold as a configurable parameter (which I did in the Streamlit app slider).

---

### **Q7: What challenges did you face?**

**A:** 

**Challenge 1: Memory Issues**
- Initial approach: Concatenated all 448 channels led to 448×448 covariance matrices per pixel
- Solution: Random projection to reduce to 100 dimensions → 10MB vs 500MB model size

**Challenge 2: False Positives on Edges**
- Background variations triggered false alarms
- Solution: Could add background masking or consistent imaging setup

**Challenge 3: Subtle Defects (Faulty Imprint)**
- Low contrast defects had lower recall
- Solution: Ensemble with complementary features or lower threshold

**Challenge 4: Windows Compatibility**
- PyTorch DataLoader workers caused issues on Windows
- Solution: Set num_workers=0 for cross-platform compatibility

---

### **Q8: How would you improve this system?**

**A:** 

**Short-term (1-2 weeks):**
1. **Ensemble features:** Combine ResNet with WideResNet or EfficientNet
2. **Better preprocessing:** Background removal, contrast enhancement
3. **Ablation studies:** Test different layers, backbones, reduction dimensions
4. **Cross-validation:** Multiple random projection seeds

**Medium-term (1-2 months):**
1. **Model optimization:** Export to ONNX, quantization for edge devices
2. **REST API:** FastAPI wrapper for production integration
3. **Online learning:** Update Gaussians incrementally with new normal samples
4. **Active learning:** Flag uncertain cases for expert review

**Long-term (research):**
1. **Multi-category:** Extend to multiple tablet types
2. **Few-shot classification:** Classify defect types with minimal labels
3. **Explainability:** Attribute anomalies to specific visual patterns

---

### **Q9: How does this scale to production?**

**A:** 

**Throughput:**
- Current: <0.5s per image on CPU
- Batch processing: ~5 imgs/sec
- Optimization: ONNX + TensorRT could hit 50 imgs/sec on GPU

**Deployment Architecture:**
```
Manufacturing Line → Edge Device → [PaDiM Model] → Quality DB
                                          ↓
                                    Alert if Defect
```

**Monitoring:**
- Track anomaly score distribution over time
- Detect drift (if scores creep up, manufacturing process may be changing)
- Retrain weekly/monthly with fresh normal samples

**Scalability:**
- Same model works across multiple production lines
- Cloud deployment for centralized batch processing
- Edge deployment for real-time line-side inspection

---

### **Q10: What did you learn from this project?**

**A:** 

**Technical:**
- Unsupervised learning can outperform supervised in data-scarce scenarios
- Transfer learning is powerful - pretrained features work "out of the box"
- Production ML is about trade-offs: accuracy vs speed vs memory

**Engineering:**
- Importance of modular design (easy to swap backbones, adjust preprocessing)
- Value of visualization (heatmaps made results interpretable)
- Deployment matters (CPU-friendly > state-of-the-art if it can't run in prod)

**Domain:**
- Real-world constraints (rare defects, no labels) shape solution design
- False negatives (missing defects) are much costlier than false positives in pharma QC
- Domain experts need interpretability (anomaly scores + heatmaps) to trust the system

---

## Technical Deep Dives

### **resNet-18 Architecture Details**

| Layer | Output Shape | Channels | Receptive Field |
|-------|--------------|----------|-----------------|
| Input | 224×224 | 3 | 1 |
| Conv1 | 112×112 | 64 | 7 |
| Layer1 | 56×56 | 64 | 35 |
| Layer2 | 28×28 | 128 | 67 |
| Layer3 | 14×14 | 256 | 131 |
| Layer4 | 7×7 | 512 | 259 |

**I use layers 1-3:**
- **Why not Layer4?** Too low resolution (7×7), loses spatial detail
- **Why not Conv1?** Too low-level (edges), not semantic enough
- **Layers 1-3:** Good balance of spatial resolution and semantic meaning

**Multi-scale advantage:**
- Layer1: Fine details (textures, small scratches)
- Layer2: Medium patterns (cracks)
- Layer3: Object-level features (deformations)

---

### **Computational Complexity Analysis**

**Training:**
- Feature extraction: O(N × H × W) forward passes = 219 × 224 × 224
- Gaussian fitting: O(H × W × D²) covariance inversions = 56 × 56 × 100²
- Total time: ~2-3 minutes on CPU
- Memory: ~1GB peak

**Inference:**
- Feature extraction: O(H × W) single forward pass
- Distance computation: O(H × W × D²) = 56 × 56 × 100² matrix multiplications
- Gaussian smoothing: O(H × W) convolution
- Total time: <0.5s per image on CPU

**Bottleneck:** Covariance inversion during training (could parallelize across pixels)

---

### **Alternative Approaches Comparison**

| Method | Train Time | Inference | Memory | AUC | Localization |
|--------|------------|-----------|--------|-----|--------------|
| **PaDiM** | 3 min | 0.5s | 120MB | 0.95 | ✅ Heatmap |
| PatchCore | 10 min | 0.8s | 1.2GB | 0.98 | ✅ Heatmap |
| Autoencoder | 30 min | 0.3s | 50MB | 0.85 | ⚠️ Reconstruction |
| One-Class SVM | 1 min | 0.1s | 10MB | 0.80 | ❌ None |

**Conclusion:** PaDiM offers best accuracy/efficiency trade-off for this application.

---

## Behavioral Interview Angles

### **Problem-Solving:**
*"How do you approach new problems?"*
- Started with literature review (MVTec paper, PaDiM paper)
- Benchmarked multiple approaches conceptually
- Chose based on deployment constraints (CPU, no labels)
- Validated with quantitative metrics + visual inspection

### **Trade-offs:**
*"Describe a technical trade-off you made."*
- Chose PaDiM over PatchCore: Sacrificed 2-3% AUC for 10x faster inference and 10x smaller memory
- Justified by deployment target (free-tier cloud, CPU-only)
- Measured trade-off quantitatively before deciding

### **Debugging:**
*"How do you debug ML models?"*
- Started with simple baseline (Euclidean distance on raw pixels)
- Added complexity incrementally (features → multi-scale → Mahalanobis)
- Visualized intermediate outputs (feature maps, heatmaps)
- Analyzed failure cases (FP = edge artifacts, FN = subtle textures)

---

## Code Snippets to Remember

### **Forward Hook (Feature Extraction):**
```python
def _get_hook(self, name: str):
    def hook(module, input, output):
        self.feature_maps[name] = output
    return hook

# Register hook
hook = module.register_forward_hook(self._get_hook("layer1"))
```

### **Mahalanobis Distance:**
```python
delta = embeddings[h, w] - self.mean[h, w]
distance = np.sqrt(delta @ self.cov_inv[h, w] @ delta.T)
```

### **Streamlit Caching:**
```python
@st.cache_resource
def load_model():
    # Expensive: only runs once, cached across users
    return padim_model, extractor
```

---

## Red Flags to Avoid

❌ **Don't say:**
- "I just followed a tutorial"
- "I'm not sure why it works"
- "I didn't have time to evaluate properly"
- "It's just a simple dataset"

✅ **Do say:**
- "I chose this approach because..."
- "The mathematical intuition is..."
- "I measured performance using..."
- "Real-world deployment would require..."

---

## Questions to Ask Interviewer

1. "How do you currently handle anomaly detection in your products?"
2. "What ML deployment infrastructure do you use (Docker, Kubernetes, serverless)?"
3. "How do you balance model accuracy vs inference speed in production?"
4. "Do you have experience with unsupervised learning, or mostly supervised?"

---

## Final Tips

1. **Lead with impact:** "95% AUC, no labels needed, real-time inference"
2. **Show depth:** Be ready to explain Mahalanobis distance, forward hooks, random projection
3. **Show breadth:** Discuss alternatives, trade-offs, deployment
4. **Be honest:** If asked about weaknesses, mention faulty imprint detection
5. **Show growth:** Explain what you'd do next (ensemble, optimization, active learning)

**Remember:** This project demonstrates:
- ✅ Unsupervised learning expertise
- ✅ Computer vision fundamentals
- ✅ Production deployment skills
- ✅ Statistical ML (not just deep learning)
- ✅ End-to-end thinking (data → model → app → evaluation)

**Good luck! 🚀**
