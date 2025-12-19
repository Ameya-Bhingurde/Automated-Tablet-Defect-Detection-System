"""
Streamlit Application for Automated Tablet Defect Detection
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.feature_extractor import FeatureExtractor, extract_embeddings
from src.padim import PaDiM
from src.visualize import apply_heatmap


@st.cache_resource
def load_model():
    """Load PaDiM model and feature extractor (cached)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PaDiM model
    model_path = config.MODEL_DIR / "padim_model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}. Please run train.py first.")
        st.stop()
    
    padim_model = PaDiM()
    padim_model.load(model_path)
    
    # Load feature extractor
    extractor = FeatureExtractor(
        backbone=config.BACKBONE,
        layers=config.FEATURE_LAYERS
    ).to(device)
    
    return padim_model, extractor, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess uploaded image"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_defect(image: Image.Image, padim_model, extractor, device):
    """Run inference on uploaded image"""
    
    # Preprocess
    img_tensor = preprocess_image(image).to(device)
    
    # Extract embeddings
    with torch.no_grad():
        embeddings = extract_embeddings(extractor, img_tensor)
    
    # Predict
    embeddings_np = embeddings.cpu().numpy()
    anomaly_score, anomaly_map = padim_model.predict(embeddings_np)
    
    return anomaly_score, anomaly_map


def main():
    """Main Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="Tablet Defect Detection",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .defect-alert {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #c62828;
            font-weight: 600;
        }
        .normal-alert {
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2e7d32;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">💊 Automated Tablet Defect Detection</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unsupervised Computer Vision Quality Inspection System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/pill.png", width=80)
        st.title("⚙️ Settings")
        
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help="Adjust sensitivity: lower = more sensitive to defects"
        )
        
        show_heatmap = st.checkbox("Show Anomaly Heatmap", value=True)
        heatmap_alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4, 0.05)
        
        st.divider()
        st.subheader("📊 Model Info")
        st.markdown(f"""
        - **Method:** PaDiM
        - **Backbone:** ResNet-18
        - **Layers:** {', '.join(config.FEATURE_LAYERS)}
        - **Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}
        """)
        
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        This system uses **PaDiM** (Patch Distribution Modeling) for 
        unsupervised anomaly detection in pharmaceutical tablets.
        
        **Features:**
        - ✅ Image-level defect classification
        - 🎯 Pixel-level defect localization
        - 📈 Anomaly score quantification
        - 🚀 CPU-friendly inference
        """)
    
    # Load model
    with st.spinner("Loading model..."):
        padim_model, extractor, device = load_model()
    
    # Main content
    st.divider()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a tablet image for inspection",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    # Demo images section
    col1, col2 = st.columns([3, 1])
    with col2:
        use_demo = st.button("🎲 Try Demo Image")
    
    if use_demo:
        # Load a random test image
        demo_dir = config.TEST_DIR / "good"
        demo_images = list(demo_dir.glob("*.png"))
        if demo_images:
            demo_path = np.random.choice(demo_images)
            uploaded_file = demo_path
    
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, Path):
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")
        
        # Display original image
        st.subheader("📸 Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # Run inference
        with st.spinner("🔍 Analyzing image..."):
            anomaly_score, anomaly_map = predict_defect(
                image, padim_model, extractor, device
            )
        
        # Display results
        st.divider()
        st.subheader("🎯 Inspection Results")
        
        # Prediction
        is_defective = anomaly_score > threshold
        
        if is_defective:
            st.markdown(f"""
                <div class="defect-alert">
                    ⚠️ DEFECTIVE TABLET DETECTED
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="normal-alert">
                    ✅ NORMAL TABLET (No Defects)
                </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Anomaly Score",
                value=f"{anomaly_score:.4f}",
                delta="Defect" if is_defective else "Normal",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Threshold",
                value=f"{threshold:.3f}",
                delta=f"{(anomaly_score/threshold - 1)*100:+.1f}%" if threshold > 0 else "N/A"
            )
        
        with col3:
            confidence = abs(anomaly_score - threshold) / threshold if threshold > 0 else 0
            st.metric(
                label="Confidence",
                value=f"{min(confidence * 100, 100):.1f}%"
            )
        
        # Heatmap visualization
        if show_heatmap:
            st.divider()
            st.subheader("🔥 Anomaly Heatmap")
            st.markdown("*Highlighted regions indicate potential defects*")
            
            # Create heatmap overlay
            img_np = np.array(image)
            heatmap_overlay = apply_heatmap(
                img_np, 
                anomaly_map, 
                alpha=heatmap_alpha,
                colormap=config.HEATMAP_COLORMAP
            )
            
            # Display side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            
            with col2:
                st.image(heatmap_overlay, caption="Defect Localization", 
                        use_container_width=True)
        
        # Download results
        st.divider()
        
        if st.button("💾 Download Results"):
            # Create annotated image
            img_np = np.array(image)
            result_img = apply_heatmap(img_np, anomaly_map, alpha=heatmap_alpha)
            
            # Add text annotation
            import cv2
            prediction_text = "DEFECTIVE" if is_defective else "NORMAL"
            color = (255, 0, 0) if is_defective else (0, 255, 0)
            cv2.putText(result_img, f"{prediction_text} ({anomaly_score:.3f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
            
            # Convert to bytes
            result_pil = Image.fromarray(result_img)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="defect_detection_result.png",
                mime="image/png"
            )
    
    else:
        # Instructions when no image uploaded
        st.info("👆 Please upload an image or click 'Try Demo Image' to start inspection.")
        
        # Example gallery
        st.divider()
        st.subheader("📚 Example Defect Types")
        
        cols = st.columns(5)
        defect_examples = {
            "Normal": config.TEST_DIR / "good",
            "Crack": config.TEST_DIR / "crack",
            "Poke": config.TEST_DIR / "poke",
            "Scratch": config.TEST_DIR / "scratch",
            "Squeeze": config.TEST_DIR / "squeeze"
        }
        
        for idx, (defect_name, defect_dir) in enumerate(defect_examples.items()):
            if defect_dir.exists():
                images = list(defect_dir.glob("*.png"))
                if images:
                    with cols[idx % 5]:
                        example_img = Image.open(images[0])
                        st.image(example_img, caption=defect_name, use_container_width=True)


if __name__ == "__main__":
    main()
