#!/bin/bash
# Render startup script

echo "=== Automated Tablet Defect Detection - Deployment ==="
echo ""

# Check if model exists
if [ ! -f "models/padim_model.pkl" ]; then
    echo "Model not found. Training now..."
    echo ""
    python train.py
    echo ""
    echo "Model training complete!"
else
    echo "Model found. Skipping training."
fi

echo ""
echo "Starting Streamlit app..."
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
