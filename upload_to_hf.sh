#!/bin/bash
# Upload script for Phi-2 analysis to HuggingFace

echo "Uploading Phi-2 Mechanistic Interpretability Analysis to HuggingFace"
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Please login first: huggingface-cli login"
    exit 1
fi

USERNAME="totalorganfailure"
REPO_NAME="phi2-weights"
REPO_ID="$USERNAME/$REPO_NAME"

echo "Repository: $REPO_ID"
echo "Upload directory: phi2_analysis_upload/"

echo "Checking/creating repository..."
huggingface-cli repo create $REPO_NAME --type dataset --private false 2>/dev/null || echo "Repository already exists or creation failed - continuing with upload..."

echo "Uploading files..."
cd phi2_analysis_upload

echo "Uploading large files..."
if [ -f "raw_data/phi2_weights.h5" ]; then
    huggingface-cli upload $REPO_ID raw_data/phi2_weights.h5 --repo-type dataset
fi

echo "Uploading remaining files..."
huggingface-cli upload $REPO_ID . --repo-type dataset

echo "Upload complete!"
echo "View your dataset: https://huggingface.co/datasets/$REPO_ID"
