#!/bin/bash
# Upload fixed structure to HuggingFace
echo "Uploading Phi-2 analysis with proper data structure..."

USERNAME="totalorganfailure"
REPO_NAME="phi2-weights"
REPO_ID="$USERNAME/$REPO_NAME"

cd hf_upload_fixed

echo "Repository: $REPO_ID"
echo "Upload directory: $(pwd)"

# Upload primary data files first (these will show in main table)
echo "Uploading primary data files..."
huggingface-cli upload $REPO_ID phi2_weights.h5 --repo-type dataset
huggingface-cli upload $REPO_ID weight_summary.csv --repo-type dataset
huggingface-cli upload $REPO_ID analysis_stats.json --repo-type dataset
huggingface-cli upload $REPO_ID README.md --repo-type dataset
huggingface-cli upload $REPO_ID analysis_index.md --repo-type dataset

# Upload point clouds
echo "Uploading point clouds..."
huggingface-cli upload $REPO_ID point_clouds/ --repo-type dataset

# Upload visualizations as subdirectory
echo "Uploading visualizations..."
huggingface-cli upload $REPO_ID visualizations/ --repo-type dataset

# Upload scripts and documentation
echo "Uploading scripts and documentation..."
huggingface-cli upload $REPO_ID scripts/ --repo-type dataset
huggingface-cli upload $REPO_ID documentation/ --repo-type dataset

echo "Upload complete!"
echo "View your dataset: https://huggingface.co/datasets/$REPO_ID"
