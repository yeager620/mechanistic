#\!/bin/bash
# Direct upload to totalorganfailure/phi2-weights
echo "Uploading to totalorganfailure/phi2-weights"
cd phi2_analysis_upload
uv run huggingface-cli upload totalorganfailure/phi2-weights . --repo-type dataset
echo "Upload complete\!"
echo "View your dataset: https://huggingface.co/datasets/totalorganfailure/phi2-weights"
