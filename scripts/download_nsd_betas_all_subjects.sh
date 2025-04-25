#!/bin/bash
# exit immediately if a command exits with a non-zero status
set -e

# AWS Config
# If you want to use AWS credentials, set them here.
# If not, you can leave these empty and use --no-sign-request.
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_REGION="us-east-1"
AWS_OUTPUT="json"

# Ensure AWS CLI is installed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI not found. Please install AWS CLI before running this script."
    exit 1
else
    echo "AWS CLI is already installed."
fi

# Configure AWS CLI (will use the credentials provided or empty values)
echo "Configuring AWS CLI..."
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_REGION"
aws configure set output "$AWS_OUTPUT"

# Verify AWS configuration
echo "AWS Configuration:"
aws configure list

# Create base data directory for NSD betas
BASE_DIR="./data/nsddata_betas/ppdata"
mkdir -p "$BASE_DIR"

# Loop Over Subjects and Download Betas
for i in {01..08}; do
    subj="subj$i"
    echo "-----------------------------------"
    echo "Downloading NSD betas for $subj..."
    
    # Define the destination directory for current subject's betas
    DEST_DIR="${BASE_DIR}/${subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    mkdir -p "$DEST_DIR"
    
    # Use AWS CLI to sync betas files (only syncing files with "betas" and "nii" in their names)
    aws s3 sync --exclude "*" --include "*betas*nii*" \
        s3://natural-scenes-dataset/nsddata_betas/ppdata/${subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ \
        "$DEST_DIR" --no-sign-request
        
    echo "Betas downloaded for $subj into $DEST_DIR."
done

echo "-----------------------------------"
echo "All NSD betas downloads completed successfully!"
