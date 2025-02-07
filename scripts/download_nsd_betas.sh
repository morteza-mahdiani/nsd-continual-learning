#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# AWS Credentials (Replace these with your credentials or load from environment variables)
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_REGION="us-east-1"
AWS_OUTPUT="json"

# Ensure AWS CLI is installed
#echo "Updating package list..."
#sudo apt update -y

if ! command -v aws &> /dev/null
then
    echo "AWS CLI not found. Installing..."
    sudo apt install -y awscli
else
    echo "AWS CLI is already installed."
fi

# Configure AWS CLI
echo "Configuring AWS CLI..."
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_REGION"
aws configure set output "$AWS_OUTPUT"

# Verify AWS configuration
echo "AWS Configuration:"
aws configure list

# Create data directory if not exists
mkdir -p ./data/

# Download betas data for all subjects
#echo "Syncing betas files from S3 into ./data/ ..."
aws s3 sync --exclude "*" --include "*betas*nii*" \
    s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ ./data/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ --no-sign-request

echo "Betas sync complete!"

# Download behavioral responses for subjects 01 to 08
echo "Downloading behavioral response files..."
for i in {01..08}
do
    subj="subj0$i"
    aws s3 cp "s3://natural-scenes-dataset/nsddata/ppdata/$subj/behav/responses.tsv" "./data/nsddata/ppdata/$subj/behav/responses.tsv" --no-sign-request
    echo "Downloaded responses.tsv for $subj"
done

echo "All downloads complete!"

# Verify the downloaded files
echo "Verifying downloaded files..."
ls -lh ./data/

echo "Script execution completed successfully."
