#!/bin/bash
# Quick upload script for trained model to HuggingFace
# Usage: ./upload_model.sh username/model-name

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HuggingFace Model Upload${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if repo ID provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Repository ID required${NC}"
    echo
    echo "Usage:"
    echo "  ./upload_model.sh username/model-name"
    echo
    echo "Example:"
    echo "  ./upload_model.sh professorsynapse/claudesidian-tools-7b-v1"
    echo
    exit 1
fi

REPO_ID="$1"
MODEL_PATH="./kto_output_rtx3090/final_model"
CREATE_GGUF="${2:-no}"  # Optional second argument for GGUF creation

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    echo
    echo "Make sure training has completed and model is saved."
    exit 1
fi

# Check if .env exists in root directory (../../.env)
ENV_PATH="../../.env"
if [ ! -f "$ENV_PATH" ]; then
    # Fallback to local .env
    ENV_PATH=".env"
    if [ ! -f "$ENV_PATH" ]; then
        echo -e "${RED}Error: .env file not found${NC}"
        echo
        echo "Create .env file with your HuggingFace token:"
        echo "  In root directory: /mnt/c/Users/Joseph/Documents/Code/Toolset-Training/.env"
        echo "  Or locally: cp .env.example .env"
        echo "  # Add: HF_TOKEN=hf_your_token_here or HF_API_KEY=hf_your_token_here"
        echo
        echo "Get token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
fi

# Load environment variables
export $(grep -v '^#' "$ENV_PATH" | xargs)

echo -e "${GREEN}✓${NC} Model found: $MODEL_PATH"
echo -e "${GREEN}✓${NC} .env file found: $ENV_PATH"
echo -e "${GREEN}✓${NC} Uploading to: $REPO_ID"
if [[ "$CREATE_GGUF" == "yes" || "$CREATE_GGUF" == "y" ]]; then
    echo -e "${GREEN}✓${NC} GGUF creation: Enabled"
else
    echo -e "${BLUE}ℹ${NC} GGUF creation: Disabled (use './upload_model.sh $REPO_ID yes' to enable)"
fi
echo

# Ask for confirmation
read -p "Continue with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Activate conda environment
echo
echo -e "${BLUE}Activating environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./venv

# Run upload
echo
echo -e "${BLUE}Uploading model...${NC}"
if [[ "$CREATE_GGUF" == "yes" || "$CREATE_GGUF" == "y" ]]; then
    python src/upload_to_hf.py "$MODEL_PATH" "$REPO_ID" --save-method merged_16bit --create-gguf
else
    python src/upload_to_hf.py "$MODEL_PATH" "$REPO_ID" --save-method merged_16bit
fi

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Upload complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "View your model at:"
echo "  https://huggingface.co/$REPO_ID"
echo
