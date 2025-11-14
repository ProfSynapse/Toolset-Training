#!/bin/bash
# Setup script for RTX 3090 KTO Training
# Run with: bash setup.sh

set -e  # Exit on error

echo "=========================================="
echo "RTX 3090 KTO Training - Setup"
echo "=========================================="

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10+ required"
    exit 1
fi

echo "✓ Python version OK"

# Check NVIDIA GPU
echo -e "\n[2/6] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
fi

# Create virtual environment
echo -e "\n[3/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[4/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo -e "\n[5/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install requirements
echo -e "\n[6/6] Installing requirements..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Test installation: python -c 'import torch; print(torch.cuda.is_available())'"
echo "  3. Run training: python train_kto.py --model-size 7b --dry-run"
echo ""
echo "For training:"
echo "  python train_kto.py --model-size 7b"
echo ""
