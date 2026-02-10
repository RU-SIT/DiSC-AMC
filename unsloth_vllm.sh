#!/bin/bash
# filepath: /mnt/d/Rowan/found-in-com/vlms-in-wireless-communication/setup_environment.sh

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   Unsloth2 Environment Setup Script   ${NC}"
echo -e "${BLUE}==============================================${NC}"

read -r -p "Do you want to continue with the installation? (y/n): " response
if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${RED}Installation aborted.${NC}"
    exit 0
fi

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Error occurred${NC}"
        if [ "${1:-}" == "critical" ]; then
            echo -e "${RED}Critical error. Installation cannot continue.${NC}"
            exit 1
        fi
    fi
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Check if unsloth environment exists and delete it if found
echo -e "\n[STEP] ${YELLOW}Checking for existing environment...${NC}"
if conda info --envs | grep -q "^unsloth2 "; then
    echo "Found existing 'unsloth2' environment. Removing it..."
    conda env remove -n unsloth2 -y
    check_status
else
    echo "No existing 'unsloth2' environment found."
fi

# Create new conda environment
echo -e "\n[STEP] ${YELLOW}Creating new 'unsloth2' environment...${NC}"
conda create --name unsloth2 python=3.10 -y
check_status "critical"

# Install PyTorch with CUDA
echo -e "\n[STEP] ${YELLOW}Installing PyTorch with CUDA...${NC}"
conda install -n unsloth2 -y \
    pytorch=2.5.1 \
    pytorch-cuda=12.1 \
    torchvision=0.20.1 \
    torchaudio=2.5.1 \
    cudatoolkit \
    xformers \
    -c pytorch -c nvidia -c xformers
check_status "critical"

# Install conda build packages
echo -e "\n[STEP] ${YELLOW}Installing conda build packages...${NC}"
conda install -n unsloth2 -y \
    cmake \
    ninja \
    gcc \
    gxx \
    pybind11 \
    setuptools \
    scikit-learn \
    matplotlib \
    ipywidgets \
    ipykernel \
    -c conda-forge
check_status

# Install pip packages
echo -e "\n[STEP] ${YELLOW}Installing pip packages...${NC}"
conda run -n unsloth2 pip install \
    unsloth \
    setuptools_scm \
    wheel \
    build \
    tqdm \
    scipy==1.13.1 \
    pandas \
    trl \
    peft \
    wandb \
    chardet \
    openpyxl
check_status

# Clone and install vllm from source
echo -e "\n[STEP] ${YELLOW}Installing vllm from source...${NC}"
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

conda run -n unsloth2 bash -c "
    set -x  # Print commands as they execute for debugging
    export VLLM_INSTALL_PUNICA_KERNELS=1
    export TORCH_CUDA_ARCH_LIST='12.1'
    
    # Clone vllm into the temporary directory
    echo 'Cloning vllm at the commit by @oteroantoniogom...'
    git clone https://github.com/vllm-project/vllm.git '$TEMP_DIR/vllm' || { echo '${RED}Error cloning vllm${NC}'; exit 1; }
    cd '$TEMP_DIR/vllm'

    # Checkout a specific commit
    git checkout 5d8e1c9279678b3342d9618167121e758ed00c05 || { echo '${RED}Error checking out commit${NC}'; exit 1; }

    echo 'Detecting existing PyTorch installation...'
    python use_existing_torch.py || { echo '${RED}Error in use_existing_torch.py${NC}'; exit 1; }

    pip install -r requirements/build.txt || { echo '${RED}Error installing vllm build dependencies${NC}'; exit 1; }
    pip install -r requirements/common.txt || { echo '${RED}Error installing vllm common dependencies${NC}'; exit 1; }

    echo 'Installing vllm...'
    # Determine the number of available cores and set MAX_JOBS to cores-1 (or 1 if only one core is available)
    CORES=\$(nproc)
    if [ \"\$CORES\" -gt 1 ]; then
        MAX_JOBS=\$((CORES - 1))
    else
        MAX_JOBS=1
    fi

    echo \"Using MAX_JOBS=\${MAX_JOBS}\"

    # Use MAX_JOBS for installing vllm
    MAX_JOBS=\${MAX_JOBS} pip install -e . --no-build-isolation || { echo \"\${RED}Error installing vllm\${NC}\"; exit 1; }

    cd '$TEMP_DIR'
"
check_status

# Uninstall PyTorch-triton
echo -e "\n[STEP] ${YELLOW}Managing Triton...${NC}"
conda run -n unsloth2 pip uninstall -y pytorch-triton
check_status

# Install custom Triton if needed
echo -e "\n[STEP] ${YELLOW}Installing custom Triton...${NC}"
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"
conda run -n unsloth2 bash -c "
    set -x  # Print commands as they execute for debugging
    
    # Clone the Triton repository into the temporary directory
    echo 'Cloning Triton from GitHub on patch-1 branch...'
    git clone --branch patch-1 https://github.com/oteroantoniogom/triton.git '$TEMP_DIR/triton' || { echo '${RED}Error cloning Triton${NC}'; exit 1; }

    cd '$TEMP_DIR/triton'

    # Update any necessary submodules
    git submodule update --init --recursive || { echo '${RED}Error updating Triton submodules${NC}'; exit 1; }

    # Install needed dependencies
    echo 'Installing Triton dependencies...'
    conda install -y ninja cmake || true
    pip install wheel pybind11 ipywidgets ipykernel chardet openpyxl wandb || { echo '${RED}Error installing Triton dependencies${NC}'; exit 1; }

    # Install Triton from source
    echo 'Building and installing Triton...'
    pip install -e python || { echo '${RED}Error installing Triton from source${NC}'; exit 1; }

    cd '$TEMP_DIR'
"
check_status

# Clean up temp directory
echo -e "\n[STEP] ${YELLOW}Cleaning up...${NC}"
rm -rf "$TEMP_DIR"
check_status

echo -e "\n${GREEN}Installation process completed successfully!${NC}"

# Verify installations
echo -e "\n${BLUE}==============================================${NC}"
echo -e "${BLUE}   Verifying installations   ${NC}"
echo -e "${BLUE}==============================================${NC}"

# Verify PyTorch
echo -e "\n[STEP] ${YELLOW}Verifying PyTorch installation...${NC}"
conda run -n unsloth2 python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
"
check_status

# Verify xformers
echo -e "\n[STEP] ${YELLOW}Verifying xformers installation...${NC}"
conda run -n unsloth2 python -c "
import xformers
print('xformers is installed')
"
check_status

# Verify vllm
echo -e "\n[STEP] ${YELLOW}Verifying vllm installation...${NC}"
conda run -n unsloth2 python -c "
import vllm
print('vllm is installed')
"
check_status

# Verify unsloth
echo -e "\n[STEP] ${YELLOW}Verifying unsloth installation...${NC}"
conda run -n unsloth2 python -c "
import unsloth
print('unsloth is installed')
"
check_status

echo -e "\n${GREEN}All installations have been verified successfully!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "${GREEN}conda activate unsloth2${NC}"