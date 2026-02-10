#!/bin/zsh

# Check if unsloth environment exists and delete it if found
if conda info --envs | grep -q "^unsloth "; then
    echo "Found existing 'unsloth' environment. Removing it..."
    conda env remove -n unsloth -y
else
    echo "No existing 'unsloth' environment found."
fi

# Create new conda environment
echo "Creating new 'unsloth' environment..."

conda create --name unsloth \
        python=3.11 \
        cudnn=9.8.0 \
        cuda-version=12.1 \
        pytorch-cuda=12.1 \
        pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers -c pytorch -c nvidia -c xformers \
        -y

conda init 
conda activate unsloth

conda install -n unsloth -c conda-forge ipywidgets -y


# Install packages using conda run (safer in scripts than direct activation)
echo "Installing required packages..."
conda run -n unsloth pip install setuptools setuptools_scm wheel build  
conda run -n unsloth conda install cmake -y

conda run -n unsloth pip install unsloth --no-deps
conda run -n unsloth pip install triton==3.1.0 --no-deps
conda run -n unsloth pip install vllm==0.8.2
conda run -n unsloth pip install numpy==1.26.4

conda run -n unsloth pip install vllm
conda run -n unsloth pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

conda run -n unsloth pip install ipykernel numpy tqdm scipy==1.13.1 pandas scikit-learn trl peft
# conda run -n unsloth pip install git+https://github.com/vllm-project/vllm.git --no-build-isolation


echo "Setup complete! Use 'conda activate unsloth' to activate the environment."