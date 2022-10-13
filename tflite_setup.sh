#!/bin/bash
#========================================================================================================================================
#
#         FILE: tflite_setup.sh
#        USAGE: ./tflite_setup.sh
#  DESCRIPTION: Setup tensorflow with tflite_model_maker, including NVIDIA GPU support on a fresh bare-metal server
#     OPTINONS:
# REQUIREMENTS: git curl python3 python3-pip nvidia-smi
#       AUTHOR: Zero
#      CREATED:
#========================================================================================================================================

function dependency_check() {
    for i in "${@}"; do
        command -v $i > /dev/null
        if [ $? -ne 0 ]; then
            echo "$i dependency not found..."
            echo "attempting installation..."
            apt-get install $i #2&> /dev/null
            if [ $? -ne 0 ]; then
              echo "error installing: $i"
              return $?
            fi
            echo "installed $i successfully"
        fi
        echo
    done
}

#apt-get update
dependency_check git curl python3 python3-pip nvidia-smi || exit

# Fetch miniconda for easy cuda installation
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create --name tf python=3.9
conda activate tf

# Install tflite_model_maker, comes bundled with tensorflow
git clone https://github.com/tensorflow/examples
pip install -r examples/tensorflow_examples/lite/model_maker/requirements.txt
pip install -e examples/tensorflow_examples/lite/model_maker/pip_package/

# Uninstall scann -> the default bundled installation with tflite causes problem (symbol table error)
pip uninstall scann
# Reinstall
pip install scann

# Install CUDA and cuDNN with conda
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Configure system paths
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Install CUDA developer keys, for additional cuda shared libraries missed by conda (fixes libnvinfer.so error)
# Uncomment the following line for Debian 10, and comment the next
#apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub

# Add CUDA Debian repository
#add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
add-apt-repository contrib
apt update
apt-get install cuda

deactivate
source ~/.bashrc
conda activate tf

# Verify GPU setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo "conda environment (tf) activate"
