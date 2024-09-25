# 🌋 VM setup
Image - Ubuntu Server 24.04 LTS - x64 Gen2
Size - Standard_NC24ads_A100_v4 - 24 vcpus, 220 GiB memory
Disk - 1TiB (P30) SSD

# Configure command
## Install nvidia driver (https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
```Shell
sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-5
sudo reboot
```

## Add env var (add below exports at the end of ~/.bashrc)
```Shell
vim ~/.bashrc 
# export PATH="/usr/local/cuda-12.5/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH"
```

## Install miniconda (https://docs.anaconda.com/miniconda/)
```Shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
bash
```

## Clone repo
```Shell
git clone https://github.com/gpminsuk/LLaVA-NeXT llava-next
cd llava
```

## Configure conda and install requirements (https://medium.com/@prayto001/how-to-train-your-own-vision-large-language-model-37e3ff82b0b7)
```Shell
conda create -n llava-next python=3.10 -y
conda activate llava-next
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Download training data for MLP
```Shell
sudo apt install unzip
mkdir blip_laion
cd blip_laion
wget -O images.zip "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true"
unzip images.zip
rm images.zip
cd ..
wget -O blip_laion_cc_sbu_558k.json "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true"
```