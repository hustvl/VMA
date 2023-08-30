# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n vma python=3.8 -y
conda activate vma
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**d. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**e. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**f. Clone VMA.**
```
git clone https://github.com/hustvl/VMA.git
```

**g. Install mmdet3d**
```shell
cd /path/to/VMA/mmdetection3d
python setup.py develop
```

**h. Install other requirements.**
```shell
cd /path/to/VMA
pip install -r requirements.txt
```

**i. Prepare pretrained models.**

We use iCurb backbone provided in [iCurb](https://arxiv.org/abs/2103.17118) to train NYC Planimetric Database and resnet-152 to train our custom data.
```shell
cd /path/to/VMA
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
```
You can download our remapped iCurb backbone in [Baidu](https://pan.baidu.com/s/12OEbGG2tVbQEAfxI84uYgw?pwd=mqkb)/[Google](https://drive.google.com/file/d/1ETNOpXzBYDswUv_w-r2cRqoouMWCszHk/view?usp=drive_link).