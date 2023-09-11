# Prerequisites

**Please ensure you have prepared the environment and the NYC dataset.**

# Train and Test

Train VMA with 8 GPUs.
```
./tools/dist_train.sh ./projects/configs/vma/vma_icurb.py 8
```

Eval VMA with 8 GPUs

You can download our trained iCurb checkpoint in [Baidu](https://pan.baidu.com/s/1up5Atj7TsTt8i6jTqG4VcA?pwd=6hs1)/[Google](https://drive.google.com/file/d/1lCr7f4nFMbUKC4AD2vnfnvqjffrQ9qxZ/view?usp=drive_link).

We support metric **icurb** for NYC dataset and **chamfer** for SD dataset.
```
./tools/dist_test.sh ./projects/configs/vma/vma_icurb.py ./path/to/ckpts.pth 8
```