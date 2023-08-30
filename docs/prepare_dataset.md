# NYC Dataset preparation
```
# File structure
VMA
├──data
│   ├── nyc
│   │   ├── cropped_tiff
│   │   ├── labels
│   │   ├── data_split.json
```

## Download Dataset and labels

Run 
```
# script to prepare tiff images
sh ./tools/icurb/get_data_new.bash

# script to prepare labels
# install gdown
pip install gdown
# download and unzip the label from Google Drive
sh .tools/icurb/get_label.bash
```
We directly download the labels and the raw aerial image data from NYC database. Then we conduct a set of processings to obtain tiff images that we need. It may take some time for download and processing.

In case the script fails, you can download and unzip the data manually [here](https://pan.baidu.com/s/1R90Hjroq-YWDiBVevxAD3w?pwd=ck1b). Then use the script in ```./tools/icurb``` to process them. 

## Dataset splitting
```./tools/icurb/data_split.json``` defines how the dataset is split into pretrain/train/valid/test. They are randomly split. It is recommended to use our provided data splitting file.

## Download saved pretrain checkpoints
We provide the remapped iCurb backbone of our implementations. You can download it in [Baidu](https://pan.baidu.com/s/12OEbGG2tVbQEAfxI84uYgw?pwd=mqkb)/[Google](https://drive.google.com/file/d/1ETNOpXzBYDswUv_w-r2cRqoouMWCszHk/view?usp=drive_link) and saved in ```./ckpts```. 

# SD Dataset preparation
We provide some sd dataset for train and eval, you can download them in [Baidu](https://pan.baidu.com/s/18XDCk2o_gOs4z4wtYZ9fbA?pwd=l971)/[Google](https://drive.google.com/file/d/1jMXu6hToU2IS8MUXsk3NtiAzSNwd2jK_/view?usp=drive_link) and place them as shown below.
```
# File structure
VMA
├── data
|   ├── sd_data
|   |   ├──line
|   |   |   ├──origin_data
|   |   |   |   ├──image_data
|   |   |   |   ├──trajectory_data
|   |   |   |   ├──line_6k_data.json
|   |   ├──box
|   |   ├──freespace
```
## Process SD Dataset
Use the script in ```./tools/custom``` to crop the orginal 6k data. The cropped data and annotation file will be generated in ```sd_data/line/cropped_data```. Then you can train these data with the configs

