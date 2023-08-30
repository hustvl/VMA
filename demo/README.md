# Inference Instruction

## Prepare checkpoints.

Download the checkpoints and place them as shown below.
You can download the checkpoints in [Baidu](https://pan.baidu.com/s/11fvTz4KmzVv-ohdZROtEJQ?pwd=v8yb)/[Google](https://drive.google.com/drive/folders/1j8foyUoFPjixoPMDqhAOpqwBIlLnFFZt?usp=drive_link).
```
VMA
├── ckpts
│   ├── sd_line.pth
│   ├── sd_box.pth
│   ├── sd_freespace.pth
```

## Inference our custom data.

You can run the aggregation scrpit as follows, the vectorized instances result (and visualization result) will be saved in out_dir.
We have provided sample image data in ```demo/image_data``` and ```demo/trajectory_data```.
```
python demo/cutsom_infer.py {root_directory} 
                            {config_path} \ 
                            {checkpoint_path} \ 
                            --trajectory_sample_num {trajectory_sample_num} \ 
                            --element_type all \ 
                            --out_dir {out_dir} \ 
                            --visualize False 
```
