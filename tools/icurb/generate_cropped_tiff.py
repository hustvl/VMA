import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse

def crop_tiff(jp2_name, raw_data_dir, out_dir):
    with open('./tools/icurb/data_split.json','r') as jf:
        json_data = json.load(jf)
    tiff_list = json_data['train'] + json_data['valid'] + json_data['test'] + json_data['pretrain']
    raw_tiff = np.array(Image.open(os.path.join(raw_data_dir, jp2_name)))
    for ii in range(5):
        for jj in range(5):
            cropped_tiff_name = f'{jp2_name[:-4]}_{ii}{jj}'
            if cropped_tiff_name in tiff_list:
                Image.fromarray(raw_tiff[1000*ii:1000*(ii+1),1000*jj:1000*(jj+1)]).save(os.path.join(out_dir, f'{cropped_tiff_name}.tiff'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir',required=True, type=str, help ='The directory to store temp raw tiff data')
    parser.add_argument('--out_dir',required=True, type=str, help ='The directory to save cropped tiff data')
    args = parser.parse_args()

    jp2_list = os.listdir(args.raw_data_dir)
    jp2_list = [x for x in jp2_list if x[-3:]=='jp2']
    with multiprocessing.Pool(processes = 4) as pool:
        func = partial(crop_tiff, raw_data_dir=args.raw_data_dir, out_dir=args.out_dir)
        result = list(tqdm(pool.imap(func, jp2_list), total=len(jp2_list)))