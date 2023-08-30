from argparse import ArgumentParser
from os.path import dirname
from mmcv import Config
from utils import str2bool
from inference_and_combine_function import driving_box_infer_and_save, driving_line_infer_and_save, driving_freespace_infer_and_save
import os
import sys
sys.path.append('')

def get_args():
    parser = ArgumentParser()
    parser.add_argument('root_path', help='Input Driving Site Path or Parking Pack Path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--trajectory_sample_num', default=15, type=int, help='Sampled point nums in one trajectory')
    parser.add_argument('--element_type', default='line', choices=['box', 'line', 'freespace', 'all'], help='Element type')
    parser.add_argument('--out_dir', default='./out', help='Out Path')
    parser.add_argument('--visualize', default=False, type=str2bool, help='Whether to visualize result')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

infer_combine_func = {
    'line':driving_line_infer_and_save,
    'box':driving_box_infer_and_save,
    'freespace':driving_freespace_infer_and_save,
}

def main():
    args = get_args()
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    data_list = os.listdir(args.root_path)
    data_list.sort()    
    infer_combine_function = infer_combine_func[args.element_type]
    infer_combine_function(args, data_list)
    print('completed')

if __name__ == '__main__':
    main()