import mmcv
import numpy as np
from PIL import Image
import torch
from mmdet.datasets.builder import PIPELINES
import torchvision.transforms.functional as F

@PIPELINES.register_module()
class LoadImageFromFiles(object):
    """Load images from a file.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, padding=True,pad_val=128, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.pad_val = pad_val

    def __call__(self, results):
        
        filename = results['img_filename']
        img = Image.open(filename)
        img = F.to_tensor(img).numpy()
        img_shape = img.shape
        h = img_shape[1]
        w = img_shape[2]

        size = (h, w)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = img
        results['img_shape'] = size
        results['ori_shape'] = size
        # Set initial values for default meta_keys
        results['pad_shape'] = size
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[0]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class NormalizeImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img = results['img'].transpose(1, 2, 0)
        results['img'] = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb).transpose(2, 0, 1)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
    
@PIPELINES.register_module()
class PadChannel(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img = results['img']
        padding_channel = np.expand_dims(np.zeros((img.shape[1], img.shape[2])), axis=0)
        img = np.concatenate([img, padding_channel])
        results['img'] = img
        return results
