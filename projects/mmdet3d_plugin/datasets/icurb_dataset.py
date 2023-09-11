from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms.functional as F
import random
import numpy as np
from mmdet.datasets import DATASETS
import torch
from mmdet.datasets.pipelines import to_tensor
from shapely.geometry import LineString, box, MultiLineString, Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from os import mkdir, path as osp
import mmcv
from PIL import Image, ImageDraw
# from mmdetection3d.mmdet3d.core import show_result
from tqdm import tqdm
from projects.mmdet3d_plugin.datasets.map_utils.icurb_evaluate import evaluate_all
from collections.abc import Sequence
import numbers
from torch import Tensor
import math
import torchvision
from torchvision.transforms.functional import InterpolationMode
import cv2 as cv
import tempfile
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
class InstanceLines(object):

    def __init__(self, 
                 instance_line_list,
                 instance_labels,
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1]
        self.max_y = self.patch_size[0]
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list
        self.instance_labels = instance_labels

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
                                dtype=torch.float32)
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
                            dtype=torch.float32)
        instance_bbox_tensor[:,0] = torch.clamp(instance_bbox_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,1] = torch.clamp(instance_bbox_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_bbox_tensor[:,2] = torch.clamp(instance_bbox_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,3] = torch.clamp(instance_bbox_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_bbox_tensor
    
    @property
    def origin_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            sampled_points = np.array(list(instance.coords)).reshape(-1, 3)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for idx, instance in enumerate(self.instance_list):
            instance_label = self.instance_labels[idx]
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                    shift_sampled_points[:, 0] /= self.max_x # normalize
                    shift_sampled_points[:, 1] /= self.max_y
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                sampled_points[:, 0] /= self.max_x
                sampled_points[:, 1] /= self.max_y
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=0,max=0.999)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=0,max=0.999)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

class VectorizedLocalMap(object):
    def __init__(self,
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000,
                 ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()

        self.vec_classes = map_classes
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value
        self.patch_size = patch_size

    def gen_vectorized_samples(self, instances):
        vectors = []
        for instance in instances:
            vectors.append((LineString(np.array(instance)), 0)) 
        gt_labels = []
        gt_instance = []
        for instance, instance_type in vectors:
            if instance_type != -1:
                gt_instance.append(instance)
                gt_labels.append(instance_type)
        gt_instance = InstanceLines(gt_instance, gt_labels, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
        )
        return anns_results


@DATASETS.register_module()
class iCurb_Dataset(Dataset):
    r'''
    DataLoader for sampling. Iterate the aerial images dataset
    '''
    CLASSES = ('curb')
    def __init__(self, 
                 split_file,
                 seq_dir, 
                 image_dir, 
                 mask_dir, 
                 points_nums, 
                 map_classes=None, 
                 pipeline=None,
                 mode="valid", 
                 test_mode=False):
        
        assert mode in {"train", "valid", "test"}
        
        self.MAPCLASSES = self.get_map_classes(map_classes)
        annotation_dict = self.load_datadir(split_file, seq_dir, image_dir, mode)
        self.annotation_dict = annotation_dict
        self.seq_len = len(annotation_dict)
        self.points_nums = points_nums
        self.padding_value = -10000
        self.mask_dir = mask_dir
        self.total_acc = 0
        self.total_recall = 0
        self.total_r_f = 0
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if not test_mode:
            self._set_group_flag()
    
    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len
    
    def __getitem__(self, idx):
        example = self.load_data(**list(self.annotation_dict.values())[idx])
        return example

    def load_datadir(self, split_file, seq_path, image_path, mode):
        with open(split_file,'r') as jf:
            json_list = json.load(jf)
        train_list = json_list['train'] + json_list['pretrain']
        test_list = json_list['valid']
        val_list = json_list['valid']

        if mode == 'valid':
            json_list = [x+'.json' for x in val_list]
        elif mode == 'test':
            json_list = [x+'.json' for x in test_list]
        else:
            json_list = [x+'.json' for x in train_list]

        annotation_dict={}
        for jsonf in json_list:
            with open(osp.join(seq_path, jsonf), 'r') as f:
                seq_list = json.load(f)
            instances= []
            for area in seq_list: 
                instances.append(area['seq'])
            annotation_dict.update({jsonf:{'image_path':osp.join(image_path, jsonf[:-4]+'tiff'),'instances':instances}})
        return annotation_dict   

    def load_data(self, image_path, instances):

        # load image
        example = self.pipeline({'img_filename':image_path})
        # load vectormap
        example = self.vectormap_pipeline(example, instances)
        return example

    def vectormap_pipeline(self, example, instances):
        vectormap = VectorizedLocalMap(patch_size=example['img_metas'].data['img_shape'], 
                                        map_classes=self.MAPCLASSES, 
                                        fixed_ptsnum_per_line=self.points_nums,
                                        padding_value=self.padding_value,)
        anns_results = vectormap.gen_vectorized_samples(instances=instances)
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], InstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes'] = DC(gt_vecs_pts_loc, cpu_only=True)

        return example

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        print('Start to convert detection format...')
        results_dict = {}
        # import pdb;pdb.set_trace()
        for result in results:
            pred_instances = []
            pred_scores = result['scores_3d'].numpy()
            pred_data = result['pts_3d'].numpy()
            pred_labels = result['labels_3d'].numpy()
            for idx in range(len(pred_scores)):
                pred_instances.append({'class':pred_labels[idx], 'data':pred_data[idx], 'confidence_level':pred_scores[idx]})
            image_path = result['img_metas']['filename']
            image_name = image_path.split('/')[-1]
            results_dict.update({image_name:{'image_path':image_path, 'pred_instances':pred_instances}})
        res_path = osp.join(jsonfile_prefix, 'results_icurb.json')
        print('Results writes to', res_path)
        
        mmcv.dump(results_dict, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                # not evaluate 2D predictions on nuScenes
                if '2d' in name:
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 show_dir=None,
                #  pipeline=None,
                #  img_metas=None
                 ):
        metric = metric if isinstance(metric, list) else [metric]
        if metric == ['icurb']:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            results_dict = evaluate_all(result_files['pts_bbox'], self.mask_dir)
        else:
            raise NotImplementedError
        if show:
            if not osp.exists(show_dir):
                mkdir(show_dir)
            self.show(results, show_dir)
        return results_dict
    
    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        # pipeline = self._get_pipeline(pipeline)
        for result in tqdm(results, desc='show the visualization result'):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            image_path = result['img_metas']['filename']
            pred_points = result['pts_3d']
            show_result(pred_points, out_dir, image_path)
        print('visualization results has been stored')

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')
        
        return class_names
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

def show_result(pred_points,
                out_dir,
                image_path):
    image_path = image_path.replace('tiff', 'jpg')
    image_name = image_path.split('/')[-1]
    image = cv.imread(image_path)
    for idx in range(len(pred_points)):
        sub_pred_points = pred_points[idx].numpy().tolist()
        sub_pred_points = [tuple([int(float(x[1])), int(float(x[0]))]) for x in sub_pred_points]
        for i in range(len(sub_pred_points)):
            if i == 0:
                cv.circle(image, sub_pred_points[i], 5, (0,255,0), -1)#中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)s
                continue
            else:
                cv.line(image, sub_pred_points[i-1], sub_pred_points[i], (0, 0, 255), 2)
            cv.circle(image, sub_pred_points[i], 5, (0,255,0), -1)#中心坐标,半径,颜色(BGR),线宽(若为-1,即为填充颜色)s
    drew_image_path = out_dir + '/' + image_name
    cv.imwrite(drew_image_path, image)
