import os 
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms.functional as F
import random
import numpy as np
from mmdet.datasets import DATASETS
import torch
from mmdet.datasets.pipelines import to_tensor
from shapely.geometry import LineString, box, MultiLineString
from os import mkdir, path as osp
import mmcv
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
import math
from collections.abc import Sequence
import numbers
from torch import Tensor
from shapely.geometry.collection import GeometryCollection
import torchvision
import cv2 as cv
import tempfile
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines import Compose

class InstanceLines(object):

    def __init__(self, 
                 map_classes,
                 attrs_dict, 
                 instance_line_list,
                 instance_labels,
                 instance_attrs,
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
        
        self.map_classes = map_classes
        self.attrs_dict = attrs_dict
        self.instance_list = instance_line_list
        self.instance_labels = instance_labels
        self.instance_attrs = instance_attrs

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
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance, instance_class, instance_attrs in zip(self.instance_list, self.instance_labels, self.instance_attrs):
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
                    # import pdb;pdb.set_trace()
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                if instance_class != self.map_classes.index('lane') and instance_attrs[list(self.attrs_dict.keys()).index('lane_direction')] != self.attrs_dict['lane_direction'].index('unidirectional'):
                    flip_sampled_points = np.flip(sampled_points, axis=0)
                    shift_pts_list.append(sampled_points)
                    shift_pts_list.append(flip_sampled_points)
                else:
                    shift_pts_list.append(sampled_points)
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
                padding = torch.full([final_shift_num - multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
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

    def gen_vectorized_samples(self, instances, attrs_dict):
        vectors = []
        for instance in instances:
            instance_class = instance['class']
            instance_data = instance['data']
            instance_attrs = instance['attrs']

            if instance_class == 'lane':
                # import pdb;pdb.set_trace()
                attr = [attrs_dict['lane_direction'].index(instance_attrs['lane_direction']), \
                        attrs_dict['lane_type'].index(instance_attrs['lane_type']), \
                        attrs_dict['lane_properties'].index(instance_attrs['lane_properties']), \
                        attrs_dict['lane_flag'].index(instance_attrs['lane_flag']), \
                        attrs_dict['lane_width'].index(instance_attrs['lane_width']),
                        len(attrs_dict['curb_type'])]
            elif instance_class == 'curb':   
                attr = [len(attrs_dict['lane_direction']), \
                        len(attrs_dict['lane_type']), \
                        len(attrs_dict['lane_properties']), \
                        len(attrs_dict['lane_flag']), \
                        len(attrs_dict['lane_width']), \
                        attrs_dict['curb_type'].index(instance_attrs['curb_type'])]
            else:
                attr = [len(attrs_dict['lane_direction']), \
                        len(attrs_dict['lane_type']), \
                        len(attrs_dict['lane_properties']), \
                        len(attrs_dict['lane_flag']), \
                        len(attrs_dict['lane_width']), \
                        len(attrs_dict['curb_type'])]
            vectors.append((LineString(np.array(instance_data)), self.vec_classes.index(instance_class), attr)) 
        gt_labels = []
        gt_instances = []
        gt_attrs = []
        for instance, instance_type, instance_attr in vectors:
            if instance_type != -1:
                gt_instances.append(instance)
                gt_labels.append(instance_type)
                gt_attrs.append(instance_attr)
        gt_instances = InstanceLines(self.vec_classes, attrs_dict, gt_instances, gt_labels, gt_attrs, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instances,
            gt_vecs_label=gt_labels,
            gt_vecs_attr=gt_attrs,
        )
        return anns_results
    
@DATASETS.register_module()
class SD_Driving_Line_Dataset(Dataset):
    r'''
    DataLoader for sampling. Iterate the aerial images dataset
    '''
    def __init__(self, 
                 data_root, 
                 sub_dir,
                 annotation_file, 
                 mask_dir, 
                 points_nums, 
                 map_classes=None,
                 attrs_dict=None, 
                 eval_use_same_gt_sample_num_flag=False,
                 pipeline=None,
                 mode="valid", 
                 test_mode=False):
        
        assert mode in {"train", "test", "valid"}
        
        self.MAPCLASSES = self.CLASSES = self.get_map_classes(map_classes)
        self.attrs_dict = attrs_dict
        self.data_root = data_root
        self.sub_dir = sub_dir
        annotation_dict = self.load_datadir(annotation_file, mode)
        self.annotation_dict = annotation_dict
        self.mask_dir = mask_dir
        self.seq_len = len(annotation_dict)
        self.points_nums = points_nums
        self.padding_value = -10000   
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag
        self.test_mode = test_mode
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
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self,idx):
        return self.load_train_data(**list(self.annotation_dict.values())[idx])
    
    def prepare_test_data(self, idx):
        return self.load_test_data(**list(self.annotation_dict.values())[idx])

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_datadir(self, annotation_file, mode):
        annotation_file = os.path.join(self.data_root, annotation_file)
        with open(annotation_file,'r') as jf:
            json_dict = json.load(jf)

        if mode == 'valid':
            json_list = json_dict['valid']
        elif mode == 'test':
            json_list = json_dict['valid']
        else:
            json_list = json_dict['train']

        annotation_dict={}
        for img_ann in json_list:
            annotation_dict.update({img_ann['image_name']:{'image_path':osp.join(self.data_root, self.sub_dir, img_ann['image_name']),'instances':img_ann['instances']}})
        return annotation_dict
    
    def load_train_data(self, image_path, instances):
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
        anns_results = vectormap.gen_vectorized_samples(instances, self.attrs_dict)
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        gt_vecs_attr = to_tensor(anns_results['gt_vecs_attr']).permute(1, 0)
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
        example['gt_attrs'] = DC(gt_vecs_attr, cpu_only=False)

        return example

    def load_test_data(self, image_path, instances):
        # load image
        example = self.pipeline({'img_filename':image_path})
        # load vectormap
        example = self.vectormap_pipeline(example, instances)
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
        for result in results:
            pred_instances = []
            pred_scores = result['scores_3d'].numpy()
            pred_data = result['pts_3d'].numpy()
            pred_labels = result['labels_3d'].numpy()
            pred_attrs_label = torch.stack(result['attrs_3d']['attrs_preds']).transpose(1,0)
            for idx in range(len(pred_scores)):
                pred_instances.append({'class':pred_labels[idx], \
                                        'data':pred_data[idx], \
                                        'attrs':pred_attrs_label[idx].tolist(), \
                                        'confidence_level':pred_scores[idx]})
            image_path = result['img_metas']['filename']
            image_name = image_path.split('/')[-1]
            results_dict.update({image_name:{'image_path':image_path, 'pred_instances':pred_instances}})
        res_path = osp.join(jsonfile_prefix, 'results_sd.json')
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
        assert isinstance(results, list), 'results must be a list'
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
 
    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from projects.mmdet3d_plugin.datasets.map_utils.sd_line_evaluate import eval_map, format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        
        print('Formating results & gts by classes')
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        gen_results, annotations = self.arrange_results_and_annotations(pred_results, self.annotation_dict)
        detail = dict()
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.points_nums,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag)
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        
        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                thresholds = [6, 15, 30]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))
            # import pdb;pdb.set_trace()
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                attrs_dict=self.attrs_dict,
                                logger=logger,
                                num_pred_pts_per_instance=self.points_nums,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['SD_Line_Map_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['SD_Line_Map_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['SD_Line_Map_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['SD_Line_Map_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail

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
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        metric = metric if isinstance(metric, list) else [metric]
        if metric == ['chamfer']:
            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name], metric=metric)
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files, metric=metric)
        else:
            raise NotImplementedError
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show:
            if not osp.exists(show_dir):
                mkdir(show_dir)
            self.show(result_files['pts_bbox'], show_dir)
        return results_dict
    
    def show(self, results_file, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        with open(results_file, 'r') as f:
            results = json.load(f)
        for result in tqdm(results.values(), desc='show the visualization result'):
            image_path = result['image_path']
            pred_instances = result['pred_instances']
            self.show_result_with_original_image(pred_instances, out_dir, image_path)
        print('visualization result has been stored in {}'.format(out_dir))

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

    def arrange_results_and_annotations(self, results, annotations):
        # this function is to arrange one big class together
        all_image_arranged_pred_results = []
        all_image_arranged_gt_annotations = []
        for pred_result, gt_annotation in zip(results.values(), annotations.values()):
            single_image_pred_results = []
            single_image_gt_annotations = []
            pred_instances = pred_result['pred_instances']
            for instance in pred_instances:
                instance_data = [[int(x[0]), int(x[1])] for x in instance['data']]
                single_image_pred_results.append({'pts':instance_data,'type':instance['class'], 'attrs':instance['attrs'], 'confidence_level':instance['confidence_level']})
            gt_instances = gt_annotation['instances']
            for instance in gt_instances:
                if instance['class'] == 'lane':
                    instance_all_attrs = instance['attrs']
                    instance_attrs = [self.attrs_dict['lane_direction'].index(instance_all_attrs['lane_direction']),\
                                      self.attrs_dict['lane_type'].index(instance_all_attrs['lane_type']), \
                                      self.attrs_dict['lane_properties'].index(instance_all_attrs['lane_properties']), \
                                      self.attrs_dict['lane_flag'].index(instance_all_attrs['lane_flag']), \
                                      self.attrs_dict['lane_width'].index(instance_all_attrs['lane_width'])]
                    single_image_gt_annotations.append({'pts':instance['data'],'type':self.MAPCLASSES.index(instance['class']), 'attrs':instance_attrs})
                elif instance['class'] == 'curb':
                    instance_all_attrs = instance['attrs']
                    instance_attrs = [self.attrs_dict['curb_type'].index(instance_all_attrs['curb_type'])] if 'curb_type' in instance_all_attrs.keys() else [self.attrs_dict['curb_type'].index(instance_all_attrs['type'])]
                    single_image_gt_annotations.append({'pts':instance['data'],'type':self.MAPCLASSES.index(instance['class']), 'attrs':instance_attrs})
                else:
                    single_image_gt_annotations.append({'pts':instance['data'],'type':self.MAPCLASSES.index(instance['class']), 'attrs':[]})
            all_image_arranged_pred_results.append(single_image_pred_results)
            all_image_arranged_gt_annotations.append(single_image_gt_annotations)
        return all_image_arranged_pred_results, all_image_arranged_gt_annotations

    def show_result_with_original_image(self,
                                        pred_instances,
                                        out_dir,
                                        image_path):
        image_name = image_path.split('/')[-1]
        image = cv.imread(image_path)
        color = {0:(0, 0, 0), 1:(0, 0, 255), 2:((0, 97,255))}
        
        # drew_image_path = out_dir + '/' + image_name
        # cv.imwrite(drew_image_path, image)
        for pred_instance in pred_instances:
            # import pdb;pdb.set_trace()
            sub_pred_points = pred_instance['data']
            pred_class = pred_instance['class']
            
            # import pdb;pdb.set_trace()
            color_fill = color[pred_class]   
            sub_pred_points = [tuple([int(x[0]), int(x[1])]) for x in sub_pred_points]
            if pred_class == 2:
                for i in range(1, len(sub_pred_points)):
                    cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
            else:
                for i in range(len(sub_pred_points)):
                    if i == 0:
                        cv.circle(image, sub_pred_points[i], 10, (0,97,255), -1)
                        continue
                    elif i == len(sub_pred_points)-1:
                        cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
                        cv.circle(image, sub_pred_points[i], 8, (0,255,255), -1)
                    else:
                        cv.line(image, sub_pred_points[i-1], sub_pred_points[i], color_fill, 4)
                        cv.circle(image, sub_pred_points[i], 5, (0, 255, 0), -1)
        original_image = cv.imread(image_path)
        drew_image = cv.copyMakeBorder(image, 0, 0, 50, 0, cv.BORDER_CONSTANT, value=(128,128,128))
        original_image = cv.imread(image_path)
        concat_image = cv.hconcat([original_image, drew_image])
        drew_image_path = out_dir + '/' + image_name
        cv.imwrite(drew_image_path, concat_image)