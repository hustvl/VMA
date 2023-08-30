import mmcv
import torch
import torchvision.transforms.functional as F
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from combine_and_save_driving_line import combine_results_and_save_driving_line
from combine_and_save_driving_box import combine_results_and_save_driving_box
from combine_and_save_driving_freespace import combine_results_and_save_driving_freespace
import numpy as np
import os
from shapely.geometry import LineString

def get_sub_img_driving(img, trajectory_data_list, sample_num):
    sub_img_list = []
    left_top_list = []
    
    # 对每条轨迹采样sample_num个点作为中心点在大图上进行裁切
    for i, trajectory_data in enumerate(trajectory_data_list):
        if len(trajectory_data) < 2:
            print('trajectory {} is not long enough'.format(i))
            continue
        trajectory_shapely = LineString(np.array(trajectory_data))
        # trajectory_length = trajectory_shapely.length
        # sample_num = round((trajectory_length // 1000)*2 + 1)
        distances = np.linspace(0, trajectory_shapely.length, sample_num)
        sampled_points = np.array([list(trajectory_shapely.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        for sampled_point in sampled_points:
            x, y = sampled_point
            if x < 500:
                x = 500
            elif x > 5500:
                x = 5500
            if y < 500:
                y = 500
            elif y > 5500:
                y = 5500
            bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = round(x - 500), round(y - 500), round(x + 499), round(y + 499)
            sub_img_list.append(img[:, bbox_miny:bbox_maxy+1, bbox_minx:bbox_maxx+1])  # (left, upper, right, lower)
            left_top_list.append([bbox_minx, bbox_miny])
    return sub_img_list, left_top_list

def get_sub_data_driving(data_dict, trajectory_data_list, sample_num):
    sub_datas = []
    img = data_dict['img']
    img_name = data_dict['filename']
    sub_imgs, left_top = get_sub_img_driving(img, trajectory_data_list, sample_num)
    for i in range(len(sub_imgs)):
        sub_data_dict = dict()
        sub_data_dict['img_metas'] = data_dict['img_metas']
        sub_data_dict['img_metas'][0]['img_shape'] = (sub_imgs[i].shape[1], sub_imgs[i].shape[2])
        sub_data_dict['img'] = sub_imgs[i]
        sub_datas.append(sub_data_dict)
    return sub_datas, left_top

def pad(img, target_size=(6000, 6000, 3), pad_value=(0, 0, 128)):
    H, W, C = img.shape
    assert C == 3
    if H == W == 6000:
        return img
    print('\npadding: {} -> {}\n'.format(tuple(img.shape), target_size))
    new_img = np.ones(target_size, dtype=img.dtype) * np.array(pad_value)[None, None, :]
    new_img[:H, :W, :] = img
    return new_img.astype(img.dtype)

class LoadBigImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img_intensity = mmcv.imread(results['img'], channel_order='rgb')
        img_intensity = pad(img_intensity)
        img = F.to_tensor(img_intensity)
        mean= torch.tensor([0.485, 0.456, 0.406])
        std= torch.tensor([0.229, 0.224, 0.225])
        img = (img - mean.reshape((3, 1, 1))) / std.reshape(3, 1, 1)
        assert img.shape[1] == img.shape[2], img.shape
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = 0
        results['scale_factor'] = 1.0
        results['img_metas'] = [{}]
        return results
    
class LoadBigImagePad:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img_intensity = mmcv.imread(results['img'], channel_order='rgb')
        img_intensity = pad(img_intensity)
        img = F.to_tensor(img_intensity)
        img_shape = (img.shape[1], img.shape[2])
        padding_channel = torch.zeros(*img_shape).unsqueeze(0)
        img = torch.cat([img, padding_channel])
        assert img.shape[1] == img.shape[2], img.shape
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = 0
        results['scale_factor'] = 1.0
        results['img_metas'] = [{}]
        return results

def inference_detector_forcurb_bigimg(model, get_sub_data_func, imgs, trajectory_path, sample_num, get_traj_pad=False, pad=False, attr=False):

    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if pad:
        test_pipeline = [LoadBigImagePad()]
    else:
        test_pipeline = [LoadBigImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=imgs)
    data = test_pipeline(data)
    ## add get_sub_img func
    trajectory_data_list = mmcv.load(trajectory_path)
    sub_datas, left_top = get_sub_data_func(data, trajectory_data_list, sample_num)
    if get_traj_pad:
        from utils import TrajPadSampler
        pad_sampler =  TrajPadSampler(imgs, left_top)
        more_sub_datas, more_left_top = pad_sampler.sample(data)
        sub_datas += more_sub_datas
        left_top += more_left_top
    if not len(sub_datas):
        return [], []
    results = []
    for sub_data in sub_datas:
        data = collate([sub_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            results.append(result)

    print('Start to convert detection format...')
    results_list = []
    if not attr:
        for batch_result in results:
            for result in batch_result:
                result = result['pts_bbox']
                pred_instances = []
                pred_scores = result['scores_3d'].numpy()
                pred_data = result['pts_3d'].numpy()
                pred_labels = result['labels_3d'].numpy()
                for idx in range(len(pred_scores)):
                    pred_instances.append({'class':pred_labels[idx], 
                                        'data':pred_data[idx], 
                                        'confidence_level':pred_scores[idx]})
            results_list.append(pred_instances)
    else: 
        for batch_result in results:
            for result in batch_result:
                result = result['pts_bbox']
                pred_instances = []
                pred_scores = result['scores_3d'].numpy()
                pred_data = result['pts_3d'].numpy()
                pred_labels = result['labels_3d'].numpy()
                pred_attrs_labels = torch.stack(result['attrs_3d']['attrs_preds'], 1)
                pred_attrs_scores = torch.cat(result['attrs_3d']['attrs_scores'], 1)
                for idx in range(len(pred_scores)):
                    pred_instances.append({'class':pred_labels[idx], 
                                        'data':pred_data[idx], 
                                        'attrs_labels':pred_attrs_labels[idx],
                                        'attrs_scores':pred_attrs_scores[idx],
                                        'confidence_level':pred_scores[idx]})
            results_list.append(pred_instances)
    print('convert completed')

    return results_list, left_top

def driving_line_infer_and_save(args, data_list):
    config = args.config
    checkpoint = args.checkpoint
    model = init_model(config, checkpoint, device=args.device)
    print('driving line model initiated')
    prog_bar = mmcv.ProgressBar(len(data_list))
    for idx, image_name in enumerate(data_list): 
        upload_img_path = os.path.join(args.root_path, image_name)
        trajectory_path = upload_img_path.replace('image_data', 'trajectory_data').replace('jpg', 'json')
        results_list, left_top = inference_detector_forcurb_bigimg(model, get_sub_data_driving, upload_img_path, trajectory_path, args.trajectory_sample_num, get_traj_pad=True, pad=False, attr=True)
        if not (len(results_list)):
            print('no data')
            continue
        combine_results_and_save_driving_line(model.cfg, upload_img_path, results_list, out_dir=args.out_dir, left_top=left_top, visualize_flag=args.visualize)
        prog_bar.update()
    print('driving line infer and combine completed')

def driving_box_infer_and_save(args, data_list):
    config = args.config
    checkpoint = args.checkpoint
    model = init_model(config, checkpoint, device=args.device)
    print('driving box model initiated')
    prog_bar = mmcv.ProgressBar(len(data_list))
    for idx, image_name in enumerate(data_list):
        upload_img_path = os.path.join(args.root_path, image_name)
        trajectory_path = upload_img_path.replace('image_data', 'trajectory_data').replace('jpg', 'json')
        results_list, left_top = inference_detector_forcurb_bigimg(model, get_sub_data_driving, upload_img_path, trajectory_path, args.trajectory_sample_num, pad=False, attr=True)
        if not (len(results_list)):
            print('no data')
            continue
        combine_results_and_save_driving_box(model.cfg, upload_img_path, results_list, out_dir=args.out_dir, left_top=left_top, visualize_flag=args.visualize)
        prog_bar.update()
    print('driving box infer and combine completed')

def driving_freespace_infer_and_save(args, data_list):
    config = args.config
    checkpoint = args.checkpoint
    model = init_model(config, checkpoint, device=args.device)
    print('driving freespace model initiated')
    prog_bar = mmcv.ProgressBar(len(data_list))
    for idx, image_name in enumerate(data_list):
        upload_img_path = os.path.join(args.root_path, image_name)
        trajectory_path = upload_img_path.replace('image_data', 'trajectory_data').replace('jpg', 'json')
        results_list, left_top = inference_detector_forcurb_bigimg(model, get_sub_data_driving, upload_img_path, trajectory_path, args.trajectory_sample_num, pad=True, attr=False)
        if not (len(results_list)):
            print('no data')
            continue
        combine_results_and_save_driving_freespace(model.cfg, upload_img_path, results_list, out_dir=args.out_dir, left_top=left_top, visualize_flag=args.visualize)
        prog_bar.update()
    print('driving freespace infer and combine completed')


