from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import imageio
import json
import pickle
import time
import cv2
import mmcv
from tqdm import tqdm
from mmcv.runner import get_dist_info
from mmgen.core.evaluation.metrics import FID, IS
from mmgen.models.architectures.common import get_module_device


def evaluate_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    # viz_step = 1
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if metrics is not None and len(metrics) > 0:
            pred_imgs = outputs_dict['pred_imgs'][:,:,:3]
            pred_imgs = pred_imgs.reshape(
                -1, 3, *outputs_dict['pred_imgs'].shape[3:]).split(feed_batch_size, dim=0)
            real_imgs = None
            for metric in metrics:
                if 'test_imgs' in data and not isinstance(metric, (FID, IS)) and real_imgs is None:
                    real_imgs = data['test_imgs'].permute(0, 1, 4, 2, 3)[:,:,3]
                    print(real_imgs[0].shape)
                    assert False
                    real_imgs = real_imgs.reshape(-1, 3, *real_imgs.shape[3:]).split(feed_batch_size, dim=0)
                for batch_id, batch_imgs in enumerate(pred_imgs):
                    # feed in fake images
                    metric.feed(batch_imgs * 2 - 1, 'fakes')
                    if not isinstance(metric, (FID, IS)) and real_imgs is not None:
                        metric.feed(real_imgs[batch_id] * 2 - 1, 'reals')

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars

def recon_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    # max_num_scenes = 64
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    viz_step=1
    # sampling fake images and directly send them to metrics
    for i, data in enumerate(dataloader):
        # if i >= 0 and i < 1:
        # if i >= 248 and i < 264:
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])
        # assert False

        if rank == 0:
            pbar.update(total_batch_size)
        if i == max_num_scenes//total_batch_size - 1:
            break

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars

def exmesh_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []

    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        model.exmesh_step(data, **sample_kwargs_)
        assert False
        batch_size_list.append(outputs_dict['num_samples'])


        if rank == 0:
            pbar.update(total_batch_size)

    return

def load_pose(path):
    with open(path, 'rb') as f:
        pose_param = json.load(f)
    w2c = np.array(pose_param['cam_param'], dtype=np.float32).reshape(36,4,4)
    cam_center = w2c[:, :3, 3]
    c2w = np.linalg.inv(w2c)
    # pose[:,:2] *= -1
    # pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(9, 4)
    c2w = torch.from_numpy(c2w)
    cam_to_ndc = torch.cat([c2w[:, :3, :3], c2w[:, :3, 3:]], dim=-1)
    pose = torch.cat([cam_to_ndc, cam_to_ndc.new_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(36, -1, -1)], dim=-2)

    return [pose, torch.from_numpy(cam_center)]


def viz_3d(model, batch_size, max_num_scenes, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    # viz_step = 1
    path = 'demo/cam_36.json'
    cam_poses = load_pose(path)

    data_indices = list(range(max_num_scenes))
    with tqdm(total=max_num_scenes) as main_pbar:
        for i in range(0, max_num_scenes, total_batch_size):
            current_batch = data_indices[i : i + total_batch_size]
            sample_kwargs_ = deepcopy(sample_kwargs)
            if viz_dir is not None and i % viz_step == 0:
                sample_kwargs_.update(viz_dir=viz_dir)
            data = {}
            for j in range(ws):
                if rank == j:
                    data['scene_id'] = current_batch[batch_size*j:batch_size*(j+1)]
                    data['scene_name'] = [f'scene_{str(idx).zfill(4)}' for idx in data['scene_id']]
            data['test_poses'] = cam_poses[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            data['test_intrinsics'] = cam_poses[1].unsqueeze(0).repeat(batch_size, 1, 1)
            outputs_dict = model.val_step(
                data, show_pbar=rank == 0,
                **sample_kwargs_)
            for k, v in outputs_dict['log_vars'].items():
                if k in log_vars:
                    log_vars[k].append(outputs_dict['log_vars'][k])
                else:
                    log_vars[k] = [outputs_dict['log_vars'][k]]
            batch_size_list.append(outputs_dict['num_samples'])

            if rank == 0:
                main_pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars


def animate_3d(model, batch_size, max_num_scenes, smplx_path, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, mini_batch=30, sample_kwargs=dict()):
    batch_size = batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')

    log_vars = dict()
    batch_size_list = []
    
    smplx_pose_param = np.load(smplx_path[0], allow_pickle=True)

    # amass
    if 'stageii' in smplx_path[0]:
        smplx_param = np.concatenate([smplx_pose_param['poses'][:, :3], smplx_pose_param['pose_body'], smplx_pose_param['pose_hand'], smplx_pose_param['pose_jaw'], smplx_pose_param['pose_eye']], axis=1)
        rotmat = torch.as_tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    else:
        smplx_param = np.concatenate([smplx_pose_param['body_pose'][:, :-6], smplx_pose_param['left_hand_pose'], smplx_pose_param['right_hand_pose'], smplx_pose_param['jaw_pose'], smplx_pose_param['expression']], axis=1)
        rotmat = torch.as_tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    smplx_param = torch.as_tensor(smplx_param).unsqueeze(0).expand(batch_size, -1, -1).float()

    cam_poses = load_pose('demo/cam_36.json')
    data_indices = list(range(max_num_scenes))
    with tqdm(total=max_num_scenes) as main_pbar:
        for i in range(0, max_num_scenes, total_batch_size):
            current_batch = data_indices[i : i + total_batch_size]
            sample_kwargs_ = deepcopy(sample_kwargs)
            if viz_dir is not None and i % viz_step == 0:
                sample_kwargs_.update(viz_dir=viz_dir)
            data = {}
            for j in range(ws):
                if rank == j:
                    data['scene_id'] = current_batch[batch_size*j:batch_size*(j+1)]
                    data['scene_name'] = [f'scene_{str(idx).zfill(4)}' for idx in data['scene_id']]
            data['test_poses'] = cam_poses[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            data['test_intrinsics'] = cam_poses[1].unsqueeze(0).repeat(batch_size, 1, 1)

            if 'stageii' in smplx_path[0]:
                data['test_poses'] = torch.bmm(data['test_poses'][:, 10], rotmat.unsqueeze(0).expand(batch_size, -1, -1).float()).unsqueeze(1)
                data['test_intrinsics'] = data['test_intrinsics'][:, 32:33]
            else:
                data['test_poses'] = torch.bmm(data['test_poses'][:, 1], rotmat.unsqueeze(0).expand(batch_size, -1, -1).float()).unsqueeze(1)
                data['test_intrinsics'] = data['test_intrinsics'][:, 32:33]

            betas = torch.zeros(batch_size, smplx_param.shape[1], 10).float()

            if 'stageii' in smplx_path[0]:
                test_smpl_param = torch.cat((smplx_param, betas), 2)[:, ::8]
            else:
                test_smpl_param = torch.cat((smplx_param, betas), 2)
            if test_smpl_param.shape[1] > mini_batch:
                batch_test_smpl_param = test_smpl_param.split(mini_batch, dim=1)
                data['mini_batch'] = mini_batch
            else:
                batch_test_smpl_param = [test_smpl_param]
            
            for batch_idx, test_smpl_param in enumerate(batch_test_smpl_param):
                data['test_smpl_param'] = test_smpl_param
                data['batch_idx'] = batch_idx
                outputs_dict = model.val_step(
                    data, show_pbar=rank == 0,
                    **sample_kwargs_)
                if batch_idx == 0:
                    test_code = outputs_dict['code']
                    data['test_code'] = test_code

            for k, v in outputs_dict['log_vars'].items():
                if k in log_vars:
                    log_vars[k].append(outputs_dict['log_vars'][k])
                else:
                    log_vars[k] = [outputs_dict['log_vars'][k]]
            batch_size_list.append(outputs_dict['num_samples'])

            if rank == 0:
                main_pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars
