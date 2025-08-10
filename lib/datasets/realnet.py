import os
import random
import numpy as np
import torch
import mmcv
import json
import pickle
from PIL import Image
from torch.utils.data import Dataset
from itertools import chain
import torchvision.transforms.functional as F

from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS


def load_pose(path, image_name):
    # print(image_name)
    # cam_dict = {'0000':'22010716', '0001':'22010708', '0002':'22053908'}
    cam_dict = {'0002':'22053903', '0003':'22053907', '0004':'22070928', '0005':'22139907'}
    # cam_dict = {'0002':'22010716', '0003':'22010708', '0004':'22010710', '0005':'22053925'}
    # cam_dict = {'0000':'22010710', '0001':'22053903', '0002':'22053925'}
    with open(path, 'rb') as f:
        # pose_param = json.load(f)['22010716']
        pose_param = json.load(f)[cam_dict[image_name[:-4]]]
    # c2w = np.array(pose_param['cam_param'], dtype=np.float32).reshape(4,4)
    # cam_center = c2w[:3, 3]
    # w2c = np.linalg.inv(c2w)
    W = 1500
    H = 2048
    K = np.array(pose_param['K'], dtype=np.float32).reshape(3,3)
    R = np.array(pose_param['R'], dtype=np.float32).reshape(3,3)
    T = np.array(pose_param['T'], dtype=np.float32).reshape(3,1)
    M = np.eye(3)
    M[0, 2] = (K[0, 2] - W / 2) / K[0, 0]
    M[1, 2] = (K[1, 2] - H / 2) / K[1, 1]
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    R = M @ R
    T = M @ T

    # R = np.transpose(R)
    T = T[:, 0]

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    c2w = np.linalg.inv(Rt)
    cam_center = c2w[:3, 3]

    cam_center_ = np.zeros((5,))
    cam_center_[:3] = cam_center
    cam_center_[3] = K[0, 0]
    cam_center_[4] = K[1, 1]

    # K[0, :] *= self.w / self.W
    # K[1, :] *= self.h / self.H
    
    return [torch.from_numpy(Rt.astype(np.float32)), torch.from_numpy(cam_center_.astype(np.float32))]

def load_smpl(path, smpl_type='smpl'):
    smpl_param_data = np.load(path)

    if smpl_type=='smpl':
        smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center'])[None], 
                    smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    elif smpl_type == 'smplx':
        # for custom
        # smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center']).reshape(1, -1), 
        #             np.zeros_like(np.array(smpl_param_data['global_orient']).reshape(1, -1)), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
        #             np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)

        # for thuman
        if path.split('/')[-3] == 'human_real_0':
            idx = 1
        elif path.split('/')[-3] == 'human_real_1':
            idx = 77
        else:
            assert False
        smpl_param = np.concatenate([np.array([1.,]).reshape(1, -1), np.array(smpl_param_data['transl'][idx]).reshape(1, -1), 
                    np.array(smpl_param_data['global_orient'][idx]).reshape(1, -1), np.array(smpl_param_data['body_pose'][idx]).reshape(1, -1), 
                    np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose'][idx]).reshape(1, -1), np.array(smpl_param_data['right_hand_pose'][idx]).reshape(1, -1),
                    np.array(smpl_param_data['jaw_pose'][idx]).reshape(1, -1), np.array([0., 0., 0.]).reshape(1, -1), np.array([0., 0., 0.]).reshape(1, -1), 
                    np.array(smpl_param_data['expression'][idx]).reshape(1, -1)], axis=1)
    else:
        assert False
    # smpl_param = np.concatenate([smpl_param_data['scale'][:, None], smpl_param_data['transl'], 
    #                 smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    return torch.from_numpy(smpl_param.astype(np.float32)).reshape(-1)


@DATASETS.register_module()
class RealDataset(Dataset):
    def __init__(self,
                 data_prefix,
                 code_dir=None,
                 code_only=False,
                 load_imgs=True,
                 load_norm=False,
                 specific_observation_idcs=None,
                 specific_observation_num=None,
                 num_test_imgs=0,
                 random_test_imgs=False,
                 scene_id_as_name=False,
                 cache_path=None,
                 test_pose_override=None,
                 num_train_imgs=-1,
                 load_cond_data=True,
                 load_test_data=True,
                 max_num_scenes=-1,  # for debug or testing
                #  radius=0.5,
                 radius=1.0,
                 img_res=1024,
                 test_mode=False,
                 step=1,  # only for debug & visualization purpose
                 ):
        super(RealDataset, self).__init__()
        self.data_prefix = data_prefix
        self.code_dir = code_dir
        self.code_only = code_only
        self.load_imgs = load_imgs
        self.load_norm = load_norm
        self.specific_observation_idcs = specific_observation_idcs
        self.specific_observation_num = specific_observation_num
        self.num_test_imgs = num_test_imgs
        self.random_test_imgs = random_test_imgs
        self.scene_id_as_name = scene_id_as_name
        self.cache_path = cache_path
        self.test_pose_override = test_pose_override
        self.num_train_imgs = num_train_imgs
        self.load_cond_data = load_cond_data
        self.load_test_data = load_test_data
        self.max_num_scenes = max_num_scenes
        self.step = step
        self.img_res = img_res

        self.radius = torch.tensor([radius], dtype=torch.float32).expand(3)
        self.center = torch.zeros_like(self.radius)

        self.load_scenes()
        self.test_poses = self.test_intrinsics = None

    def load_scenes(self):
        # self.cache_path 'data/humanscan/human_train_cache.pkl'
        if self.cache_path is not None and os.path.exists(self.cache_path):
            scenes = mmcv.load(self.cache_path)
        else:
            data_prefix_list = self.data_prefix if isinstance(self.data_prefix, list) else [self.data_prefix]
            scenes = []
            for data_prefix in data_prefix_list:
                dataset_name = data_prefix.split('/')[1]
                sample_dir_list = os.listdir(data_prefix)
                # sample_dir_list.sort()
                for name in sample_dir_list:
                    sample_dir = os.path.join(data_prefix, name)
                    if os.path.isdir(sample_dir):
                        image_dir = os.path.join(sample_dir, 'rgb')
                        # image_dir = os.path.join(sample_dir, 'norm')
                        if dataset_name[:6] == 'custom':
                            smpl_path = os.path.join(
                                sample_dir, 'smplx/' + 'mesh-f' + os.path.split(image_dir)[0][-5:] + '.json')
                            smpl_params = load_smpl(smpl_path, smpl_type='smplx')
                        else:
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + os.path.split(image_dir)[0][-4:] + '_smpl.pkl')
                            smpl_path = os.path.join(
                                sample_dir, 'smpl_params.npz')
                            smpl_params = load_smpl(smpl_path, smpl_type='smplx')
                        image_names = os.listdir(image_dir)
                        image_names.sort()
                        image_paths = []
                        poses = []
                        # smpl_params = []
                        for image_name in image_names:
                            image_paths.append(os.path.join(image_dir, image_name))
                            pose_path = os.path.join(
                                sample_dir,  'calibration_full.json')
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + 'mesh-f' + os.path.split(image_dir)[0][-5:] + '.json')
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + os.path.split(image_dir)[0][-4:] + '_smpl.pkl')
                            poses.append(load_pose(pose_path, image_name))
                            # smpl_params.append(load_smpl(smpl_path))
                        scenes.append(dict(
                            # intrinsics=intrinsics,
                            image_paths=image_paths,
                            poses=poses,
                            smpl_params=smpl_params))
            scenes = sorted(scenes, key=lambda x: x['image_paths'][0].split('/')[-3])
            if self.cache_path is not None:
                mmcv.dump(scenes, self.cache_path)
        end = len(scenes)
        if self.max_num_scenes >= 0:
            end = min(end, self.max_num_scenes * self.step)
        self.scenes = scenes[:end:self.step]
        self.num_scenes = len(self.scenes)

    def parse_scene(self, scene_id):
        scene = self.scenes[scene_id]
        image_paths = scene['image_paths']
        scene_name = image_paths[0].split('/')[-3]
        results = dict(
            scene_id=DC(scene_id, cpu_only=True),
            scene_name=DC(
                '{:04d}'.format(scene_id) if self.scene_id_as_name else scene_name,
                cpu_only=True))
        
        if not self.code_only:
            # fx, fy, cx, cy, h, w = scene['intrinsics']
            # intrinsics_single = torch.FloatTensor([fx, fy, cx, cy])
            poses = scene['poses']
            smpl_params = scene['smpl_params']

            def gather_imgs(img_ids):
                imgs_list = [] if self.load_imgs else None
                norm_list = [] if self.load_norm else None
                poses_list = []
                cam_centers_list = []
                # smpl_list = []
                img_paths_list = []

                for img_id in img_ids:
                    pose = poses[img_id][0]
                    cam_centers_list.append(torch.FloatTensor(poses[img_id][1]))
                    # smpl_param = smpl_params[img_id]
                    # smpl_list.append(smpl_param)
                    c2w = torch.FloatTensor(pose)
                    cam_to_ndc = torch.cat(
                        [c2w[:3, :3], (c2w[:3, 3:] - self.center[:, None]) / self.radius[:, None]], dim=-1)
                    poses_list.append(
                        torch.cat([
                            cam_to_ndc,
                            cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                        ], dim=-2))
                    img_paths_list.append(image_paths[img_id])
                    if self.load_imgs:
                        # img = mmcv.imread(image_paths[img_id], channel_order='rgb')
                        img = Image.open(image_paths[img_id]).convert('RGB') # RGB default order in PIL
                        # img = F.to_tensor(img).permute(1, 2, 0)
                        img = np.asarray(img.resize((750, 1024), resample=Image.Resampling.LANCZOS))[:,:,:3]
                        img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
                        # img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
                        mask = Image.open(image_paths[img_id].replace('rgb', 'mask')).convert('RGB')
                        mask = np.asarray(mask.resize((750,1024), resample=Image.Resampling.LANCZOS))[:,:,:3]
                        mask = torch.from_numpy(mask.astype(np.float32) / 255)
                        # norm = Image.open(image_paths[img_id].replace('rgb', 'norm'))
                        # norm = np.asarray(norm.resize((self.img_res,self.img_res), resample=Image.Resampling.LANCZOS))[:,:,:3]
                        # norm = torch.from_numpy(norm.astype(np.float32) / 255)
                        imgs_list.append(torch.cat([img, mask], dim=-1))
                        # imgs_list.append(img)
                    if self.load_norm:
                        norm = mmcv.imread(image_paths[img_id].replace('rgb', 'norm'), channel_order='rgb')
                        norm = torch.from_numpy(norm.astype(np.float32) / 255)
                        norm_list.append(norm)
                poses_list = torch.stack(poses_list, dim=0)  # (n, 4, 4)
                cam_centers_list = torch.stack(cam_centers_list, dim=0)
                # smpl_list = torch.stack(smpl_list, dim=0)
                # intrinsics = intrinsics_single[None].expand(len(img_ids), -1)
                # smpl_params = smpl_params_single[None].expand(len(img_ids), -1)
                if self.load_imgs:
                    imgs_list = torch.stack(imgs_list, dim=0)  # (n, h, w, 3)
                if self.load_norm:
                    norm_list = torch.stack(norm_list, dim=0)
                # return imgs_list, poses_list, intrinsics, img_paths_list, smpl_params
                return imgs_list, poses_list, cam_centers_list, img_paths_list, smpl_params, norm_list

            num_imgs = len(image_paths)
            if self.specific_observation_idcs is None:
                if self.num_train_imgs >= 0:
                    num_train_imgs = self.num_train_imgs
                else:
                    num_train_imgs = num_imgs - self.num_test_imgs
                if self.random_test_imgs:
                    # cond_inds = random.sample(chain(range(num_imgs),range(9)), num_train_imgs)
                    cond_inds = random.sample(range(num_imgs), num_train_imgs)
                elif self.specific_observation_num:
                    cond_inds = torch.randperm(num_imgs)[:self.specific_observation_num]
                    # cond_inds = torch.randperm(num_imgs+9)[:self.specific_observation_num] % num_imgs
                    # cond_inds = (torch.randperm(num_imgs//3)[:self.specific_observation_num]) * 3 + 1
                    # cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)[1::3]
                else:
                    cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)
            else:
                cond_inds = self.specific_observation_idcs

            test_inds = list(range(num_imgs))
            # cond_inds = list(range(num_imgs))
            if self.specific_observation_num:
                test_inds = []
            else:
                for cond_ind in cond_inds:
                    test_inds.remove(cond_ind)
            
            if self.load_cond_data and len(cond_inds) > 0:
                cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm = gather_imgs(cond_inds)
                results.update(
                    cond_poses=cond_poses,
                    cond_intrinsics=cond_intrinsics,
                    cond_img_paths=DC(cond_img_paths, cpu_only=True),
                    cond_smpl_param=cond_smpl_param)
                if cond_imgs is not None:
                    results.update(cond_imgs=cond_imgs)
                if cond_norm is not None:
                    results.update(cond_norm=cond_norm)

            if self.load_test_data and len(test_inds) > 0:
                test_imgs, test_poses, test_intrinsics, test_img_paths, test_smpl_param, test_norm = gather_imgs(test_inds)
                results.update(
                    test_poses=test_poses,
                    test_intrinsics=test_intrinsics,
                    test_img_paths=DC(test_img_paths, cpu_only=True),
                    test_smpl_param=test_smpl_param)
                if test_imgs is not None:
                    results.update(test_imgs=test_imgs)
                if test_norm is not None:
                    results.update(test_norm=test_norm)

        if self.code_dir is not None:
            code_file = os.path.join(self.code_dir, scene_name + '.pth')
            if os.path.exists(code_file):
                results.update(
                    code=DC(torch.load(code_file, map_location='cpu'), cpu_only=True))

        if self.test_pose_override is not None:
            assert False
            results.update(test_poses=self.test_poses, test_intrinsics=self.test_intrinsics)

        return results

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, scene_id):
        return self.parse_scene(scene_id)
