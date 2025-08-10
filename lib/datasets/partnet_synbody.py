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


# def load_intrinsics(path):
#     with open(path, 'r') as file:
#         f, cx, cy, _ = map(float, file.readline().split())
#         grid_barycenter = list(map(float, file.readline().split()))
#         scale = float(file.readline())
#         height, width = map(int, file.readline().split())
#     fx = fy = f
#     return fx, fy, cx, cy, height, width
def load_intrinsics(near=0.01, far=40):
    # with open(path, 'r') as file:
    #     width, height = map(float, file.readline().split())
    w = h = 128
    # f = np.sqrt(width * width + height * height)
    f = 150
    fx = fy = f
    cx = w / 2
    cy = w / 2
    opengl_proj = np.array([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                            [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                            [0.0, 0.0, 1.0, 0.0]])
    assert False
    return fx, fy, cx, cy, h, w


def load_pose(pose_param):
    # with open(path, 'rb') as f:
    #     pose_param = json.load(f)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = np.array(pose_param['R'], dtype=np.float32)
    w2c[:3, 3] = np.array(pose_param['T'], dtype=np.float32)
    c2w = np.linalg.inv(w2c)
    # c2w = np.array(pose_param['cam_param'], dtype=np.float32).reshape(4,4)
    # cam_center = np.array(pose_param['T'], dtype=np.float32)
    cam_center = c2w[:3, 3]
    # w2c = np.linalg.inv(c2w)
    # pose[:,:2] *= -1
    # pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(9, 4)
    return [torch.from_numpy(w2c), torch.from_numpy(cam_center)]

def load_smpl(path, smpl_type='smpl'):
    filetype = path.split('.')[-1]
    if filetype=='pkl':
        with open(path, 'rb') as f:
            smpl_param_data = pickle.load(f)
    elif filetype == 'json':
        with open(path, 'rb') as f:
            smpl_param_data = json.load(f)
    elif filetype == 'npz':
        # all gender is neutral :) happy thing
        smpl_param_data = dict(np.load(path, allow_pickle=True))[smpl_type].item()
    else:
        assert False

    # print(smpl_param_data['smplx'].shape)
    # print(smpl_param_data['meta'].shape)
    # print(smpl_param_data.files)
    # with open(os.path.join(os.path.split(path)[0][:-5], 'pose', '000_000.json'), 'rb') as f:
    #     tf_param = json.load(f)
    # scale = smpl_param_data['scale'] * tf_param['scale']
    # transl = (smpl_param_data['transl'] * tf_param['scale'] - np.array(tf_param['center'], dtype=np.float32)) / scale
    if smpl_type=='smpl':
        smpl_param = np.concatenate([np.array([1.]).reshape(1, -1), smpl_param_data['transl'].reshape(1, -1), 
                    smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    elif smpl_type == 'smplx':
        # for custom
        # smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center']).reshape(1, -1), 
        #             np.zeros_like(np.array(smpl_param_data['global_orient']).reshape(1, -1)), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
        #             np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)

        # for thuman
        # betas = [0.6822166,-0.8279,-0.12413712,-0.3038,-0.19082598,1.,-0.542097,0.7116524,-1.,0.2815806]
        smpl_param = np.concatenate([np.array([1.]).reshape(1, -1), np.array(smpl_param_data['transl'].detach().cpu()).reshape(1, -1), 
                    np.array(smpl_param_data['global_orient'].detach().cpu()).reshape(1, -1), np.array(smpl_param_data['body_pose'].detach().cpu()).reshape(1, -1), 
                    np.array(smpl_param_data['betas'].detach().cpu()).reshape(1, -1), np.array(smpl_param_data['left_hand_pose'].detach().cpu()).reshape(1, -1), np.array(smpl_param_data['right_hand_pose'].detach().cpu()).reshape(1, -1),
                    np.array(smpl_param_data['jaw_pose'].detach().cpu()).reshape(1, -1), np.array(smpl_param_data['leye_pose'].detach().cpu()).reshape(1, -1), np.array(smpl_param_data['reye_pose'].detach().cpu()).reshape(1, -1), 
                    np.array(smpl_param_data['expression'].detach().cpu()).reshape(1, -1)], axis=1)
    else:
        assert False
    # smpl_param = np.concatenate([smpl_param_data['scale'][:, None], smpl_param_data['transl'], 
    #                 smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    return torch.from_numpy(smpl_param.astype(np.float32)).reshape(-1)


@DATASETS.register_module()
class PartDataset_Syn(Dataset):
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
        super(PartDataset_Syn, self).__init__()
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
        self.set_seed = 0

        self.radius = torch.tensor([radius], dtype=torch.float32).expand(3)
        self.center = torch.zeros_like(self.radius)

        self.load_scenes()

        if self.test_pose_override is not None:
            pose_dir = os.path.join(self.test_pose_override, 'pose')
            # smpl_dir = os.path.join(self.test_pose_override, 'smpl')
            pose_names = os.listdir(pose_dir)
            pose_names.sort()
            poses_list = []
            for pose_name in pose_names:
                pose_path = os.path.join(pose_dir, pose_name)
                c2w = torch.FloatTensor(load_pose(pose_path))
                cam_to_ndc = torch.cat(
                    [c2w[:3, :3], (c2w[:3, 3:] - self.center[:, None]) / self.radius[:, None]], dim=-1)
                poses_list.append(
                    torch.cat([
                        cam_to_ndc,
                        cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                    ], dim=-2))
            self.test_poses = torch.stack(poses_list, dim=0)  # (n, 4, 4)
            # self.smpl_params = torch.stack(smpl_param_list, dim=0) # (n, 72)
            # fx, fy, cx, cy, h, w = load_intrinsics(os.path.join(self.test_pose_override, 'intrinsics.txt'))
            fx, fy, cx, cy, h, w = load_intrinsics()
            print(self.test_pose_override)
            assert False
            intrinsics_single = torch.FloatTensor([fx, fy, cx, cy])
            smpl_path = os.path.join(
                self.test_pose_override, 'smplx/' + os.path.split(self.test_pose_override)[0][-4:] + '_smpl.pkl')
            self.smpl_params = load_smpl(smpl_path)
            self.test_intrinsics = intrinsics_single[None].expand(self.test_poses.size(0), -1)
        else:
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
                        if dataset_name[-8:] == 'tightcap' or dataset_name[-5:] == 'debug':
                            smpl_path = os.path.join(
                                sample_dir, 'smplx/' + 'smplx_params.pkl')
                            smpl_params = load_smpl(smpl_path, smpl_type='smplx')
                        else:
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + os.path.split(image_dir)[0][-4:] + '_smpl.pkl')
                            smpl_path = os.path.join(
                                sample_dir, 'smplx/' + 'smplx.npz')
                            smpl_params = load_smpl(smpl_path, smpl_type='smplx')

                        poses_path = os.path.join(sample_dir, 'pose', 'cameras.json')
                        with open(poses_path, 'rb') as f:
                            poses_data = json.load(f)
                        image_names = os.listdir(os.path.join(image_dir, 'layer0'))
                        image_names.sort()
                        image_paths = []
                        poses = []
                        # smpl_params = []
                        for image_name in image_names:
                            image_paths.append(os.path.join(image_dir, 'layer0', image_name))
                            # pose_path = os.path.join(
                            #     sample_dir, 'pose/' + os.path.splitext(image_name)[0] + '.json')
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + 'mesh-f' + os.path.split(image_dir)[0][-5:] + '.json')
                            # smpl_path = os.path.join(
                            #     sample_dir, 'smplx/' + os.path.split(image_dir)[0][-4:] + '_smpl.pkl')
                            poses.append(load_pose(poses_data['camera'+image_name[:4]]))
                            # smpl_params.append(load_smpl(smpl_path))
                        scenes.append(dict(
                            # intrinsics=intrinsics,
                            image_paths=image_paths,
                            poses=poses,
                            smpl_params=smpl_params))
            scenes = sorted(scenes, key=lambda x: x['image_paths'][0].split('/')[-4])
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
        scene_name = image_paths[0].split('/')[-4]
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
            # print(type(smpl_params))
            # print(smpl_params.shape)
            # torch.save(smpl_params, '/home/zhangweitian/HighResAvatar/debug/smplx_param.pth')
            # assert False

            def gather_imgs(img_ids):
                imgs_list = [] if self.load_imgs else None
                segs_list = [] if self.load_imgs else None
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
                        # if self.img_res == 1024:
                        img0 = mmcv.imread(image_paths[img_id], channel_order='rgb')
                        img1 = mmcv.imread(image_paths[img_id].replace('layer0', 'layer1'), channel_order='rgb')
                        img2 = mmcv.imread(image_paths[img_id].replace('layer0', 'layer2'), channel_order='rgb')
                        img3 = mmcv.imread(image_paths[img_id].replace('layer0', 'layer3'), channel_order='rgb')
                        seg0 = mmcv.imread(image_paths[img_id].replace('rgb', 'mask')[:-3]+'png', flag='grayscale')
                        seg1 = mmcv.imread(image_paths[img_id].replace('rgb', 'mask').replace('layer0', 'layer1')[:-3]+'png', flag='grayscale')
                        seg2 = mmcv.imread(image_paths[img_id].replace('rgb', 'mask').replace('layer0', 'layer2')[:-3]+'png', flag='grayscale')
                        seg3 = mmcv.imread(image_paths[img_id].replace('rgb', 'mask').replace('layer0', 'layer3')[:-3]+'png', flag='grayscale')
                        # img0[seg0 == 0] = 255
                        # img1[seg1 == 0] = 255
                        # img2[seg2 == 0] = 255
                        # img3[seg3 == 0] = 255
                            # seg_debug = np.load(image_paths[img_id].replace('rgb', 'mask')[:-3]+'npy')
                            # seg = mmcv.imread(image_paths[img_id].replace('rgb', 'new_mask_refine'), flag='grayscale')
                        # else:
                        #     assert False
                        #     img = Image.open(image_paths[img_id]).convert('RGB') # RGB default order in PIL
                        #     # img = F.to_tensor(img).permute(1, 2, 0)
                        #     # seg = Image.open(image_paths[img_id].replace('rgb', 'new_mask_refine')).convert('L')
                        #     img = np.asarray(img.resize((self.img_res,self.img_res), resample=Image.Resampling.LANCZOS))[:,:,:3]
                        #     # seg = np.asarray(seg.resize((self.img_res,self.img_res), resample=Image.Resampling.LANCZOS))
                        # img0 = img0.astype(np.float32) / 255  # (h, w, 3)
                        # img1 = img1.astype(np.float32) / 255
                        # img2 = img2.astype(np.float32) / 255
                        # img3 = img3.astype(np.float32) / 255
                        img = np.concatenate([img3, img0, img1, img2], axis=-1).astype(np.float32) / 255
                        seg = np.stack([seg3, seg0, seg1, seg2], axis=-1) / 255
                        # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                        # seg = cv2.resize(seg, (512, 512), interpolation=cv2.INTER_NEAREST)
                        # print(seg_debug.shape)
                        # print(seg.shape)
                        # print((seg_debug-seg).sum())
                        # assert False
                        # seg_part = [torch.from_numpy(seg==(i+1)) for i in range(5)]
                        # seg_full = torch.from_numpy(seg)
                        # seg_part.append(seg_full)
                        # seg = torch.stack(seg_part, dim=-1)
                        seg = torch.from_numpy(seg)
                        img = torch.from_numpy(img)
                        segs_list.append(seg)
                        imgs_list.append(img)
                    # if self.load_norm:
                    #     norm = mmcv.imread(image_paths[img_id].replace('rgb', 'norm'), channel_order='rgb')
                    #     norm = torch.from_numpy(norm.astype(np.float32) / 255)
                    #     norm_list.append(norm)
                poses_list = torch.stack(poses_list, dim=0)  # (n, 4, 4)
                cam_centers_list = torch.stack(cam_centers_list, dim=0)
                # smpl_list = torch.stack(smpl_list, dim=0)
                # intrinsics = intrinsics_single[None].expand(len(img_ids), -1)
                # smpl_params = smpl_params_single[None].expand(len(img_ids), -1)
                if self.load_imgs:
                    imgs_list = torch.stack(imgs_list, dim=0)  # (n, h, w, 3)
                    # segs_list = torch.stack(segs_list, dim=0).unsqueeze(-1)
                    segs_list = torch.stack(segs_list, dim=0)
                if self.load_norm:
                    norm_list = torch.stack(norm_list, dim=0)
                # return imgs_list, poses_list, intrinsics, img_paths_list, smpl_params
                return imgs_list, poses_list, cam_centers_list, img_paths_list, smpl_params, norm_list, segs_list

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
                    generator = torch.Generator()
                    generator.manual_seed(self.set_seed)
                    # cond_inds = random.sample(range(num_imgs), self.specific_observation_num)
                    cond_inds = torch.randperm(num_imgs, generator=generator)[:self.specific_observation_num]
                    # cond_inds = torch.randperm(num_imgs)[:self.specific_observation_num]
                    # print(cond_inds)
                    # assert False
                    # cond_inds = torch.randperm(num_imgs+9)[:self.specific_observation_num] % num_imgs
                    # cond_inds = (torch.randperm(num_imgs//3)[:self.specific_observation_num]) * 3 + 1
                    # cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)[1::3]
                else:
                    cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)
            else:
                cond_inds = self.specific_observation_idcs
            # print(cond_inds)
            # test_inds = list(range(num_imgs))[::20]
            test_inds = list(range(num_imgs))
            # cond_inds = list(range(num_imgs))
            if self.specific_observation_num:
                test_inds = []
                # test_inds = test_inds[::9]
            else:
                pass
                # for cond_ind in cond_inds:
                #     test_inds.remove(cond_ind)
            if self.load_cond_data and len(cond_inds) > 0:
                cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm, cond_segs = gather_imgs(cond_inds)
                results.update(
                    cond_poses=cond_poses,
                    cond_intrinsics=cond_intrinsics,
                    cond_img_paths=DC(cond_img_paths, cpu_only=True),
                    cond_smpl_param=cond_smpl_param)
                if cond_imgs is not None:
                    results.update(cond_imgs=cond_imgs)
                if cond_norm is not None:
                    results.update(cond_norm=cond_norm)
                if cond_segs is not None:
                    results.update(cond_segs=cond_segs)

            if self.load_test_data and len(test_inds) > 0:
                test_imgs, test_poses, test_intrinsics, test_img_paths, test_smpl_param, test_norm, test_segs = gather_imgs(test_inds)
                results.update(
                    test_poses=test_poses,
                    test_intrinsics=test_intrinsics,
                    test_img_paths=DC(test_img_paths, cpu_only=True),
                    test_smpl_param=test_smpl_param)
                if test_imgs is not None:
                    results.update(test_imgs=test_imgs)
                if test_norm is not None:
                    results.update(test_norm=test_norm)
                if test_segs is not None:
                    results.update(test_segs=test_segs)

        if self.code_dir is not None:
            code_file = os.path.join(self.code_dir, scene_name + '.pth')
            if os.path.exists(code_file):
                results.update(
                    code=DC(torch.load(code_file, map_location='cpu'), cpu_only=True))

        if self.test_pose_override is not None:
            results.update(test_poses=self.test_poses, test_intrinsics=self.test_intrinsics)

        return results

    def set_epoch(self, epoch):
        # assert False
        self.set_seed = self.set_seed + epoch

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, scene_id):
        return self.parse_scene(scene_id)
