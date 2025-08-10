'''
Deformer of AG3D adapted from fast-SNARF
'''

from .fast_snarf.lib.model.deformer_smplx import ForwardDeformer, skinning
from .smplx import SMPLX
import torch
from pytorch3d import ops
import numpy as np
import pickle
import json
from pytorch3d.transforms import matrix_to_axis_angle

class SMPLXDeformer(torch.nn.Module):
    
    def __init__(self, gender) -> None:
        super().__init__()
        # self.body_model = SMPLX('/home/zhangweitian/HighResAvatar/lib/models/deformers/smplx/SMPLX', gender=gender, \
        #                         create_body_pose=False, \
        #                         create_betas=False, \
        #                         create_global_orient=False, \
        #                         create_transl=False,
        #                         create_expression=False,
        #                         create_jaw_pose=False,
        #                         create_leye_pose=False,
        #                         create_reye_pose=False,
        #                         create_right_hand_pose=False,
        #                         create_left_hand_pose=False,
        #                         use_pca=True,
        #                         num_pca_comps=12,
        #                         num_betas=10,
        #                         flat_hand_mean=True,)
        self.body_model = SMPLX('lib/models/deformers/smplx/SMPLX', gender='neutral', \
                                create_body_pose=False, \
                                create_betas=False, \
                                create_global_orient=False, \
                                create_transl=False,
                                create_expression=False,
                                create_jaw_pose=False,
                                create_leye_pose=False,
                                create_reye_pose=False,
                                create_right_hand_pose=False,
                                create_left_hand_pose=False,
                                use_pca=False,
                                flat_hand_mean=True,
                                # flat_hand_mean=False,
                                # num_pca_comps=12,
                                # num_expression_coeffs=50,
                                num_betas=10)

        # self.body_model_12 = SMPLX('lib/models/deformers/smplx/SMPLX', gender='neutral', \
        #                         create_body_pose=False, \
        #                         create_betas=False, \
        #                         create_global_orient=False, \
        #                         create_transl=False,
        #                         create_expression=False,
                                # create_jaw_pose=False,
                                # create_leye_pose=False,
                                # create_reye_pose=False,
                                # create_right_hand_pose=False,
                                # create_left_hand_pose=False,
                                # # use_pca=False,
                                # # flat_hand_mean=True,
                                # flat_hand_mean=False,
                                # num_pca_comps=12,
                                # # num_expression_coeffs=50,
                                # num_betas=10)

        # self.body_model_nu = SMPLX('/home/zhangweitian/HighResAvatar/lib/models/deformers/smplx/SMPLX', gender='neutral', \
        #                         create_body_pose=False, \
        #                         create_betas=False, \
        #                         create_global_orient=False, \
        #                         create_transl=False,
        #                         create_expression=False,
        #                         create_jaw_pose=False,
        #                         create_leye_pose=False,
        #                         create_reye_pose=False,
        #                         create_right_hand_pose=False,
        #                         create_left_hand_pose=False,
        #                         use_pca=False,
        #                         # num_expression_coeffs=50,
        #                         num_betas=10,
        #                         flat_hand_mean=True,)
        # for animate
        # self.body_model = SMPLX('/home/zhangweitian/HighResAvatar/lib/models/deformers/smplx/SMPLX', gender=gender, \
        #                         create_body_pose=False, \
        #                         create_betas=False, \
        #                         create_global_orient=False, \
        #                         create_transl=False,
        #                         create_expression=False,
        #                         create_jaw_pose=False,
        #                         create_leye_pose=False,
        #                         create_reye_pose=False,
        #                         create_right_hand_pose=False,
        #                         create_left_hand_pose=False,
        #                         # use_pca=False,
        #                         use_pca=True,
        #                         num_pca_comps=12,
        #                         num_betas=10,
        #                         flat_hand_mean=False,)
        self.deformer = ForwardDeformer()
        
        # threshold for rendering (need to be larger for loose clothing)
        # self.threshold = 0.12
        self.threshold = 0.12
        # self.threshold = 0.2

        init_spdir = torch.as_tensor(np.load('work_dirs/cache/init_spdir_smplx_full.npy'))
        # init_spdir = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_spdir_smplx_thu_50.npy'))
        self.register_buffer('init_spdir', init_spdir, persistent=False)

        init_podir = torch.as_tensor(np.load('work_dirs/cache/init_podir_smplx_full.npy'))
        self.register_buffer('init_podir', init_podir, persistent=False)
        
        init_faces = torch.as_tensor(np.load('work_dirs/cache/init_faces_smplx_full.npy'))
        self.register_buffer('init_faces', init_faces.unsqueeze(0), persistent=False)

        init_lbs_weights = torch.as_tensor(np.load('work_dirs/cache/init_lbsw_smplx_body.npy'))
        self.register_buffer('init_lbsw', init_lbs_weights.unsqueeze(0), persistent=False)

        self.initialize()
        self.initialized = True

        # smplx_params = torch.load('/home/zhangweitian/HighResAvatar/debug/smplx_param.pth').unsqueeze(0)
        # scale, transl, global_orient, pose, betas, hf_pose, expression = torch.split(smplx_params, [1, 3, 3, 63, 10, 33, 10], dim=1)
        # self.register_buffer('op_scale',  torch.nn.Parameter(scale.float()))
        # self.register_buffer('op_global_orient', torch.nn.Parameter(global_orient.float()))
        # self.register_buffer('op_pose', torch.nn.Parameter(pose.float()))
        # self.register_buffer('op_transl', torch.nn.Parameter(transl.float()))
        # self.register_buffer('op_hf_pose', torch.nn.Parameter(hf_pose.float()))
        # self.register_buffer('op_betas', torch.nn.Parameter(betas.float()))
        # self.register_buffer('op_expression', torch.nn.Parameter(expression.float()))
        # scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
        # init_idx = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_idx_mou_smplx.npy'))
        # self.register_buffer('init_idx', init_idx.unsqueeze(0))
        # init_bar = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_bar_mou_smplx.npy'))
        # self.register_buffer('init_bar', init_bar.unsqueeze(0))

    # def initialize(self, betas):
    def initialize(self):
        
        # device = betas.device
        batch_size = 1
        
        # canonical space is defined in t-pose / star-pose
        body_pose_t = torch.zeros((batch_size, 63))
        # body_pose_t[0, 2] = np.pi / 18
        # body_pose_t[0, 5] = -np.pi / 18

        jaw_pose_t = torch.zeros((batch_size, 3))
        # jaw_pose_t[:, 0] = 0.2

        ##flat_hand_mean = False
        # left_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
        #  -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0)
        # right_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
        #  -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0)
        # ## flat_hand_mean = True
        # left_hand_pose_t = torch.zeros((batch_size, 12))
        # right_hand_pose_t = torch.zeros((batch_size, 12))
        left_hand_pose_t = torch.zeros((batch_size, 45))
        right_hand_pose_t = torch.zeros((batch_size, 45))
        leye_pose_t = torch.zeros((batch_size, 3))
        reye_pose_t = torch.zeros((batch_size, 3))
        expression_t = torch.zeros((batch_size, 10))
        # expression_t = torch.zeros((batch_size, 50))
        # body_pose_t[:, 2] = torch.pi / 6
        # body_pose_t[:, 5] = -torch.pi / 6
        # transl = torch.zeros((batch_size, 3))
        # transl[:,1] = 0.3
        # transl = torch.as_tensor([-0.0012,  0.4668, -0.0127]).unsqueeze(0)
        transl = torch.as_tensor([0., 0.35, 0.]).unsqueeze(0)
        # transl = torch.as_tensor([0., 0., 0.]).unsqueeze(0)
        global_orient = torch.zeros((batch_size, 3))
        # self.body_model = self.body_model.to(device)
        betas = torch.zeros((batch_size, 10))
        smpl_outputs = self.body_model(betas=betas, body_pose=body_pose_t, jaw_pose=jaw_pose_t, 
                                        left_hand_pose=left_hand_pose_t, right_hand_pose=right_hand_pose_t,
                                        leye_pose=leye_pose_t, reye_pose=reye_pose_t, expression=expression_t,
                                        transl=transl, global_orient=global_orient)
        
        tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        vs_template = smpl_outputs.vertices
        smpl_faces = torch.as_tensor(self.body_model.faces.astype(np.int64))
        pose_offset_cano = torch.matmul(smpl_outputs.pose_feature, self.init_podir).reshape(1, -1, 3)
        pose_offset_cano = torch.cat([pose_offset_cano[:, self.init_faces[..., i]] for i in range(3)], dim=1).mean(1)
        self.register_buffer('tfs_inv_t', tfs_inv_t, persistent=False)
        self.register_buffer('vs_template', vs_template, persistent=False)
        self.register_buffer('smpl_faces', smpl_faces, persistent=False)
        self.register_buffer('pose_offset_cano', pose_offset_cano, persistent=False)

        # initialize SNARF
        # self.deformer.device = betas.device
        smpl_verts = smpl_outputs.vertices.float().detach().clone()
        #TODO: add batch operation
        smpl_verts = smpl_verts[0][None,:,:]

        self.deformer.switch_to_explicit(resolution=64,
                                         smpl_verts=smpl_verts,
                                         smpl_faces=self.smpl_faces,
                                         smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
                                         use_smpl=True)

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)
        # self.deformer.voxel_d = self.deformer.voxel_d.type(self.dtype)
        # self.deformer.voxel_J = self.deformer.voxel_J.type(self.dtype)

    def prepare_deformer(self, smpl_params=None, num_scenes=1, device=None):
        _, _, _, _, betas, _, _, _, _, _, _ = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
        smpl_params_cano = torch.zeros_like(smpl_params).expand(6, -1)
        # smpl_params_cano = torch.zeros_like(smpl_params)
        betas = betas.repeat(6, 1)
        betas[0] -= 0.5
        # betas[-1] += 0.5
        # betas[1] += 0
        # betas[2] -= 0.2
        device = smpl_params.device
        smpl_params = None
        if smpl_params is None:
            # path = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_train/0021/smplx/smplx_param.pkl'
            # path = '/home/zhangweitian/HighResAvatar/ani_file/mesh-f00101.json'
            # path = '/home/zhangweitian/HighResAvatar/ani_file/mesh-f00080_smplx.pkl'
            # face_path = '/home/zhangweitian/HighResAvatar/ani_file/pexels-megan-markham-2448525_param.pkl'
            # scale, transl, global_orient, pose, betas
            # with open(path, 'rb') as f:
            #     smpl_param_data = pickle.load(f)
            # with open(face_path, 'rb') as f:
            #     flame_param_data = pickle.load(f)
                # jaw_pose = torch.as_tensor(flame_param_data['pose'][:, 6:9]).to(device)
                # leye_pose = torch.as_tensor(flame_param_data['pose'][:, 9:12]).to(device)
                # reye_pose = torch.as_tensor(flame_param_data['pose'][:, 12:15]).to(device)
                # expression = torch.as_tensor(flame_param_data['betas'][:, -50:]).to(device)
            # with open(path, 'rb') as f:
            #     smpl_param_data = json.load(f)
            # smpl_param = np.concatenate([np.array([1.]).reshape(1, -1), 
            #         np.array(smpl_param_data['global_orient']).reshape(1, -1), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
            #         np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
            #         np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
            #         np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)
            # jaw_pose = matrix_to_axis_angle(torch.as_tensor(flame_param_data['jaw_pose']))
            # smpl_param = np.concatenate([np.array([1.]).reshape(1, -1), 
            #         np.array(smpl_param_data['global_orient']).reshape(1, -1), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
            #         np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
            #         np.array(jaw_pose).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
            #         np.array(flame_param_data['exp']).reshape(1, -1)], axis=1)
            # smpl_params = torch.as_tensor(smpl_param).to(device).expand(num_scenes, -1).float()
            # smpl_params = torch.zeros((num_scenes, 120)).to(device)
            # scale, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            scale, _, global_orient, pose, _, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params_cano, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            # scale, _, global_orient, pose, _, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            # betas = betas * 0 + 0
            # transl = torch.as_tensor([[-0.0012,  0.4668, -0.0127],]).to(device).expand(num_scenes, -1)
            transl = torch.as_tensor([[0., 0.35, 0.],]).to(device).repeat(num_scenes, 1)
            pose[:, 47] = -np.pi / 4
            pose[:, 50] = np.pi / 4
            pose[:, 38] = -np.pi / 9
            pose[:, 41] = np.pi / 9
            # left_hand_pose = self.body_model.left_hand_mean_data.unsqueeze(0).to(device).repeat(num_scenes, 1)
            # right_hand_pose = self.body_model.right_hand_mean_data.unsqueeze(0).to(device).repeat(num_scenes, 1)
            # left_hand_pose = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
            #     -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device).repeat(num_scenes, 1)
            # right_hand_pose = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
            #     -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device).repeat(num_scenes, 1)
            # jaw_pose[:, 0] = 0.5
            smpl_params = {
                'betas': betas,
                'expression': expression,
                'body_pose': pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                'jaw_pose': jaw_pose,
                'leye_pose': leye_pose,
                'reye_pose': reye_pose,
                'global_orient': global_orient,
                'transl': transl,
                'scale': scale,
            }
            # smpl_params = {
            #     'betas': betas,
            #     'expression': expression,
            #     'body_pose': pose,
            #     'left_hand_pose': left_hand_pose,
            #     'right_hand_pose': right_hand_pose,
            #     'jaw_pose': jaw_pose,
            #     'leye_pose': leye_pose,
            #     'reye_pose': reye_pose,
            #     'global_orient': global_orient,
            #     'transl': transl,
            #     'scale': scale,
            # }
            # smplx_params = {
            #     'betas': betas.reshape(-1, 10),
            #     'body_pose': pose,
            #     'global_orient': torch.zeros((1,3)).to(device),
            #     'transl': transl.to(device),
            # }
            # smpl_params = {
            # 'betas': torch.zeros((1,10)).cuda(),
            # 'body_pose': pose,
            # 'global_orient': torch.zeros((1,3)).cuda(),
            # 'transl': transl.cuda(),
            # # 'transl': torch.zeros((1,)).cuda(),
            # # 'scale': torch.tensor([1.8/1.6,]).cuda()
            # }
            
        else:
            # left_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
            # -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device)
            # right_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
            # -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device)
            # global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas = torch.split(smpl_params, [3, 63, 45, 45, 3, 3, 3, 10], dim=1)
            # global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas = torch.split(smpl_params, [3, 63, 12, 12, 3, 3, 3, 10], dim=1)
            # expression = torch.zeros(4, 10).to(device).float()
            # transl = torch.as_tensor([[0., 0.35, 0.],]).to(device).repeat(num_scenes, 1).float()
            # # global_orient = torch.as_tensor([[0., 0., 0.],]).to(device).repeat(num_scenes, 1).float()
            # scale = torch.as_tensor([[1.,],]).to(device).repeat(num_scenes, 1).float()
            # path = '/home/zhangweitian/HighResAvatar/ani_file/mesh-f00089_smplx.pkl'
            # with open(path, 'rb') as f:
            #     smpl_param_data = pickle.load(f)
            # smpl_param = np.concatenate([np.array([1.]).reshape(1, -1), 
            #         np.array(smpl_param_data['global_orient']).reshape(1, -1), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
            #         np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
            #         np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
            #         np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)
            # smpl_params_exp = torch.as_tensor(smpl_param).to(device).expand(num_scenes, -1).float()
            # _, _, _, _, _, _, jaw_pose, _, _, expression = torch.split(smpl_params_exp, [1, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            # smpl_params = smpl_params.expand(5, -1)
            if smpl_params.shape[1] == 179 or smpl_params.shape[1] == 175:
                # hands_12 = True
                # global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas = torch.split(smpl_params, [3, 63, 45, 45, 3, 10, 10], dim=1)
                global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas = torch.split(smpl_params, [3, 63, 45, 45, 3, 3, 3, 10], dim=1)
                expression = torch.zeros(num_scenes, 10).to(device).float()
                transl = torch.as_tensor([[0., 0.35, 0.],]).to(device).repeat(num_scenes, 1).float()
                scale = torch.as_tensor([[1.,],]).to(device).repeat(num_scenes, 1).float()
                # leye_pose = torch.zeros(num_scenes, 3).to(device).float()
                # reye_pose = torch.zeros(num_scenes, 3).to(device).float()
                smpl_params = {
                    'betas': betas.reshape(-1, 10),
                    'expression': expression.reshape(-1, 10),
                    'body_pose': pose.reshape(-1, 63),
                    'left_hand_pose': left_hand_pose.reshape(-1, 45),
                    'right_hand_pose': right_hand_pose.reshape(-1, 45),
                    'jaw_pose': jaw_pose.reshape(-1, 3),
                    'leye_pose': leye_pose.reshape(-1, 3),
                    'reye_pose': reye_pose.reshape(-1, 3),
                    'global_orient': global_orient.reshape(-1, 3),
                    'transl': transl.reshape(-1, 3),
                    'scale': scale.reshape(-1, 1)
                }
            else:
                scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
                # scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
                # transl = transl * 0 
                # transl[:, 1] += 0.35
                # global_orient = global_orient * 0
            # scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            # scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(self.smplx_params, [1, 3, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
                smpl_params = {
                    'betas': betas.reshape(-1, 10),
                    'expression': expression.reshape(-1, 10),
                    'body_pose': pose.reshape(-1, 63),
                    'left_hand_pose': left_hand_pose.reshape(-1, 45),
                    'right_hand_pose': right_hand_pose.reshape(-1, 45),
                    'jaw_pose': jaw_pose.reshape(-1, 3),
                    'leye_pose': leye_pose.reshape(-1, 3),
                    'reye_pose': reye_pose.reshape(-1, 3),
                    'global_orient': global_orient.reshape(-1, 3),
                    'transl': transl.reshape(-1, 3),
                    'scale': scale.reshape(-1, 1)
                }
            # left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose = torch.split(self.op_hf_pose, [12, 12, 3, 3, 3], dim=1)
            # smpl_params = {
            #     'betas': self.op_betas.reshape(-1, 10),
            #     'expression': self.op_expression.reshape(-1, 10),
            #     'body_pose': self.op_pose.reshape(-1, 63),
            #     'left_hand_pose': left_hand_pose.reshape(-1, 12),
            #     'right_hand_pose': right_hand_pose.reshape(-1, 12),
            #     'jaw_pose': jaw_pose.reshape(-1, 3),
            #     'leye_pose': leye_pose.reshape(-1, 3),
            #     'reye_pose': reye_pose.reshape(-1, 3),
            #     'global_orient': self.op_global_orient.reshape(-1, 3),
            #     'transl': self.op_transl.reshape(-1, 3),
            #     'scale': self.op_scale.reshape(-1, 1)
            # }
            # print(self.op_betas)
            # print(self.op_betas.requires_grad)
            # print(self.op_betas.grad)
        # if device == None:
        # device = smpl_params["betas"].device
        # if next(self.body_model.parameters()).device != device:
        if self.body_model.lbs_weights.device != device:
            self.body_model = self.body_model.to(device)
            assert False
        
        
        if not self.initialized:
            self.initialize(smpl_params["betas"])
            self.initialized = True
    
        
        # smpl_outputs = self.body_model(betas=smpl_params["betas"],
        #                                body_pose=smpl_params["body_pose"],
        #                                global_orient=smpl_params["global_orient"],
        #                                transl=smpl_params["transl"],
        #                                scale = smpl_params["scale"] if "scale" in smpl_params.keys() else None)
        # if hands_12:
        #     smpl_outputs = self.body_model_12(**smpl_params)
        # else:
        # smpl_outputs = self.body_model_12(**smpl_params)
        smpl_outputs = self.body_model(**smpl_params)
        
        self.smpl_outputs = smpl_outputs
        
        tfs = (smpl_outputs.A @ self.tfs_inv_t.expand(smpl_outputs.A.shape[0],-1,-1,-1))
        # self.deformer.precompute(tfs)
        self.tfs = tfs
        self.tfs_A = smpl_outputs.A

        self.shape_offset = torch.einsum('bl,mkl->bmk', [smpl_outputs.betas, self.init_spdir])
        self.pose_offset = torch.matmul(smpl_outputs.pose_feature, self.init_podir).reshape(self.shape_offset.shape) # batch_size, 
        # print(self.shape_offset.shape)
        # self.pose_offset = torch.matmul(smpl_outputs.pose_feature, self.init_podir).view(num_scenes, -1, 3)
        # self.pose_offset = smpl_outputs.pose_offset
        # self.shape_offset = smpl_outputs.shape_offset
        # print(self.shape_offset.shape)
        # assert False
        # self.faces = self.body_model.faces

    def __call__(self, pts_in, mask=None, cano=True, eval_mode=True, render_skinning=False, is_normal=True):
        pts = pts_in.clone()

        if cano:
            # mask = (pts[:,:,2] < 0.4) & (pts[:,:,2] > -0.4)
            return pts, None
        else:
            init_faces = self.init_faces
            # shape_offset = self.shape_offset.expand(pts.shape[0], -1, -1)
            # init_idx = self.init_idx[valid]
            # init_bar = self.init_bar[valid].unsqueeze(0)

        b, n, _ = pts.shape
        # dist_sq, idx, neighbors = ops.knn_points(pts.float(), self.vs_template.float()[:,::10].expand(b, -1, -1), K=1)
        # mask = dist_sq < self.threshold ** 2
        # mask = torch.ones([b, n, 1]).bool()

        smpl_nn = False

        if smpl_nn:
            # deformer based on SMPL nearest neighbor search
            k = 1
            try:
                dist_sq, idx, neighbors = ops.knn_points(pts, self.smpl_outputs.vertices.float().expand(b, -1, -1), K=k, return_nn=True)
            except:
                print(pts.shape)
                print(self.smpl_outputs.vertices.shape)
                assert False
            
            # dist = dist_sq[0].sqrt().clamp_(0.00001,1.)
            # dist = dist_sq.sqrt().clamp_(0.00001,1.)
            dist = dist_sq.sqrt().clamp_(0.00003, 0.1)
            weights = self.body_model.lbs_weights.clone()[idx]
            # mask = dist_sq < 0.02

            ws=1./dist
            ws=ws/ws.sum(-1,keepdim=True)
            weights = (ws[..., None]*weights).sum(2).detach()

            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pts += shape_offset
            pts_cano_all, w_tf = skinning(pts, weights, self.tfs, inverse=False)
            pts_cano_all = pts_cano_all.unsqueeze(2)
            # others = {"valid_ids": torch.zeros_like(pts_cano_all)[...,0]}
            # others["valid_ids"][:,:,0] = mask[...,0]
            # others["valid_ids"] = others["valid_ids"].bool()
            
        else:
            # defromer based on fast-SNARF
            # with torch.no_grad():
                # try:
                # print(pts.min(1))
                # print(pts.max(1))
                # pts_cano_all, others = self.deformer.forward(pts, cond=None, mask=mask, tfs=tfs, eval_mode=eval_mode)
            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pose_offset = torch.cat([self.pose_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            # pts += shape_offset
            pts_cano_all, w_tf = self.deformer.forward_skinning(pts, shape_offset, pose_offset, cond=None, tfs=self.tfs_A, tfs_inv=self.tfs_inv_t, poseoff_ori=self.pose_offset_cano, lbsw=self.init_lbsw, mask=mask)
                # print(pts_cano_all.flatten(1,2).min(1))
                # print(pts_cano_all.flatten(1,2).max(1))
                # assert False
                # except:
                #     print(pts.shape)
                #     print(pts.device)
                #     print(mask.device)
                #     print(mask.shape)
                #     print(eval_mode)
                #     assert False
        pts_cano_all = pts_cano_all.reshape(b, n, -1, 3)
        # valid = others["valid_ids"].reshape(b, n, -1)
        # pts_cano_all = pts_cano_all.reshape(b, n, -1, 3)
        # valid = others["valid_ids"].reshape(b, n, -1)

        # rgb_cano, sigma_cano, grad_cano, grad_pred_cano = model(pts_cano_all, valid)

        # sigma_cano, idx = torch.max(sigma_cano.squeeze(-1), dim=-1)

        # pts_cano = torch.gather(pts_cano_all, 2, idx[:, :, None, None].repeat(1,1,1,pts_cano_all.shape[-1]))
        # rgb_cano = torch.gather(rgb_cano, 2, idx[:, :, None, None].repeat(1,1,1,rgb_cano.shape[-1]))
        # if is_normal:
        #     grad_cano = torch.gather(grad_cano, 2, idx[:, :, None, None].repeat(1,1,1,grad_cano.shape[-1]))
        #     grad = self.deformer.skinning_normal(pts_cano.squeeze(2), grad_cano.squeeze(2), self.tfs)
        #     grad_pred_cano = torch.gather(grad_pred_cano, 2, idx[:, :, None, None].repeat(1,1,1,grad_cano.shape[-1]))
        # else:
        #     grad = None
        #     grad_pred_cano = None
            
        # return rgb_cano, sigma_cano.unsqueeze(-1), grad, grad_pred_cano
        if pts_in.dim() == 2:
            assert False
            pts_cano_all = pts_cano_all.squeeze(0)
        return pts_cano_all, w_tf.clone()