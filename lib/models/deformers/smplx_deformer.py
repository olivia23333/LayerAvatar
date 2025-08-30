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

        self.deformer = ForwardDeformer()
        
        # threshold for rendering (need to be larger for loose clothing)
        self.threshold = 0.12

        init_spdir = torch.as_tensor(np.load('work_dirs/cache/init_spdir_smplx_full.npy'))
        self.register_buffer('init_spdir', init_spdir, persistent=False)

        init_podir = torch.as_tensor(np.load('work_dirs/cache/init_podir_smplx_full.npy'))
        self.register_buffer('init_podir', init_podir, persistent=False)
        
        init_faces = torch.as_tensor(np.load('work_dirs/cache/init_faces_smplx_full.npy'))
        self.register_buffer('init_faces', init_faces.unsqueeze(0), persistent=False)

        init_lbs_weights = torch.as_tensor(np.load('work_dirs/cache/init_lbsw_smplx_body.npy'))
        self.register_buffer('init_lbsw', init_lbs_weights.unsqueeze(0), persistent=False)

        self.initialize()
        self.initialized = True


    def initialize(self):
        
        batch_size = 1
        
        # canonical space is defined in t-pose / star-pose
        body_pose_t = torch.zeros((batch_size, 63))
        # body_pose_t[0, 2] = np.pi / 18
        # body_pose_t[0, 5] = -np.pi / 18

        jaw_pose_t = torch.zeros((batch_size, 3))
        # jaw_pose_t[:, 0] = 0.2

        left_hand_pose_t = torch.zeros((batch_size, 45))
        right_hand_pose_t = torch.zeros((batch_size, 45))
        leye_pose_t = torch.zeros((batch_size, 3))
        reye_pose_t = torch.zeros((batch_size, 3))
        expression_t = torch.zeros((batch_size, 10))
        transl = torch.as_tensor([0., 0.35, 0.]).unsqueeze(0)
        global_orient = torch.zeros((batch_size, 3))
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

    def prepare_deformer(self, smpl_params=None, num_scenes=1, device=None):
        if smpl_params is None:
            smpl_params = torch.zeros((num_scenes, 189)).to(device)
            scale, _, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            transl = torch.as_tensor([[0., 0.35, 0.],]).to(device).repeat(num_scenes, 1)
            pose[:, 47] = -np.pi / 4
            pose[:, 50] = np.pi / 4
            pose[:, 38] = -np.pi / 9
            pose[:, 41] = np.pi / 9

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
            
        else:
            if smpl_params.shape[1] == 179 or smpl_params.shape[1] == 175:
                # hands_12 = True
                if smpl_params.shape[1] == 179:
                    global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas = torch.split(smpl_params, [3, 63, 45, 45, 3, 10, 10], dim=1)
                    leye_pose = torch.zeros(num_scenes, 3).to(device).float()
                    reye_pose = torch.zeros(num_scenes, 3).to(device).float()
                else:
                    global_orient, pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas = torch.split(smpl_params, [3, 63, 45, 45, 3, 3, 3, 10], dim=1)
                    expression = torch.zeros(num_scenes, 10).to(device).float()
                transl = torch.as_tensor([[0., 0.35, 0.],]).to(device).repeat(num_scenes, 1).float()
                scale = torch.as_tensor([[1.,],]).to(device).repeat(num_scenes, 1).float()
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

        if not self.initialized:
            self.initialize(smpl_params["betas"])
            self.initialized = True
    
        smpl_outputs = self.body_model(**smpl_params)
        
        self.smpl_outputs = smpl_outputs
        
        tfs = (smpl_outputs.A @ self.tfs_inv_t.expand(smpl_outputs.A.shape[0],-1,-1,-1))
        self.tfs = tfs
        self.tfs_A = smpl_outputs.A

        self.shape_offset = torch.einsum('bl,mkl->bmk', [smpl_outputs.betas, self.init_spdir])
        self.pose_offset = torch.matmul(smpl_outputs.pose_feature, self.init_podir).reshape(self.shape_offset.shape) # batch_size, 

    def __call__(self, pts_in, mask=None, cano=True, eval_mode=True, render_skinning=False, is_normal=True):
        pts = pts_in.clone()

        if cano:
            return pts, None
        else:
            init_faces = self.init_faces

        b, n, _ = pts.shape

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
            
            dist = dist_sq.sqrt().clamp_(0.00003, 0.1)
            weights = self.body_model.lbs_weights.clone()[idx]

            ws=1./dist
            ws=ws/ws.sum(-1,keepdim=True)
            weights = (ws[..., None]*weights).sum(2).detach()

            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pts += shape_offset
            pts_cano_all, w_tf = skinning(pts, weights, self.tfs, inverse=False)
            pts_cano_all = pts_cano_all.unsqueeze(2)
            
        else:
            # defromer based on fast-SNARF
            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pose_offset = torch.cat([self.pose_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pts_cano_all, w_tf = self.deformer.forward_skinning(pts, shape_offset, pose_offset, cond=None, tfs=self.tfs_A, tfs_inv=self.tfs_inv_t, poseoff_ori=self.pose_offset_cano, lbsw=self.init_lbsw, mask=mask)
        pts_cano_all = pts_cano_all.reshape(b, n, -1, 3)

        # if pts_in.dim() == 2:
        #     assert False
        #     pts_cano_all = pts_cano_all.squeeze(0)
        return pts_cano_all, w_tf.clone()