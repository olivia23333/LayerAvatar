'''
Deformer of AG3D adapted from fast-SNARF
'''

from .fast_snarf.lib.model.deformer_smpl import ForwardDeformer, skinning
from .smplx import SMPL
import torch
from pytorch3d import ops
import numpy as np

class SNARFDeformer(torch.nn.Module):
    
    def __init__(self, gender) -> None:
        super().__init__()
        self.body_model = SMPL('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender=gender, \
                                create_body_pose=False, \
                                create_betas=False, \
                                create_global_orient=False, \
                                create_transl=False)
        self.deformer = ForwardDeformer()
        
        # threshold for rendering (need to be larger for loose clothing)
        # self.threshold = 0.12
        self.threshold = 0.12
        # self.threshold = 0.2
        
        self.initialize()
        self.initialized = True

        init_idx = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_idx_smpl.npy'))
        self.register_buffer('init_idx', init_idx.unsqueeze(0))
        init_bar = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_bar_smpl.npy'))
        self.register_buffer('init_bar', init_bar.unsqueeze(0))

    # def initialize(self, betas):
    def initialize(self):
        
        # device = betas.device
        batch_size = 1
        
        # canonical space is defined in t-pose
        body_pose_t = torch.zeros((batch_size, 69))
        # body_pose_t[:, 2] = torch.pi / 6
        # body_pose_t[:, 5] = -torch.pi / 6
        transl = torch.zeros((batch_size, 3))
        transl[:,1] = 0.3
        global_orient = torch.zeros((batch_size, 3))
        # self.body_model = self.body_model.to(device)
        betas = torch.zeros((batch_size, 10))
        smpl_outputs = self.body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient)
        
        tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        vs_template = smpl_outputs.vertices
        smpl_faces = torch.as_tensor(self.body_model.faces.astype(np.int64))
        self.register_buffer('tfs_inv_t', tfs_inv_t)
        self.register_buffer('vs_template', vs_template)
        self.register_buffer('smpl_faces', smpl_faces)

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

    def prepare_deformer(self, smpl_params=None, device=None):
        # scale, transl, global_orient, pose, betas = torch.split(smpl_params, [1, 3, 3, 69, 10], dim=1)
        # smpl_params = None
        if smpl_params is None:
            smpl_params = torch.zeros((1, 82)).to(device)
            global_orient, pose, betas = torch.split(smpl_params, [3, 69, 10], dim=1)
            # pose = torch.zeros((1,69)).to(device)
            # pose = torch.zeros((1,69)).cuda()
            # pose[:, 2] = torch.pi / 6
            # pose[:, 5] = -torch.pi / 6
            transl = torch.zeros((1, 3))
            transl[:, 1] = 0.25

            # betas += 1

            # smpl_params = {
            # 'betas': ,
            # 'body_pose': pose,
            # 'global_orient': torch.zeros((1,3)).to(device),
            # 'transl': torch.zeros((1,)).to(device),
            # }
            smpl_params = {
            # 'betas': torch.zeros((1,10)).to(device),
                'betas': betas,
                'body_pose': pose,
                'global_orient': global_orient,
                'transl': transl.to(device),
            # 'transl': torch.zeros((1,)).cuda(),
            # 'scale': torch.tensor([1.8/1.6,]).cuda()
            }
            # smpl_params = {
            # 'betas': torch.zeros((1,10)).cuda(),
            # 'body_pose': pose,
            # 'global_orient': torch.zeros((1,3)).cuda(),
            # 'transl': transl.cuda(),
            # # 'transl': torch.zeros((1,)).cuda(),
            # # 'scale': torch.tensor([1.8/1.6,]).cuda()
            # }
            
        else:
            scale, transl, global_orient, pose, betas = torch.split(smpl_params, [1, 3, 3, 69, 10], dim=1)
            smpl_params = {
                'betas': betas.reshape(-1, 10),
                'body_pose': pose.reshape(-1, 69),
                'global_orient': global_orient.reshape(-1, 3),
                'transl': transl.reshape(-1, 3),
                'scale': scale.reshape(-1, 1)
            }
        
        # if device == None:
        device = smpl_params["betas"].device
        # if next(self.body_model.parameters()).device != device:
        if self.body_model.lbs_weights.device != device:
            self.body_model = self.body_model.to(device)
            assert False
        
        
        if not self.initialized:
            self.initialize(smpl_params["betas"])
            self.initialized = True
    
        
        smpl_outputs = self.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"],
                                       scale = smpl_params["scale"] if "scale" in smpl_params.keys() else None)
        
        self.smpl_outputs = smpl_outputs
        
        tfs = (smpl_outputs.A @ self.tfs_inv_t.expand(smpl_outputs.A.shape[0],-1,-1,-1))
        # self.deformer.precompute(tfs)
        self.tfs = tfs
        self.pose_offset = smpl_outputs.pose_offset
        self.shape_offset = smpl_outputs.shape_offset

    def __call__(self, pts_in, valid=None, cano=True, eval_mode=True, render_skinning=False, is_normal=True):
        pts = pts_in.clone()

        if cano:
            # mask = (pts[:,:,2] < 0.4) & (pts[:,:,2] > -0.4)
            return pts, None
        else:
            tfs = self.tfs.expand(pts.shape[0], -1, -1, -1)
            init_idx = self.init_idx[valid]
            init_bar = self.init_bar[valid].unsqueeze(0)
        b, n, _ = pts.shape
        dist_sq, idx, neighbors = ops.knn_points(pts.float(), self.vs_template.float()[:,::10].expand(b, -1, -1), K=1)
        mask = dist_sq < self.threshold ** 2

        smpl_nn = False

        if smpl_nn:
            # deformer based on SMPL nearest neighbor search
            k = 3
            try:
                dist_sq, idx, neighbors = ops.knn_points(pts,self.smpl_outputs.vertices.float().expand(b, -1, -1),K=k,return_nn=True)
            except:
                print(pts.shape)
                print(self.smpl_outputs.vertices.shape)
                assert False
            
            dist = dist_sq[0].sqrt().clamp_(0.00001,1.)
            idx = idx[0]
            weights = self.body_model.lbs_weights.clone()[idx]
            mask = dist_sq < 0.02

            ws=1./dist
            ws=ws/ws.sum(-1,keepdim=True)

            weights = (ws[:,:,None].expand(-1,-1,24)*weights).sum(1)[None]

            pts_cano_all = skinning(pts, weights, self.tfs, inverse=True)
            pts_cano_all = pts_cano_all.unsqueeze(2).expand(-1,-1,3,-1)
            others = {"valid_ids": torch.zeros_like(pts_cano_all)[...,0]}
            others["valid_ids"][:,:,0] = mask[...,0]
            others["valid_ids"] = others["valid_ids"].bool()
            
        else:
            # defromer based on fast-SNARF
            # with torch.no_grad():
                # try:
                # print(pts.min(1))
                # print(pts.max(1))
                # pts_cano_all, others = self.deformer.forward(pts, cond=None, mask=mask, tfs=tfs, eval_mode=eval_mode)
            v0 = self.shape_offset[:, init_idx[..., 0]]
            v1 = self.shape_offset[:, init_idx[..., 1]]
            v2 = self.shape_offset[:, init_idx[..., 2]]
            shape_offset = init_bar[..., 0:1] * v0 + init_bar[..., 1:2] * v1 + init_bar[..., 2:] * v2
            pts += shape_offset
            # pts += pose_offset
            pts_cano_all, w_tf = self.deformer.forward_skinning(pts, cond=None, tfs=tfs, mask=mask[..., 0])
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
        assert False
        return pts_cano_all, mask, w_tf.clone()
