import os
import json
import numpy as np
import cv2 as cv
import torch
import smplx  # (please setup the official SMPL-X model according to: https://pypi.org/project/smplx/)

# subject = './avatarrex_zzr'
subject = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_real_1/0001'
# subject = './avatarrex_lbn1'
# subject = './avatarrex_lbn2'

# initialize smpl model
smpl = smplx.SMPLX(model_path = '/home/zhangweitian/HighResAvatar/lib/models/deformers/smplx/SMPLX', gender = 'neutral', use_pca = False, flat_hand_mean = True, num_betas=10, batch_size = 1)

# load camera data
with open(os.path.join(subject, 'calibration_full.json'), 'r') as fp:
    cam_data = json.load(fp)

# load smpl data
smpl_data = np.load(os.path.join(subject, 'smpl_params.npz'), allow_pickle = True)
smpl_data = dict(smpl_data)
smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}

# frame_num = smpl_data['body_pose'].shape[0]
# for frame_id in range(0, frame_num, 30):
frame_id = 77
smpl_out = smpl.forward(
    global_orient = smpl_data['global_orient'][frame_id].unsqueeze(0),
    transl = smpl_data['transl'][frame_id].unsqueeze(0),
    body_pose = smpl_data['body_pose'][frame_id].unsqueeze(0),
    jaw_pose = smpl_data['jaw_pose'][frame_id].unsqueeze(0),
    betas = smpl_data['betas'][0].unsqueeze(0),
    expression = smpl_data['expression'][frame_id].unsqueeze(0),
    left_hand_pose = smpl_data['left_hand_pose'][frame_id].unsqueeze(0),
    right_hand_pose = smpl_data['right_hand_pose'][frame_id].unsqueeze(0),
)
smpl_verts = smpl_out.vertices  # smpl vertices in live poses
smpl_verts = smpl_verts.detach().cpu().numpy().squeeze(0)
print(smpl_verts.shape)
print(smpl_verts.min(0))
print(smpl_verts.max(0))

smpl_proj_vis = []

cam_sn = '22010716'



img = cv.imread('/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_real_1/0001/rgb/0001.jpg', cv.IMREAD_UNCHANGED)
img_ = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# transform smpl from world to camera
cam_R = np.array(cam_data[cam_sn]['R']).astype(np.float32).reshape((3, 3))
cam_t = np.array(cam_data[cam_sn]['T']).astype(np.float32).reshape((3,))
K = np.array(cam_data[cam_sn]['K']).astype(np.float32).reshape((3, 3))
M = np.eye(3)
M[0, 2] = (K[0, 2] - 1500 / 2) / K[0, 0]
M[1, 2] = (K[1, 2] - 2048 / 2) / K[1, 1]
K[0, 2] = 1500 / 2
K[1, 2] = 2048 / 2
cam_R = M @ cam_R
cam_t = M @ cam_t
print(cam_R)
print(cam_t)
# cam_t = np.array([0.1579, -0.5851,  0.8631])
# smpl_verts_cam = np.matmul(smpl_verts, cam_R) + cam_t.reshape(1, 3)
smpl_verts_cam = np.matmul(smpl_verts, cam_R.transpose()) + cam_t.reshape(1, 3)

# project smpl vertices to the image
cam_K = np.array(cam_data[cam_sn]['K']).astype(np.float32).reshape((3, 3))
cam_K *= np.array([img_.shape[1] / img.shape[1], img_.shape[0] / img.shape[0], 1.0], dtype = np.float32).reshape(3, 1)
cam_K[0, 2] = 375
cam_K[1, 2] = 512
smpl_verts_proj = np.matmul(smpl_verts_cam / smpl_verts_cam[:, 2:], cam_K.transpose())

# visualize the projection
smpl_verts_proj = np.round(smpl_verts_proj).astype(np.int32)
smpl_verts_proj[:, 0] = np.clip(smpl_verts_proj[:, 0], 0, img_.shape[1] - 1)
smpl_verts_proj[:, 1] = np.clip(smpl_verts_proj[:, 1], 0, img_.shape[0] - 1)

for v in smpl_verts_proj:
    # img_[v[1], v[0], :] = np.array([255, 255, 255], dtype = np.uint8)
    img_[v[1], v[0], :] = np.array([0, 0, 0], dtype = np.uint8)

print(img_.shape)
cv.imwrite('/home/zhangweitian/HighResAvatar/debug/viz_0001.jpg', img_)