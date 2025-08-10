import numpy as np

up_mask = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/up_mask_cop.npy')
low_mask = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/low_mask_cop.npy')
hair_mask = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/hair_mask_cop.npy')
shoes_mask = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/shoes_mask_cop.npy')
body_mask = np.ones_like(up_mask)
combine_mask = np.concatenate([body_mask, up_mask, low_mask, hair_mask, shoes_mask], axis=0)
print(combine_mask.shape)
np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/all_mask.npy', combine_mask)
assert False

up_mask_v = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/up_mask_v.npy')[..., 0]
low_mask_v = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/low_mask_v.npy')[..., 0]
hair_mask_v = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/hair_mask_v.npy')[..., 0]
shoes_mask_v = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/shoes_mask_v.npy')[..., 0]

body_pcd = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_pcd_smplx_cop.npy')
up_pcd = body_pcd[up_mask]
low_pcd = body_pcd[low_mask]
hair_pcd = body_pcd[hair_mask]
shoes_pcd = body_pcd[shoes_mask]
full_pcd = np.concatenate([body_pcd, up_pcd, low_pcd, hair_pcd, shoes_pcd], axis=0)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_pcd_smplx_abl.npy', full_pcd)

# body_rot = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_rot_smplx_cop.npy')
# up_rot = body_rot[up_mask]
# low_rot = body_rot[low_mask]
# hair_rot = body_rot[hair_mask]
# shoes_rot = body_rot[shoes_mask]
# full_rot = np.concatenate([body_rot, up_rot, low_rot, hair_rot, shoes_rot], axis=0)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_rot_smplx_abl.npy', full_rot)

# body_uv = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_uv_smplx_cop.npy')
# up_uv = body_uv[up_mask]
# low_uv = body_uv[low_mask]
# hair_uv = body_uv[hair_mask]
# shoes_uv = body_uv[shoes_mask]
# full_uv = np.concatenate([body_uv, up_uv, low_uv, hair_uv, shoes_uv], axis=0)
# print(full_uv.shape)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_uv_smplx_abl.npy', full_uv)

# body_spdir = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_spdir_smplx_cop.npy')
# up_spdir = body_spdir[up_mask_v]
# low_spdir = body_spdir[low_mask_v]
# hair_spdir = body_spdir[hair_mask_v]
# shoes_spdir = body_spdir[shoes_mask_v]
# full_spdir = np.concatenate([body_spdir, up_spdir, low_spdir, hair_spdir, shoes_spdir], axis=0)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_spdir_smplx_abl.npy', full_spdir)

# body_podir = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_podir_smplx_cop.npy').reshape(486, -1, 3)
# up_podir = body_podir[:, up_mask_v]
# low_podir = body_podir[:, low_mask_v]
# hair_podir = body_podir[:, hair_mask_v]
# shoes_podir = body_podir[:, shoes_mask_v]
# full_podir = np.concatenate([body_podir, up_podir, low_podir, hair_podir, shoes_podir], axis=1).reshape(486, -1)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_podir_smplx_abl.npy', full_podir)


len_up = up_pcd.shape[0]
len_low = low_pcd.shape[0]
len_hair = hair_pcd.shape[0]
len_shoes = shoes_pcd.shape[0]
# len_b = b.shape[1]
# len_c = c.shape[1]

body_faces = np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_faces_smplx_cop.npy')
up_faces = body_faces[up_mask]
low_faces = body_faces[low_mask]
hair_faces = body_faces[hair_mask]
shoes_faces = body_faces[shoes_mask]
# b += len_a
# c += len_a + len_b

d = np.concatenate([body_faces, up_faces, low_faces, hair_faces, shoes_faces], axis=0)
np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/init_faces_smplx_abl.npy', d)

# a = np.ones_like(body_pcd)[..., 0].astype(np.int32)
# b = np.ones_like(up_pcd)[..., 0].astype(np.int32) * 2
# c = np.ones_like(low_pcd)[..., 0].astype(np.int32) * 5
# d = np.ones_like(hair_pcd)[..., 0].astype(np.int32) * 3
# e = np.ones_like(shoes_pcd)[..., 0].astype(np.int32) * 4
# f = np.concatenate([a, b, c, d, e], axis=0)
# print(f.shape)
# print(f.min())
# print(f.max())
# print(f.dtype)
# np.save('/mnt/sdb/zwt/LayerAvatar/work_dirs/ablation_cache/part_label_cop.npy', f)