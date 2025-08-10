import numpy as np
# a = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_rot_smplx_body.npy')
# b = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_rot_smplx_l1.npy')
# c = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_rot_smplx_l2.npy')
# d = np.concatenate([a, b, c], axis=0)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_rot_smplx_full.npy', d)

a_pcd = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_pcd_smplx_body.npy')
b_pcd = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_pcd_smplx_l1.npy')
c_pcd = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_pcd_smplx_l2.npy')
d_pcd = np.concatenate([a_pcd, b_pcd, c_pcd], axis=0)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_pcd_smplx_full.npy', d_pcd)

# a = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_uv_smplx_body.npy')
# b = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_uv_smplx_l1.npy')
# c = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_uv_smplx_l2.npy')
# a[:, 0] /= 3
# b[:, 0] /= 3
# b[:, 0] += 1/3
# c[:, 0] /= 3
# c[:, 0] += 2/3
# d = np.concatenate([a, b, c], axis=0)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_uv_smplx_full.npy', d)

# a = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/body_spdir_smplx_cop.npy')
# b = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/hair_shoe_spdir_smplx_cop.npy')
# c = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/uplow_spdir_smplx_cop.npy')
# d = np.concatenate([a, b, c], axis=0)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_spdir_smplx_full.npy', d)

# a = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/body_podir_smplx_cop.npy').reshape(486, -1, 3)
# b = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/hair_shoe_podir_smplx_cop.npy').reshape(486, -1, 3)
# c = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/uplow_podir_smplx_cop.npy').reshape(486, -1, 3)
# d = np.concatenate([a, b, c], axis=1).reshape(486, -1)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_podir_smplx_full.npy', d)

# len_a = a.shape[1]
# len_b = b.shape[1]
# len_c = c.shape[1]

# a = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_faces_smplx_body.npy')
# b = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_faces_smplx_l1.npy')
# c = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_faces_smplx_l2.npy')
# b += len_a
# c += len_a + len_b

# d = np.concatenate([a, b, c], axis=0)
# np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_faces_smplx_full.npy', d)

a = np.ones_like(a_pcd)[..., 0].astype(np.int32)
b = np.zeros_like(b_pcd)[..., 0].astype(np.int32)
hair_mask = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/hair_mask_l1.npy')
shoes_mask = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/shoes_mask_l1.npy')
b[hair_mask] = 3
b[shoes_mask] = 4
c = np.zeros_like(c_pcd)[..., 0].astype(np.int32)
low_mask = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/low_mask_l2.npy')
up_mask = np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/up_mask_l2.npy')
c[low_mask] = 5
c[up_mask] = 2
d = np.concatenate([a, b, c], axis=0)
print(d.shape)
np.save('/home/zhangweitian/HighResAvatar/work_dirs/cache/part_label_full.npy', d)