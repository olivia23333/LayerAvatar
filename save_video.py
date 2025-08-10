import os
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image

# subs = os.listdir('/home/zhangweitian/HighResAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_tri_thuman/supp_video_nv')

# for sub in tqdm(subs):
#     sub_name = sub[6:10]
img_all = []
for i in range(50):
    # path = f'/home/zhangweitian/HighResAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_thuman_conv_final/viz_uncond/scene_0024_{str(i).zfill(3)}.png'
    # path = f'/home/zhangweitian/HighResAvatar/trans_viz/{str(i)}.jpg'
    # path = f'/home/zhangweitian/HighResAvatar/cloth_trans/{i}.png'
    # path = f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_wobug/viz_uncond_part/scene_0000_{str(i).zfill(3)}.png'
    # path = f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_thu/viz_uncond_/scene_0000_{str(i).zfill(3)}.png'
    path = f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_wobug/viz_uncond_ani/scene_0335_{str(i).zfill(3)}.png'
    # path = f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_wobug/viz_uncond_video/scene_0093_{str(i).zfill(3)}.png'
    # path = f'/mnt/sdb/zwt/LayerAvatar/work_dirs/stage1_avatar_16bit_finalfit_sample4_depth_baseline_huber/viz_cond_sample0000/scene_0000_{str(i).zfill(3)}.png'
    img = np.array(Image.open(path))
    # img = img.reshape(1024, 5, 1024, 4).transpose(1, 0, 2, 3)
    # img = img[:, :, 256:768, :].transpose(1, 0, 2, 3).reshape(1024, 512*5, 4)
    img_all.append(img)

# img_save = img_all[34:] + img_all[:34]
    # img_save = img_all[:1] + img_all[1:][::-1]
img_save = img_all
# imageio.mimsave(f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_wobug/video_full/video_0093.mp4', img_save, codec='libx264')
imageio.mimsave(f'/mnt/sdb/zwt/LayerAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_composite_new_wobug/viz_video_ani/video_0034.mp4', img_save, codec='libx264')
# imageio.mimsave('/home/zhangweitian/HighResAvatar/trans_viz/tryon.mp4', img_save, codec='libx264')