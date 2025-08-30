import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='HumanData/THuman/THuman2.0_Release')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    root_path = args.root_path
    l = sorted(os.listdir(root_path))
    if 'THuman2_1' in root_path:
        l = l[526:]

    for file in tqdm(l):
        seg_path = os.path.join(root_path, file, '18views_3_wbg', 'new_mask')
        img_list = os.listdir(seg_path)
        os.makedirs(os.path.join(root_path, file, '18views_3_wbg', 'seg'), exist_ok=True)
        for img_name in img_list:
            npy_path = os.path.join(seg_path, img_name)
            seg = np.asarray(Image.open(npy_path).convert('L'))
            seg_part = [seg==(i+1) for i in range(5)]
            seg_part.append(seg)
            seg = np.stack(seg_part, axis=-1)
            cv2.imwrite(os.path.join(root_path, file, '18views_3_wbg', 'seg', img_name[:-4]+'_0'+'.png'), seg[..., :3])
            cv2.imwrite(os.path.join(root_path, file, '18views_3_wbg', 'seg', img_name[:-4]+'_1'+'.png'), seg[..., 3:])
