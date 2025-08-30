import numpy as np
import glob
import os
import cv2
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
    # root_path = "HumanData/THuman/THuman2.0_Release"
    # root_path = "HumanData/THuman2_1/model"
    # root_path = "HumanData/CustomHumans/mesh"
    # sublist = sorted(os.listdir(root_path))[526:]
    if 'THuman2_1' not in root_path:
        sublist = sorted(os.listdir(root_path))
    else:
        sublist = sorted(os.listdir(root_path))[526:]

    num_classes = 28
    palette = [
        [0, 0, 0], # 0, Background
        [3, 3, 3], # 1, Apparel
        [1, 1, 1], # 2, Face_Neck
        [3, 3, 3], # 3, Hair
        [1, 1, 1], # 4, Left_Foot
        [1, 1, 1], # 5, Left_Hand
        [1, 1, 1], # 6, Left_Lower_Arm
        [1, 1, 1], # 7, Left_Lower_Leg
        [4, 4, 4], # 8, Left_Shoe
        [4, 4, 4], # 9, Left_Sock
        [1, 1, 1], # 10, Left_Upper_Arm
        [1, 1, 1], # 11, Left_Upper_Leg
        [5, 5, 5], # 12, Lower_Clothing
        [1, 1, 1], # 13, Right_Foot
        [1, 1, 1], # 14, Right_Hand
        [1, 1, 1], # 15, Right_Lower_Arm
        [1, 1, 1], # 16, Right_Lower_Leg
        [4, 4, 4], # 17, Right_Shoe
        [4, 4, 4], # 18, Right_Sock
        [1, 1, 1], # 19, Right_Upper_Arm
        [1, 1, 1], # 20, Right_Upper_Leg
        [1, 1, 1], # 21, Torso
        [2, 2, 2], # 22, Upper_Clothing
        [1, 1, 1], # 23, Lower_Lip
        [1, 1, 1], # 24, Upper_Lip
        [1, 1, 1], # 25, Lower_Teeth
        [1, 1, 1], # 26, Upper_Teeth
        [1, 1, 1], # 27, Tongue
    ]


    for sub in tqdm(sublist):
        # print(sub)
        # if os.path.exists(os.path.join(root_path, sub, '18views_3_wbg', 'new_mask_refine')):
        #     continue
        mask_path = os.path.join(root_path, sub, '18views_3_wbg', 'mask')
        filelist = glob.glob(os.path.join(mask_path, '*_seg.npy'))
        os.makedirs(os.path.join(root_path, sub, '18views_3_wbg', 'new_mask'), exist_ok=True)
        os.makedirs(os.path.join(root_path, sub, '18views_3_wbg', 'new_mask_viz'), exist_ok=True)
        
        refine_filelist = []
        problem_filelist = []
        for file in filelist:
            sem_seg = np.load(file)

            ids = np.unique(sem_seg)[::-1]
            # if 1 in ids:
            #     seg_label_coord = np.argwhere(sem_seg == 1)
            #     min_coord = seg_label_coord.min(0)
            #     max_coord = seg_label_coord.max(0)
            #     area = (max_coord[1] - min_coord[1]) * (max_coord[0] - min_coord[0])
            #     extract_label = sem_seg[min_coord[0]-5:max_coord[0]+5, min_coord[1]-5:max_coord[1]+5]
            #     select_label = extract_label[extract_label!=1]
            #     unique_values, counts = np.unique(select_label, return_counts=True)
            #     predict_label = unique_values[np.argmax(counts)]
            #     refine_filelist.append(sub)
            #     if area > 9000:
            #         problem_filelist.append(sub)
            legal_indices = ids < num_classes
            ids = ids[legal_indices]
            labels = np.array(ids, dtype=np.int64)
            colors = []
            for label in labels:
                # if label == 1:
                #     colors.append(palette[predict_label])
                # else:
                colors.append(palette[label])
            # colors = [palette[label] for label in labels]
            mask = np.zeros([1024, 1024, 3], dtype=np.uint8)
            img = Image.open(file.replace('mask', 'image').replace('_seg.npy', '.png'))
            img = np.asarray(img)
            ori_mask = img[..., -1]>0
            # ori_mask = cv2.imread(file.replace('clothing', 'masks').replace('_seg.npy', '.png'), cv2.IMREAD_GRAYSCALE)
            # ori_mask = ori_mask > 128
            for label, color in zip(labels, colors):
                mask[sem_seg == label, :] = color
            mask = mask * ori_mask[..., None].repeat(3, axis=2)
            cv2.imwrite(file.replace('mask', 'new_mask').replace('_seg.npy', '.png'), mask)
            cv2.imwrite(file.replace('mask', 'new_mask_viz').replace('_seg.npy', '.png'), mask*50)

    # np.save('/mnt/sdc/zwt_data/HumanData/combine_dataset/refine_filelist_21.npy', np.array(refine_filelist))
    # np.save('/mnt/sdc/zwt_data/HumanData/combine_dataset/problem_filelist_21.npy', np.array(problem_filelist))
