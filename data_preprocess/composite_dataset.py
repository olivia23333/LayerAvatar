import shutil
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='HumanData/CustomHumans')
    parser.add_argument('--output_path', type=str, default='humanscan_composite')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    dataset_path = args.dataset_path
    items = sorted(os.listdir(dataset_path))

    if 'CustomHumans' in dataset_path:
        for item in tqdm(items):
            item_folder = os.path.join(dataset_path, item, '18views_3_wbg', 'image')
            item_folder_seg = os.path.join(dataset_path, item, '18views_3_wbg', 'seg')
            item_folder_pose = os.path.join(dataset_path, item, '18views_3_wbg', 'calib')
            if not os.path.exists(os.path.join(output_path, item)):
                os.mkdir(os.path.join(output_path, item))
                os.mkdir(os.path.join(output_path, item, 'smplx'))
            shutil.copy(os.path.join(os.path.dirname(dataset_path), 'smplx', item, 'mesh-f'+item[-5:]+'_refine.json'), os.path.join(output_path, item, 'smplx'))
            shutil.copytree(item_folder, os.path.join(output_path, item, 'rgb'))
            shutil.copytree(item_folder_seg, os.path.join(output_path, item, 'seg'))
            shutil.copytree(item_folder_pose, os.path.join(output_path, item, 'pose'))

    elif 'THuman2_1' in dataset_path:
        for item in tqdm(items[526:]):
            item_folder = os.path.join(dataset_path, item, '18views_3_wbg', 'image')
            item_folder_seg = os.path.join(dataset_path, item, '18views_3_wbg', 'seg')
            item_folder_pose = os.path.join(dataset_path, item, '18views_3_wbg', 'calib')
            if not os.path.exists(os.path.join(output_path, item)):
                os.mkdir(os.path.join(output_path, item))
                os.mkdir(os.path.join(output_path, item, 'smplx'))
            shutil.copy(os.path.join(os.path.dirname(dataset_path), 'smplx', item, 'smplx_param.pkl'), os.path.join(output_path, item, 'smplx'))
            shutil.copytree(item_folder, os.path.join(output_path, item, 'rgb'), dirs_exist_ok=True)
            shutil.copytree(item_folder_seg, os.path.join(output_path, item, 'seg'), dirs_exist_ok=True)
            shutil.copytree(item_folder_pose, os.path.join(output_path, item, 'pose'), dirs_exist_ok=True)

    elif 'THuman' in dataset_path:
        for item in tqdm(items):
            item_folder = os.path.join(dataset_path, item, '18views_3_wbg', 'image')
            item_folder_seg = os.path.join(dataset_path, item, '18views_3_wbg', 'seg')
            item_folder_pose = os.path.join(dataset_path, item, '18views_3_wbg', 'calib')
            if not os.path.exists(os.path.join(output_path, item)):
                os.mkdir(os.path.join(output_path, item))
                os.mkdir(os.path.join(output_path, item, 'smplx'))
            shutil.copy(os.path.join(os.path.dirname(dataset_path), 'THUman20_Smpl-X', item, 'smplx_param_refine.pkl'), os.path.join(output_path, item, 'smplx'))
            shutil.copytree(item_folder, os.path.join(output_path, item, 'rgb'))
            shutil.copytree(item_folder_seg, os.path.join(output_path, item, 'seg'))
            shutil.copytree(item_folder_pose, os.path.join(output_path, item, 'pose'))

    else:
        raise ValueError(f'Dataset path {dataset_path} is not supported')