import os
import numpy as np
import json
import torch
import pickle
from smplx import SMPLX
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='xxx/THuman/THuman2.0_Release')
    parser.add_argument('--output_path', type=str, default='xxx/humanscan_composite')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path

    smplx_model = SMPLX('lib/models/deformers/smplx/SMPLX', gender='neutral', \
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
                                use_pca=True,
                                num_pca_comps=12,
                                num_betas=10,
                                flat_hand_mean=False).cuda()

    file_list = os.listdir(dataset_path)
    for file_name in tqdm(file_list):
        pose_path = os.path.join(output_path, file_name, 'smplx', 'smplx_param_refine.pkl')
        cam_path = os.path.join(output_path, file_name, 'pose', '000_000.json')
        smplx_params = np.load(pose_path, allow_pickle=True)
        del smplx_params['translation']
        with open(cam_path, 'r') as file:
            center = np.array(json.load(file)['center'])
        smplx_params['transl'] = center
        for key in smplx_params.keys():
            smplx_params[key] = torch.FloatTensor(smplx_params[key]).cuda()
        smplx_params['transl'] = smplx_params['transl'].unsqueeze(0)
        output = smplx_model(**smplx_params)
        smplx_params['left_hand_pose'] = output.left_hand_pose + smplx_model.left_hand_mean.unsqueeze(0)
        smplx_params['right_hand_pose'] = output.right_hand_pose + smplx_model.right_hand_mean.unsqueeze(0)
        for key in smplx_params.keys():
            smplx_params[key] = smplx_params[key].detach().cpu().numpy()
        # os.remove(os.path.join(output_path, file_name, 'smplx', 'smplx_param.pkl'))
        with open(os.path.join(output_path, file_name, 'smplx', 'smplx_param.pkl'), 'wb') as savefile:
            pickle.dump(smplx_params, savefile)