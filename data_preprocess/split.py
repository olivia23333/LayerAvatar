import os
import numpy as np
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='xxx/humanscan_thuman')
    parser.add_argument('--txt_path', type=str, default='data_preprocess/dataset_composite.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.txt_path, 'r') as file:
        select_sample_list = [line.strip() for line in file]
    dataset_path = args.root_path
    sample_list = os.listdir(os.path.join(dataset_path))
    os.makedirs(os.path.join(dataset_path, 'human_train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'human_novel'), exist_ok=True)

    for item in tqdm(sample_list):
        if item in select_sample_list:
            shutil.move(os.path.join(dataset_path, item), os.path.join(dataset_path, 'human_train', item))
        else:
            shutil.move(os.path.join(dataset_path, item), os.path.join(dataset_path, 'human_novel', item))
