# LayerAvatar: Disentangled Clothed Avatar Generation with Layered Representation
## [Arxiv](https://arxiv.org/abs/2501.04631) | [Project Page](https://olivia23333.github.io/LayerAvatar/) | ICCV Highlight

<img src="assets/teaser.pdf" /> 

Official PyTorch implementation of ICCV paper: [*Disentangled Clothed Avatar Generation with Layered Representation*](https://arxiv.org/abs/2501.04631).

## Getting Started

### Prerequisites

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 18.04 LTS)
- Python 3.8
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11.7
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.13.0
- [MMCV](https://github.com/open-mmlab/mmcv) 1.7.0
- [MMGeneration](https://github.com/open-mmlab/mmgeneration) 0.7.2

### Installation
1. Set up a conda environment as follows:
```bash
# Export the PATH of CUDA toolkit(other 11.x version can also be used)
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n layeravatar python=3.8
conda activate layeravatar

# Install PyTorch (this script is for CUDA 11.7)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install MMCV and MMGeneration
wget --no-check-certificate  https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl
pip install mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl
git clone https://github.com/open-mmlab/mmgeneration && cd mmgeneration && git checkout v0.7.3
pip install -v -e .
cd ..

# Clone this repo and install other dependencies
git clone <this repo> && cd <repo folder>
pip install -r requirements.txt

# Install simple-knn
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/submodules/simple-knn/
python setup.py develop
cd ../../../

# Install diff-gaussian-rasterization
git clone https://github.com/ashawkey/diff-gaussian-rasterization
cd diff-gaussian-rasterization
python setup.py develop
cd ..

# Install dependencies for deformation module
python setup.py develop

# Install pytorch3d
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.2/download/linux-64/pytorch3d-0.7.2-py38_cu117_pyt1130.tar.bz2
conda install --use-local pytorch3d-0.7.2-py38_cu117_pyt1130.tar.bz2
```

2. Download the SMPLX model and related files for avatar representation template and gaussian initialization.

**(Recommend) You can run the following command to automatically download main files.**

*Before running, please remember to register on the [SMPL-X website](https://smpl-x.is.tue.mpg.de) and [FLAME website](http://flame.is.tue.mpg.de/).*
```bash
bash scripts/fetch_template.sh 
```
Then download components' templates from [Google Drive](https://drive.google.com/file/d/1o-D_Dc2CBI1VqhOFXd--U8kLcXOH1Oc1/view?usp=sharing) and put them under `work_dirs/cache/template` folder.

After downloading, the structure should look like this:
```
.
├── assets
├── ...
├── lib
│   ├── models
│       ├── deformers
│           ├── smplx
│               ├── SMPLX
│                   ├── models
│                       ├── smplx
│                           ├── SMPLX_FEMALE.npz
│                           ├── SMPLX_FEMALE.pkl
│                           ├── SMPLX_MALE.npz
│                           ├── SMPLX_MALE.pkl
│                           ├── SMPLX_NEUTRAL.npz
│                           ├── SMPLX_NEUTRAL.pkl
│                           ├── smplx_npz.zip
│                           └── version.txt
└── work_dirs
    ├── cache
        ├── template
            ├── FLAME_masks.pkl
            ├── body.obj
            ├── hair_shoes.obj
            ├── up_and_low.obj
            ├── SMPL-X__FLAME_vertex_ids.npy
            └── smplx_vert_segmentation.json
```

(You can also download them manually and place them in the correct folders.

Put the following files in the `work_dirs/cache/template` folder.
- [SMPL-X segmentation file](https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json)(smplx_vert_segmentation.json)
- [SMPL-X FLAME Correspondence](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip)(SMPL-X__FLAME_vertex_ids.npy)
- [FLAME Mask](https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip)(FLAME_masks.pkl)
- [Components Templates](https://drive.google.com/file/d/1o-D_Dc2CBI1VqhOFXd--U8kLcXOH1Oc1/view?usp=sharing)(components.tar.gz)

Put the [SMPL-X model](https://smpl-x.is.tue.mpg.de) (models_smplx_v1_1.zip) in `lib/models/deformers/smplx/SMPLX/`)

3. Extract avatar representation template from downloaded files:
```bash
cd lib/models/deformers

# save subdivide component mesh and corresponding uv
python subdivide_multi_layer_smplx.py

# save parameters for init
python utils_smplx_layer.py

# combine init parameters from different layer into single file
python combine_template.py
```

4. (**Optional**, for training and local editing process)Download the [Pretrained VGG](https://drive.google.com/file/d/1x1Kc41DqolkNve70I_OCsiyamrPuNB1b/view?usp=sharing) for perceptual loss calculation, and put the files to `work_dirs/cache/vgg16.pt`.

## Inference
Downloaded trained model from following link and put it to corresponding local path for testing.
| Model | Training Data | Download Link | Local Path | Config |
| :--- | :--- | :--- | :--- | :--- |
| Layeravatar-THUman | THUman2.0 dataset | [Link](https://drive.google.com/file/d/12G_Yp-GRitMF31fKYuIxkOynnNPuZUC6/view?usp=sharing) | work_dirs/layeravatar_uncond_thuman/ckpt | 
| Layeravatar-Composite | THUman2.0, THUman2.1, CustomHuman | [Link](https://drive.google.com/file/d/12QrShr4UjNLnV1poKI-Z3JGAkmsNmIt0/view?usp=sharing) | work_dirs/layeravatar_uncond_composite/ckpt |

Example `.pth` files can be download from [here](https://drive.google.com/file/d/1EXBueuArrSDpZ95lGd8TOztCYZ_cUSxA/view?usp=share_link) for `transfer` mode.

If you are using RTX 3090 for testing, please set `samples_per_gpu=4` and `workers_per_gpu=2` in config file to 1 to avoid CUDA OUT OF MEMORY problem.

```bash
# For novel view synthesis (We provide 36 views)
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --gpu-ids 0 1 --mode 'viz'

# For animation (We use the motion file from the X-Avatar repo for demo. The code also support AMASS motion dataset, you can download the CMU data from the AMASS dataset and put it in ./demo/ani_exp/ani)
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --gpu-ids 0 1 --mode 'animate' --pose_path demo/ani/xxx

# For attribute transfer(upper cloth, pants, hair and shoes), use composte pre-train model
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --gpu-ids 0 1 --mode 'transfer'

# For evaluation, to run this you need to finish the data preparation process below
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --gpu-ids 0 1 --mode 'eval'
```

## Training
### Data preparation

1. Download [THUman2.0 Dataset](https://github.com/ytrock/THuman2.0-Dataset) and its corresponding SMPL-X fitting parameters from [here](https://drive.google.com/file/d/1yM_xN-W-UfSvzmAue58O3cw93oWmXtHU/view?usp=sharing).
Unzip them to `./data/THuman`.
(**optional**, if you want to expand the dataset, we also provide processing code for CustomHuman Dataset and THUman2.1 Dataset)
[Custom Human](https://custom-humans.github.io), SMPL-X fitting parametes[here](https://drive.google.com/file/d/15590YEC9sh6Dgq_X1uwo4HAaY-cAQ9Y9/view?usp=sharing).
[THUman2.1](https://github.com/ytrock/THuman2.0-Dataset), SMPL-X fitting parameters[here](https://drive.google.com/file/d/194s_b2y3wpp699XVc4KNaqD6Bb7T14JS/view?usp=sharing)

We adapted The SMPL-X fitting parameters of THUman2.0 and CustomHuman to conform to the version used by THUman2.1.

2. Render the RGB image with [ICON](https://github.com/YuliangXiu/ICON).

We made some modifications to the ICON rendering part, so please install our version:
```bash
git clone https://github.com/olivia23333/ICON

cd ICON
git checkout e3gen
conda create -n icon python=3.8
conda activate icon
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler -c conda-forge nvidiacub pyembree
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt --use-deprecated=legacy-resolver
git clone https://github.com/YuliangXiu/rembg
cd rembg 
pip install -e .
cd ..

bash fetch_data.sh
```

After the installation, run
```bash
# rendering 54 views for each scan
bash scripts/render_thuman.sh
# for Custom Human and THUman2.1 dataset
bash scripts/render_custom.sh
bash scripts/render_thuman2_1.sh
```

If `scripts/render_thuman.sh` is stuck at the `mesh.ray.intersects_any` function, you can refer to [this issue](https://github.com/YuliangXiu/ICON/issues/62).

3. Obtain segmentation masks using [Sapiens](https://github.com/facebookresearch/sapiens), we recommend to install sapiens-lite according to its `README.md` in a new conda environment.
```bash
# After installing sapiens-lite, run following code to obtain segmentation masks for each dataset.
python data_preprocess/run_seg_thuman.py
python data_preprocess/run_seg_custom.py
python data_preprocess/run_seg_thuman_21.py

# combine sapiens segmentation results to five labels(body, top, bottom, hair, shoes)
# by changing the root_path you can deal with different dataset, root_path is the folder containing all samples
python data_preprocess/transfer_seg.py --root_path xxx/THuman/THuman2.0_Release

# save preprocessed segmentation data for training speed up
python data_preprocess/save_seg_pre.py --root_path xxx/THuman/THuman2.0_Release
```

4. Finally, run the following commands:
```bash
# change rendered images into training dataset format. You can change the output path
# To train on multiple dataset just change the dataset_path and remain the same output_path
python data_preprocess/composite_dataset.py --dataset_path xxx/THuman/THuman2.0_Release --output_path xxx/humanscan_composite

# transfer all the smplx parameters to a unified version
python data_preprocess/smplx_transfer/change_thuman_smplx_format.py --dataset_path xxx/THuman/THuman2.0_Release --output_path xxx/humanscan_composite
python data_preprocess/smplx_transfer/change_custom_smplx_format.py --dataset_path xxx/CustomHumans/mesh --output_path xxx/humanscan_composite
python data_preprocess/smplx_transfer/change_thuman21_smplx_format.py --dataset_path xxx/THuman2_1/model --output_path xxx/humanscan_composite

# obtain training set, we also provide dataset_thuman.txt
python data_preprocess/split.py --txt_path data_preprocess/dataset_composite.txt --root_path xxx/humanscan_composite

# generate test cache
# if you want to use composite dataset for training, use configs/layeravatar_uncond_16bit_composite.py here
# else use configs/layeravatar_uncond_16bit_thuman.py
conda activate layeravatar
CUDA_VISIBLE_DEVICES=0 python tools/inception_stat.py /PATH/TO/CONFIG
```

The final structure of the training dataset is as follows:
```
data
└── humanscan_composite
    ├── human_train
        ├── 0000
            ├── pose    # camera parameter
            ├── rgb     # rendered images
            ├── seg     # preprocessed segmentation results to speed up training
            ├── smplx   # smplx parameter
        ├── ...
    ├── human_novel
    └── human_train_cache.pkl
```

### Training scripts

Run the following command to train a model:
To avoid CUDA OUT OF MEMORY problem, uncomment line 977, 978 and comment 979, 980 in `LayerAvatar/lib/models/autodecoders/base_nerf.py`

```bash
# For /PATH/TO/CONFIG, we use configs/ssdnerf_avatar_uncond_thuman_conv_16bit.py here
python train.py /PATH/TO/CONFIG --gpu-ids 0 1
```
Our model is trained using 2 RTX 3090 (24G) GPUs.

Model checkpoints will be saved into `./work_dirs`. UV features plane for scans will be saved into `./cache`.

## Acknowledgements
This project is built upon many amazing works: 
- [SSDNeRF](https://github.com/Lakonik/SSDNeRF) and [E3Gen](https://github.com/olivia23333/E3Gen)for Base Diffusion Backbone
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [AG3D](https://github.com/zj-dong/AG3D) for deformation module
- [DELTA](https://github.com/yfeng95/DELTA) for loss design
- [ICON](https://github.com/YuliangXiu/ICON), [NHA](https://github.com/philgras/neural-head-avatars), [MVP](https://github.com/facebookresearch/mvp), [TADA](https://github.com/TingtingLiao/TADA),
[DECA](https://github.com/yfeng95/DECA) and [PointAvatar](https://github.com/zhengyuf/PointAvatar) for data preprocessing
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) for perceptual loss

## Citation
```
@article{zhang2025disentangled,
  title={Disentangled clothed avatar generation with layered representation},
  author={Zhang, Weitian and Wu, Sijing and Liao, Manwen and Yan, Yichao},
  journal={arXiv preprint arXiv:2501.04631},
  year={2025}
}
```
