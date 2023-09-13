## Directed Ray Distance Function for Scenes
The repository below contains code for papers

**Learning to Predict Scene-Level impilict functions from Posed RGBD Data**, CVPR 2023
<br>
*Nilesh Kulkarni, Linyi Jin, Justin Johnson, David F. Fouhey*

**Directed Ray Distance Functions 3D Scene Reconstruction**, ECCV 2022
*Nilesh Kulkarni, Justin Johnson, David F. Fouhey*


## Installation Instructions
```sh
conda create -n drdf python=3.7.15
conda activate drdf
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install shapely rtree graph-tool pyembree -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install trimesh[all]==3.9.36 opencv-python==3.4.8.29
pip install  imageio scipy scikit-learn numba loguru tensorboard dominate yattag visdom ray scikit-image ipdb
pip install -U "ray[default]"
pip install open3d==0.13.0
pip install loguru, visdom, numba
python setup.py develop
```


### Download the pretrained model 
```
wget https://fouheylab.eecs.umich.edu/~nileshk/mv_drdf/cachedir.tar
mv cachedir.tar rgbd_drdf/
cd rgbd_drdf && tar -xf cachedir.tar
```


### Download the pretrained model with preprocess data
```
wget https://fouheylab.eecs.umich.edu/~nileshk/mv_drdf/cachedir_preprocessed.tar
mv cachedir_preprocessed.tar rgbd_drdf/
cd rgbd_drdf && tar -xf cachedir_preprocessed.tar;  mv cachedir_preprocessed cachedir
```

## Matterport3D Dataset Setup
Download the Matterport3D dataset and organize it in a directory to look as follows:
```
Matterport3d
├── scans
    ├── undistorted_color_images
       ├── 17DRP5sb8fy/undistorted_color_images
            ├── *.jpg
        ├── ... <other house ids>
        ├── ...
    ├──undistorted_depth_images
        ├──17DRP5sb8fy/undistorted_depth_images
            ├── *.png
        ├── ... <other house ids>
        ├── ...
   ├── undistorted_camera_parameters
        ├──17DRP5sb8fy/undistorted_camera_parameters
            |- *.conf
        ├── ... <other house ids>
        ├── ...
   ├── poisson_meshes
        ├──17DRP5sb8fy/poisson_meshes
            |- *.ply
        ├── ... <other house ids>
        ├── ...
```

## Inference
The code structured as a package rgbd_drdf, and allows to use the pretrained model on new unseen images. Inference only supported on Matterport Images for now.
```
d2drdf
|   README.md
|   setup.py
|   install.sh
├── data_dir
|    ├──Matteport3D
|
├── rgbd_drdf
|   ├──cachedir
|       ├── checkpoints
|       ├──pcl_render_dir
|       ├── eval
├──rgbd_drdf_preprocess
├──data_dir

```
### Run pretrained model
#### Download cachedir and pretrained checkpoint 
```

```

#### Run inference commands.
This will generate outputs for 1200 image.  Tip, please set `NUM_ITER` to run a smaller subset.
```sh
python run_scripts/test_drdf.py \
    --set NAME cvpr_mesh_drdf_clip_tanh_nbrs_20_2080 \ 
    DATALOADER.SPLIT val  TEST.NUM_ITER 1200  \
    TRAIN.NUM_EPOCHS 199 True DATALOADER.NO_CROP True \
    DATALOADER.INFERENCE_ONLY True TEST.HIGH_RES False RAY.NUM_WORKERS
```

### Visualize outputs as GIF
Visualizations are created under `rgbd_drdf/cachedir/pcl_render_dir_gif`. Needs `ffmpeg` to run.
```sh
python  rgbd_drdf/viz_scripts/render_viz_pcl_logs.py --set DIR_NAME cvpr_mesh_drdf_clip_tanh_nbrs_20_2080  SPLIT val HIGH_RES False DATASET matterport GIF_OUT False TEST_EPOCH_NUMBER 199
```

## Run Evaluations
Only run after running the pre-trained model on all the examples (1200 of them)
<details>
<summary> Create GT data </summary>

```sh
sh python run_scripts/gt_eval.py  --cfg rgbd_drdf/config/eval_mp3d/mp3d_gt_eval_data.yaml 
```
</details>
<details>
<summary> Scene PR Evaluations </summary>

```sh
python rgbd_drdf/benchmark/evaluate_scene_pr.py  --cfg rgbd_drdf/config/eval_configs/matterport/drdf_1pt0.yaml --set  TEST_EPOCH_NUMBER 199 EVAL_SPLIT val
```
</details>

<details>
<summary> Ray PR Evaluations </summary>

```
python rgbd_drdf/benchmark/evaluate_ray_pr.py  --cfg rgbd_drdf/config/eval_configs/matterport/drdf_1pt0.yaml --set  TEST_EPOCH_NUMBER 199 EVAL_SPLIT val
```
</details>



## Preprocessing Matterport3D data
Skip this step if you want to use the already preprocess Matterport3D in the cachedir. This step creates the `rgbd_drdf/cachedir/mp3d_data_zmx8` directory.  Running this step takes times. If you have access to multiple nodes parallelize this by running for different house_ids indpedently
```sh
## Runs it for the first 10 houses.
python rgbd_preprocess/preprocess_frustum/preprocess_matterport.py  --start_index 0 --end_index 10 
```
Change `start_index` increments of 10 until 90

## Training
Ideal configuration of the machine is 1 GPU and 16 CPUS. The computation of ray distance function in the dataloader is expensive hence to speed prefer to use a more CPUs during training.
```sh
python train_scripts/train_drdf.py --cfg rgbd_drdf/config/train/base_train_drdf.yaml  --set NAME matterport_drdf_model  DATALOADER.DATASET_TYPE matterport TRAIN.NUM_EPOCHS 200 MODEL.ALLOW_CLIPPING  True MODEL.CLIP_ACTIVATION tanh TRAIN.NUM_WORKERS 16
```


## Further Release Details (Time line)
>[Sept'23] We will be releasing the code RGBD DRDF in sometime; the DRDF model trained with full dataset using mesh supervision has similar performance compared to training with only RGBD data. Expected code release end of Sept'23.
>[Oct'23]



### Citation
If you find this work or code useful for your research please consider citing the following two papers
```
@inproceedings{kulkarni2023learning,
  title={Learning to Predict Scene-Level Implicit 3D from Posed RGBD Data},
  author={Kulkarni, Nilesh and Jin, Linyi and Johnson, Justin and Fouhey, David F},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
```
@inproceedings{kulkarni2022directed,
  title={Directed Ray Distance Functions for 3D Scene Reconstruction},
  author={Kulkarni, Nilesh and Johnson, Justin and Fouhey, David F},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```

