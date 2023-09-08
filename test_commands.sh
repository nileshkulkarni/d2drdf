python run_scripts/test_drdf.py --set NAME cvpr_mesh_drdf_clip_tanh_nbrs_20_2080  DATALOADER.SPLIT val  TEST.NUM_ITER 1200  TEST.LATEST_CKPT True DATALOADER.NO_CROP True DATALOADER.INFERENCE_ONLY True TEST.HIGH_RES False RAY.NUM_WORKERS 2

python  rgbd_drdf/viz_scripts/render_viz_ray_logs.py --set DIR_NAME cvpr_mesh_drdf_clip_tanh_nbrs_20_2080  SPLIT val

