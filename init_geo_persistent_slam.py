#!/usr/bin/env python
from flask import Flask, request, jsonify
import torch
import numpy as np
import os
from pathlib import Path
from time import time

# Set up CUDA allocation configuration if needed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import required functions and modules from your SLAM project
from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (
    save_intrinsics, save_intrinsics_slam, save_extrinsic, save_extrinsic_slam,
    save_points3D, save_points3D_slam, save_time, save_images_and_masks,
    init_filestructure, get_sorted_image_files, split_train_test, load_images,
    compute_co_vis_masks
)
from utils.camera_utils import generate_interpolated_path

app = Flask(__name__)

# -------------------- Global Config and Model Loading --------------------
MODEL_CHECKPOINT = './mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
IMAGE_SIZE = 512
SCHEDULE = 'cosine'
LR = 0.01
NITER = 300
MIN_CONF_THR = 5
LLFFHOLD = 8

# Default parameters for inference (can be overwritten via the API payload)
DEFAULT_N_VIEWS = 2
DEFAULT_CO_VIS_DSP = False
DEFAULT_DEPTH_THR = 0.01

print("Loading SLAM model...")
model = AsymmetricMASt3R.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)
print("SLAM model loaded successfully.")

# ------------------------ Flask Inference Endpoint ------------------------
@app.route('/infer', methods=['POST'])
def infer():
    """
    Expects a JSON payload with the following keys:
      - source_path: directory containing the SLAM data/images
      - model_path: directory to save the results
      - img_idx: frame index (integer)
    Optional keys:
      - n_views, co_vis_dsp, depth_thre, infer_video, conf_aware_ranking, focal_avg
    """
    data = request.get_json()

    source_path = data.get("source_path", "")
    model_path = data.get("model_path", "")
    img_idx = int(data.get("img_idx", 1))
    n_views = int(data.get("n_views", DEFAULT_N_VIEWS))
    co_vis_dsp = bool(data.get("co_vis_dsp", DEFAULT_CO_VIS_DSP))
    depth_thre = float(data.get("depth_thre", DEFAULT_DEPTH_THR))
    infer_video = bool(data.get("infer_video", False))
    conf_aware_ranking = bool(data.get("conf_aware_ranking", False))
    focal_avg = bool(data.get("focal_avg", False))
    tmp_folder = data.get("tmp_folder", "")                             # comment this line if running with run_infer_slam_fast.sh and uncomment for parallel.sh

    # Validate required parameters
    if not source_path or not model_path:
        return jsonify({"error": "Missing required parameters (source_path or model_path)."}), 400

    try:
        # ------------------ (1) Setup Files and Load Images ------------------
        save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
        print(f"source_path is: {source_path}")
        # image_dir = Path(source_path) / 'images'                 # uncomment this line if running with run_infer_slam_fast.sh
        image_dir = Path(tmp_folder)                                 # uncomment for parallel.sh
        image_files, image_suffix = get_sorted_image_files(image_dir)
        
        if infer_video:
            train_img_files = image_files
        else:
            train_img_files, test_img_files = split_train_test(image_files, LLFFHOLD, n_views, verbose=True)
        
        image_files = train_img_files

        images, org_imgs_shape = load_images(image_files, size=IMAGE_SIZE)
        start_time = time()

        # ------------------ (2) Inference Steps ------------------
        print(">> Making pairs...")
        pairs = make_pairs(images, scene_graph='oneref-0', prefilter=None, symmetrize=False)
        print(">> Running inference...")
        output = inference(pairs, model, DEVICE, batch_size=BATCH_SIZE, verbose=True)
        print(">> Global alignment...")
        scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=100, schedule=SCHEDULE, lr=LR, focal_avg=focal_avg)
        # end_time = time()
        # loss_Time = end_time - start_time
        # save_time(model_path, '[1] loss_computation_time', loss_Time)

        # ------------------ (3) Extract Scene Data ------------------
        extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
        intrinsics = to_numpy(scene.get_intrinsics())
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d = np.array(to_numpy(scene.get_pts3d()))
        depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
        values = [param.detach().cpu().numpy() for param in scene.im_conf]
        confs = np.array(values)

        if conf_aware_ranking:
            avg_conf_scores = confs.mean(axis=(1, 2))
            sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        else:
            sorted_conf_indices = np.arange(n_views)

        if depth_thre > 0:
            overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics,
                                                       extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
            overlapping_masks = ~overlapping_masks
        else:
            co_vis_dsp = False
            overlapping_masks = None

        end_time = time()
        Train_Time = end_time - start_time
        save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

        # ------------------ (4) Pose Interpolation (if not infer_video) ------------------
        if not infer_video:
            n_train = len(train_img_files)
            n_test = len(test_img_files)
            if n_train < n_test:
                n_interp = (n_test // (n_train - 1)) + 1
                all_inter_pose = []
                for i in range(n_train - 1):
                    tmp_inter_pose = generate_interpolated_path(poses=extrinsics_w2c[i:i+2], n_interp=n_interp)
                    all_inter_pose.append(tmp_inter_pose)
                all_inter_pose = np.concatenate(all_inter_pose, axis=0)
                all_inter_pose = np.concatenate([all_inter_pose, extrinsics_w2c[-1][:3, :].reshape(1, 3, 4)], axis=0)
                indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
                sampled_poses = np.array(all_inter_pose[indices]).reshape(-1, 3, 4)
                inter_pose_list = []
                for p in sampled_poses:
                    tmp_view = np.eye(4)
                    tmp_view[:3, :3] = p[:3, :3]
                    tmp_view[:3, 3] = p[:3, 3]
                    inter_pose_list.append(tmp_view)
                pose_test_init = np.stack(inter_pose_list, 0)
            else:
                indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
                pose_test_init = extrinsics_w2c[indices]
            save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
            test_focals = np.repeat(focals[0], n_test)
            save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False)

        focals = np.repeat(focals[0], n_views)
        end_time = time()
        save_time(model_path, '[1] init_geo', end_time - start_time)
        if img_idx == 1:
            save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
            save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
        else:
            img_idx_skip = img_idx + 1
            save_extrinsic_slam(sparse_0_path, img_idx_skip, extrinsics_w2c, image_files, image_suffix)
            save_intrinsics_slam(sparse_0_path, img_idx_skip, focals, org_imgs_shape, imgs.shape, save_focals=True)

        pts_num = save_points3D_slam(sparse_0_path, img_idx, imgs, pts3d,
                                      confs.reshape(pts3d.shape[0], -1), overlapping_masks,
                                      use_masks=co_vis_dsp, save_all_pts=True,
                                      save_txt_path=model_path, depth_threshold=depth_thre)
        save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)

        response = {
            "status": "success",
            "Train_Time": Train_Time,
            "points_num": pts_num,
            "img_idx": img_idx
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
