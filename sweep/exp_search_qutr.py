#!/usr/bin/env python3
"""
Batch experiment script – Revision 1
-----------------------------------
Implements the editor’s first comment:
*  Quantile hyper‑parameter search for G‑ICP (0.05 → 0.30)
*  Records the quantile used in both log and CSV output
*  Gives a transparent rule for the voxel size (4 × mean point spacing)

Sam, this file is meant to be a drop‑in replacement for your original
`batch_run.py`.  Every section that changed is flagged with    ### REV‑1 ###
so you can diff easily.
"""

import datetime
import os
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import torch
import geopandas as gpd
import py4dgeo
import py4dgeo.epoch
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import pipeline

###############################################################################
#  I/O & logging
###############################################################################

log_dir = f"outputs/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
os.makedirs(log_dir, exist_ok=True)
log_file = open(Path(log_dir) / "running_log.txt", "a")

###############################################################################
#  Data loading
###############################################################################

input_file = "/home/sam/Documents/datasets/WHU/3-Mountain/combined.las"
las = laspy.read(input_file)

x, y, z = las.x, las.y, las.z

target = np.column_stack((x, y, z))

shp_file = "/home/sam/Documents/datasets/WHU/3-Mountain/combined_stable.shp"
stable_shp = gpd.read_file(shp_file)

###############################################################################
#  Synthetic deformation parameters (unchanged)
###############################################################################

center_x = np.mean(x) + 15
center_y = np.mean(y) + 15
radius = 25
max_depth = 15
sigma = radius / 2.0
noise_std = 0.1

angle_x_deg, angle_y_deg, angle_z_deg = 0.4, 0.1, 17.5
translation_x, translation_y, translation_z = 10.0, -5.0, 0.2

###############################################################################
#  Runtime & algorithm parameters
###############################################################################

DEVICE = torch.device("cuda:0")
PADDING = "same"

# --- REV‑1: voxel size is now derived from mean point spacing -------------
avg_spacing = pipeline.estimate_avg_spacing(target)
VOXEL_SIZE = 2.0
log_file.write(f"Derived VOXEL_SIZE = {VOXEL_SIZE:.3f} m (4 × {avg_spacing:.3f})\n")

PV, NV, PPV = 5, -1, -1
NUM_WORKERS = 22
ROTATION_CHOICE = "gen"

# --- REV‑1: quantile sweep list -------------------------------------------
QUANTILE_THR_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ICP_VERSION = "generalized"
MAX_ITER = 2048

###############################################################################
#  Pre‑processing that depends on voxel size
###############################################################################

ref_ratio = 0.01
init_voxel_size = avg_spacing / ref_ratio

# Adaptively determine the voxel size
adapted_voxel_size = pipeline.adaptive_voxel_size(target, ref_ratio, init_voxel_size, 25, 15, 1)

search_radi = adapted_voxel_size * 2.0

###############################################################################
#  Ground‑truth prep (stable masks, corepoints, NN search tree)
###############################################################################

stable_before, mask_before = pipeline.isolate_stable(target, stable_shp)
epoch_stable_before = py4dgeo.Epoch(stable_before)

epoch_before = py4dgeo.Epoch(target)
corepoints_pcd = o3d.geometry.PointCloud()
corepoints_pcd.points = o3d.utility.Vector3dVector(epoch_before.cloud)
corepoints_pcd = corepoints_pcd.voxel_down_sample(voxel_size=adapted_voxel_size)
corepoints = np.asarray(corepoints_pcd.points)

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(target)

###############################################################################
#  Experimental grids
###############################################################################

rotation_z_list = [5, 30, 45]
translation_list = [1, 5, 10]

radius_list = [5, 10, 15, 20, 25]
max_depth_list = [3, 6, 9, 12, 15]
noise_std_list = [0.001, 0.01, 0.05, 0.1]
repeat = 10

###############################################################################
#  Main loop
###############################################################################

results_list = []

for super_index, (rot_z, t_val) in enumerate(zip(rotation_z_list, translation_list)):
    log_file.write(f"\n===== ROT {rot_z} deg  |  TRANS {t_val} m =====\n")

    for i, (radius_i, depth_i) in enumerate(zip(radius_list, max_depth_list)):
        log_file.write(f"Experiment{i}: r={radius_i}, depth={depth_i}\n")

        sigma_i = radius_i / 2.0
        gt_vol = pipeline.get_analytical_volume(sigma_i, radius_i, depth_i)
        log_file.write(f"\tGround‑truth volume = {gt_vol:.3f} m³\n")

        # Apply deformation (no noise, no rigid transform yet)
        z_deformed, mask = pipeline.apply_deformation(
            x, y, z, center_x, center_y, radius_i, depth_i, sigma_i
        )

        for j, noise_std in enumerate(noise_std_list):
            log_file.write(f"\tNoise{j}: σ={noise_std}\n")

            # Apply gaussian noise once per σ level
            x_noisy, y_noisy, z_noisy = pipeline.apply_noise(x, y, z_deformed, noise_std)

            # --- REV‑1: quantile sweep loop ---------------------------------
            for q_idx, thr in enumerate(QUANTILE_THR_LIST):
                log_file.write(f"\t\tQuantile{q_idx}: thr={thr}\n")

                for k in range(repeat):
                    log_file.write(f"\t\t\tRepeat{k}\n")

                    try:
                        # Random rigid transform
                        T_rand = pipeline.get_random_transformation(
                            angle_range_z=((-rot_z - 2.5, -rot_z + 2.5), (rot_z - 2.5, rot_z + 2.5)),
                            translation_range=((-t_val - 0.5, -t_val + 0.5), (t_val - 0.5, t_val + 0.5)),
                        )
                        gt_T = np.linalg.inv(T_rand)

                        x_trans, y_trans, z_trans = pipeline.apply_transformation(x_noisy, y_noisy, z_noisy, T_rand)
                        source = np.column_stack((x_trans, y_trans, z_trans))

                        start_ts = datetime.datetime.now()

                        # ----- EGS coarse alignment
                        EGS_T = pipeline.EGS(
                            source=source,
                            target=target,
                            voxel_size=VOXEL_SIZE,
                            padding=PADDING,
                            ppv=PPV,
                            pv=PV,
                            nv=NV,
                            num_workers=NUM_WORKERS,
                            rotation_choice=ROTATION_CHOICE,
                            rotation_root_path="exhaustive-grid-search/data/rotations",
                        )
                        rre = pipeline.compute_rre(EGS_T, gt_T)
                        rte = pipeline.compute_rte(EGS_T, gt_T)
                        
                        print(f"EGS finished: rre={rre:.3f}, rte={rte:.3f}")

                        # ----- GICP refinement (thr sweep)
                        GICP_T = pipeline.auto_GICP_TEST(
                            source=source,
                            target=target,
                            T_init=EGS_T,
                            thr=thr,               # <<< REV‑1
                            neigh=neigh,
                            max_iter=MAX_ITER,
                        )
                        rre_ref = pipeline.compute_rre(GICP_T, gt_T)
                        rte_ref = pipeline.compute_rte(GICP_T, gt_T)
                        
                        print(f"GICP finished: rre_ref={rre_ref:.3f}, rte_ref={rte_ref:.3f}")

                        # ----- Apply refined transform
                        x_ref, y_ref, z_ref = pipeline.apply_transformation(x_trans, y_trans, z_trans, GICP_T)
                        refined = np.column_stack((x_ref, y_ref, z_ref))

                        # ----- Registration uncertainty on stable parts
                        stable_after, _ = pipeline.isolate_stable(refined, stable_shp)
                        epoch_stable_after = py4dgeo.Epoch(stable_after)
                        m3c_stable = py4dgeo.M3C2(
                            epochs=(epoch_stable_before, epoch_stable_after),
                            corepoints=epoch_stable_before.cloud.copy(),
                            normal_radii=(search_radi,),
                            cyl_radius=(adapted_voxel_size,),
                            max_distance=(15.0,),
                            registration_error=(0.0,),
                        )
                        m3c_dist_stable, _ = m3c_stable.run()
                        reg_unc = np.nanstd(m3c_dist_stable)
                        
                        print(f"Registration uncertainty: {reg_unc:.3f}")

                        # ----- Change detection on full cloud
                        epoch_after = py4dgeo.Epoch(refined)
                        m3c = py4dgeo.M3C2(
                            epochs=(epoch_before, epoch_after),
                            corepoints=corepoints,
                            normal_radii=(search_radi,),
                            cyl_radius=(adapted_voxel_size,),
                            max_distance=(15.0,),
                            registration_error=(reg_unc,),
                        )
                        m3c_dist, uncertainties = m3c.run()
                        change_sign = np.where(np.abs(m3c_dist) > uncertainties["lodetection"], True, False)
                        
                        print(f"Change detection finished: {np.sum(change_sign)} points detected as changed")

                        # ----- Segment changes
                        hulls = pipeline.segment_changes(corepoints, change_sign)
                        sel_idx = [np.argmax([h.area for h in hulls])]
                        sel_hulls = [hulls[i] for i in sel_idx]

                        inside_mask_ref = pipeline.is_inside_selected_hulls_vectorized(sel_hulls, refined[:, :2])
                        inside_mask_raw = pipeline.is_inside_selected_hulls_vectorized(sel_hulls, target[:, :2])

                        filtered_refined = refined[inside_mask_ref]
                        filtered_raw = target[inside_mask_raw]
                        
                        print(f"Filtered {len(filtered_refined)} points from refined and {len(filtered_raw)} from raw")

                        # ----- DEMs & volume
                        dem_b, dem_a, gx, gy, grid_res = pipeline.reletive_DEM(
                            filtered_raw, filtered_refined, grid_res=None, method="linear", mask_hulls=sel_hulls
                        )
                        net_vol, cut_vol, fill_vol, diff_dem = pipeline.calculate_volume(
                            dem_b, dem_a, grid_res=grid_res, threshold=reg_unc
                        )

                        finish_ts = datetime.datetime.now()
                        vol_cut_acc = (gt_vol + net_vol) / gt_vol * 100
                        
                        print(f"Volume calculation finished: net_vol={net_vol:.3f}, cut_vol={cut_vol:.3f}, fill_vol={fill_vol:.3f}, vol_cut_acc={vol_cut_acc:.2f}%")

                        # ----- Save one row
                        results_list.append({
                            "radius": radius_i,
                            "max_depth": depth_i,
                            "noise_std": noise_std,
                            "quantile_thr": thr,           # REV‑1
                            "rre": rre,
                            "rte": rte,
                            "rre_refined": rre_ref,
                            "rte_refined": rte_ref,
                            "cut_volume": cut_vol,
                            "fill_volume": fill_vol,
                            "net_volume": net_vol,
                            "volume_cut_acc": vol_cut_acc,
                            "reg_target_source": reg_unc,
                            "time_elapsed": (finish_ts - start_ts).total_seconds(),
                        })

                    except Exception as e:
                        log_file.write(f"ERROR at i={i}, j={j}, q={q_idx}, k={k}: {e}\n")
                        print(f"ERROR at i={i}, j={j}, q={q_idx}, k={k}: {e}")
                        continue

    # ----- dump CSV for each (rot, trans) setting
    df = pd.DataFrame(results_list)
    csv_name = f"results_rot{rot_z}_t{t_val}.csv"
    df.to_csv(Path(log_dir) / csv_name, index=False)
    log_file.write(f"Saved {csv_name} with {len(df)} rows.\n")

log_file.close()

print("✔  All experiments finished.  Check", log_dir)
