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

log_dir = f"outputs/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
os.makedirs(log_dir, exist_ok=True)
log_file = open(Path(log_dir) / "running_log.txt", "a")

input_file = "/home/sam/Documents/datasets/WHU/3-Mountain/combined.las"
las = laspy.read(input_file)
x = las.x
y = las.y
z = las.z

shp_file = "/home/sam/Documents/datasets/WHU/3-Mountain/combined_stable.shp"
satble_shp = gpd.read_file(shp_file)

center_x = np.mean(x) + 15  # Center x of deformation
center_y = np.mean(y) + 15  # Center y of deformation
radius = 25  # Radius of the mining region
max_depth = 15  # Maximum depth at the center
sigma = radius / 2.0  # Spread of the deformation
noise_std = 0.1  # Noise standard deviation in meters

angle_x_deg = 0.4  # Rotation around x-axis in degrees
angle_y_deg = 0.1  # Rotation around y-axis in degrees
angle_z_deg = 17.5  # Rotation around z-axis in degrees
translation_x = 10.0  # Translation along x-axis in meters
translation_y = -5.0  # Translation along y-axis in meters
translation_z = 0.2  # Translation along z-axis in meters

DEVICE = torch.device("cuda:{}".format(0))
PADDING = "same"
VOXEL_SIZE = 2.0
PV = 5
NV = -1
PPV = -1
NUM_WORKERS = 22
ROTATION_CHOICE = "gen"

QUANTILE_THR = 0.2
ICP_VERSION = "generalized"
MAX_ITER = 2048

target = np.column_stack((x, y, z))

avg_spacing = pipeline.estimate_avg_spacing(target)
print("Estimated average spacing:", avg_spacing)

ref_ratio = 0.01
init_voxel_size = avg_spacing / ref_ratio

# Adaptively determine the voxel size
adapted_voxel_size = pipeline.adaptive_voxel_size(target, ref_ratio, init_voxel_size, 25, 15, 1)
print("Adapted voxel size:", adapted_voxel_size)

search_radi = adapted_voxel_size * 2.0

print("Search radius:", search_radi)

repeat = 1
QUANTILE_THR_LIST = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
radius_list = [5, 10, 15, 20, 25]
max_depth_list = [3, 6, 9, 12, 15]
noise_std_list = [0.001, 0.01, 0.05, 0.1]

stable_before, mask_before = pipeline.isolate_stable(target, satble_shp)
epoch_stabel_before = py4dgeo.Epoch(stable_before)

epoch_before = py4dgeo.Epoch(target)
corepoints_pcd = o3d.geometry.PointCloud()
corepoints_pcd.points = o3d.utility.Vector3dVector(epoch_before.cloud)
corepoints_pcd = corepoints_pcd.voxel_down_sample(voxel_size=adapted_voxel_size)
corepoints = np.asarray(corepoints_pcd.points)

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(target)

avg_spacing = pipeline.estimate_avg_spacing(target)
print("Estimated average spacing:", avg_spacing)

rotation_z_list = [5, 30, 45]
translation_list = [1, 5, 10]

results_list = []

for quantile_thr in QUANTILE_THR_LIST:

    for super_index in range(len(rotation_z_list)):

        for i in range(len(radius_list)):
            print(f"Experiment {super_index} started.")
            rot = rotation_z_list[super_index]
            tas = translation_list[super_index]

            # Deformation applied
            sigma = radius_list[i] / 2.0
            gt_Vol = pipeline.get_analytical_volume(
                sigma, radius_list[i], max_depth_list[i]
            )
            log_file.write(
                f"Experiment{i}:r={radius_list[i]}, d={max_depth_list[i]}, gt_Vol={gt_Vol}\n"
            )

            z_deformed, mask = pipeline.apply_deformation(
                x, y, z, center_x, center_y, radius_list[i], max_depth_list[i], sigma
            )
            for j in range(len(noise_std_list)):
                log_file.write(f"\tNoise{j}: {noise_std_list[j]}\n")

                # Noise applied
                x_noisy, y_noisy, z_noisy = pipeline.apply_noise(
                    x, y, z_deformed, noise_std_list[j]
                )
                for k in range(repeat):
                    log_file.write(f"\t\tRepeat{k}\n")

                    try:
                        # Random transformation applied
                        T = pipeline.get_random_transformation(
                            angle_range_z=((-rot - 2.5, -rot + 2.5), (rot - 2.5, rot + 2.5)),
                            translation_range=((-tas - 0.5, -tas + 0.5), (tas - 0.5, tas + 0.5)),
                        )
                        gt_T = np.linalg.inv(T)
                        x_trans, y_trans, z_trans = pipeline.apply_transformation(
                            x_noisy, y_noisy, z_noisy, T
                        )
                        source = np.column_stack((x_trans, y_trans, z_trans))
                        log_file.write(f"\t\t\tTransformation: {T}\n")

                        # # Altered point cloud saved
                        # new_header = laspy.LasHeader(
                        #     point_format=las.header.point_format, version=las.header.version
                        # )
                        # new_header.scales = las.header.scales
                        # new_header.offsets = las.header.offsets

                        # new_las = laspy.LasData(new_header)
                        # new_las.x = x_trans
                        # new_las.y = y_trans
                        # new_las.z = z_trans

                        # new_las.write(
                        #     f"{log_dir}/{super_index}_radius{i}noise{noise_std_list[j]}_{k}.las"
                        # )

                        start_timestamp = datetime.datetime.now()
                        log_file.write(f"\t\t\tStart Time: {start_timestamp}\n")
                        # EGS
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
                        log_file.write(f"\t\t\tEGS_T: {EGS_T}\n")
                        log_file.write(f"\t\t\tRRE: {rre}\n")
                        log_file.write(f"\t\t\tRTE: {rte}\n")

                        # GICP
                        GICP_T = pipeline.auto_GICP_TEST(
                            source=source,
                            target=target,
                            T_init=EGS_T,
                            thr=quantile_thr,
                            neigh=neigh,
                            max_iter=2048,
                        )
                        rre_refined = pipeline.compute_rre(GICP_T, gt_T)
                        rte_refined = pipeline.compute_rte(GICP_T, gt_T)
                        log_file.write(f"\t\t\tGICP_T: {GICP_T}\n")
                        log_file.write(f"\t\t\tRRE_refined: {rre_refined}\n")
                        log_file.write(f"\t\t\tRTE_refined: {rte_refined}\n")

                        # Estimate transformation applied
                        x_refined, y_refined, z_refined = pipeline.apply_transformation(
                            x_trans, y_trans, z_trans, GICP_T
                        )
                        refined = np.column_stack((x_refined, y_refined, z_refined))

                        # Calculate Point Cloud Registration Uncertainty
                        stable_after, mask_after = pipeline.isolate_stable(
                            refined, satble_shp
                        )
                        epoch_stabel_after = py4dgeo.Epoch(stable_after)
                        m3c2 = py4dgeo.M3C2(
                            epochs=(epoch_stabel_before, epoch_stabel_after),
                            corepoints=epoch_stabel_before.cloud[::],
                            normal_radii=(search_radi,),
                            cyl_radius=(adapted_voxel_size),
                            max_distance=(15.0),
                            registration_error=(0.0),
                        )
                        m3c2_distances_stableparts, uncertainties_stableparts = m3c2.run()
                        reg_target_source = np.nanstd(m3c2_distances_stableparts)
                        log_file.write(
                            f"\t\t\tRegistration Uncertainty Est: {reg_target_source}\n"
                        )

                        # Calculate Point Cloud Changes
                        epoch_after = py4dgeo.Epoch(refined)
                        m3c2 = py4dgeo.M3C2(
                            epochs=(epoch_before, epoch_after),
                            corepoints=corepoints,
                            normal_radii=(search_radi,),
                            cyl_radius=(adapted_voxel_size),
                            max_distance=(15.0),
                            registration_error=(reg_target_source),
                        )
                        m3c2_distances, uncertainties = m3c2.run()
                        change_sign = np.where(
                            abs(m3c2_distances) > uncertainties["lodetection"], True, False
                        )

                        # Segment changes
                        hulls = pipeline.segment_changes(corepoints, change_sign)

                        selected_indices = [np.argmax([hull.area for hull in hulls])]
                        selected_hulls = [hulls[i] for i in selected_indices]
                        inside_mask_refined = pipeline.is_inside_selected_hulls_vectorized(
                            selected_hulls, refined[:, :2]
                        )
                        inside_mask_raw = pipeline.is_inside_selected_hulls_vectorized(
                            selected_hulls, target[:, :2]
                        )

                        filtered_refined = refined[inside_mask_refined]
                        filtered_raw = target[inside_mask_raw]

                        filtered_refined_x, filtered_refined_y, filtered_refined_z = (
                            filtered_refined.T
                        )
                        filtered_raw_x, filtered_raw_y, filtered_raw_z = filtered_raw.T

                        # Generate DEMs
                        dem_before, dem_after, dem_grid_x, dem_grid_y, grid_res = (
                            pipeline.reletive_DEM(
                                filtered_raw,
                                filtered_refined,
                                grid_res=None,
                                method="linear",
                                mask_hulls=selected_hulls,
                            )
                        )

                        # Calculate volume difference
                        net_volume, cut_volume, fill_volume, diff_DEMs = (
                            pipeline.calculate_volume(
                                dem_before,
                                dem_after,
                                grid_res=grid_res,
                                threshold=reg_target_source,
                            )
                        )
                        finish_timestamp = datetime.datetime.now()
                        log_file.write(f"\t\t\tFinish Time: {finish_timestamp}\n")
                        log_file.write(
                            f"\t\t\tTime Elapsed: {finish_timestamp - start_timestamp}\n"
                        )

                        volume_cut_acc = (gt_Vol + net_volume) / gt_Vol * 100
                        log_file.write(f"\t\t\tVolume Cut Accuracy: {volume_cut_acc}\n")
                        log_file.write(f"\t\t\tNet Volume: {net_volume}\n")
                        log_file.write(f"\t\t\tCut Volume: {cut_volume}\n")
                        log_file.write(f"\t\t\tFill Volume: {fill_volume}\n")

                        # Save results
                        result_entry = {
                            "radius": radius_list[i],
                            "max_depth": max_depth_list[i],
                            "noise_std": noise_std_list[j],
                            "rre": rre,
                            "rte": rte,
                            "rre_refined": rre_refined,
                            "rte_refined": rte_refined,
                            "cut_volume": cut_volume,
                            "fill_volume": fill_volume,
                            "net_volume": net_volume,
                            "volume_cut_acc": volume_cut_acc,
                            "reg_target_source": reg_target_source,
                            "time_elapsed": finish_timestamp - start_timestamp,
                            "quantile_thr": quantile_thr,
                        }
                        results_list.append(result_entry)
                        print(result_entry["volume_cut_acc"])
                    except Exception as e:
                        log_file.write(f"\t\t\tError at i={i}, j={j}, k={k}: {e}\n")
                        print(f"Error at i={i}, j={j}, k={k}: {e}")
                        continue

        

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"{log_dir}/results_{super_index}_{rot}_{tas}.csv", index=False)
        print("Results saved to CSV file.")

log_file.close()
