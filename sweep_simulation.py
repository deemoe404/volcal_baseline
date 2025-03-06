#!/usr/bin/env python3
"""
Script to sweep simulation parameters and output key results to a CSV file.
Usage example:
    python sweep_simulation.py --input_file "/path/to/combined.las" \
      --shp_file "/path/to/combined_stable.shp" \
      --radius 25 --max_depth 15 --device "cuda:0" --voxel_size 2.0 \
      --noise_start 100 --noise_step 50 --noise_count 3 \
      --quantile_start 40 --quantile_step 5 --quantile_count 3 \
      --output_csv results.csv
"""

import argparse
import csv
import datetime
import time
import os
import sys
import copy
import numpy as np
import laspy
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.vectorized import contains

# Add submodule path (adjust if needed)
current_dir = os.path.abspath(".")
submodule_dir = os.path.join(current_dir, 'exhaustive-grid-search')
sys.path.append(submodule_dir)

from fft_conv_pytorch import fft_conv
from utils.pc_utils import voxelize, unravel_index_pytorch
from utils.data_utils import preprocess_pcj_B1
from utils.rot_utils import create_T_estim_matrix, load_rotations, homo_matmul
from utils.padding import padding_options
from icp.icp_versions import ICP
import py4dgeo

def run_simulation(noise_std, quantile_thr, base_log_dir,
                   input_file, shp_file, radius, max_depth, sigma,
                   device, padding, batch_size, voxel_size, PV, NV, PPV,
                   num_workers, rotation_choice, icp_version, max_iter):
    """
    Run one simulation using the provided parameters.
    noise_std is in meters; quantile_thr is given as a fraction (e.g. 0.45 for 45%).
    Returns a dictionary with key outputs.
    """
    start_time = time.time()
    
    # Create a unique logging directory for this run
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(base_log_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    
    # -------------------------
    # Load Input Data
    # -------------------------
    las = laspy.read(input_file)
    x = las.x
    y = las.y
    z = las.z

    # -------------------------
    # Define Mining Region and Ground Truth
    # -------------------------
    center_x = np.mean(x) + 15
    center_y = np.mean(y) + 15

    # sigma is either provided or computed as radius/2.0
    if sigma is None:
        sigma = radius / 2.0

    # Compute analytical (ground truth) removed volume
    analytical_volume = 2 * np.pi * sigma**2 * max_depth * (1 - np.exp(- (radius**2) / (2 * sigma**2)))
    print("Analytical ground truth volume removed (m^3):", analytical_volume)
    
    with open(os.path.join(log_dir, "Alter.txt"), "w") as log_file:
        log_file.write(f"Center X: {center_x}\n")
        log_file.write(f"Center Y: {center_y}\n")
        log_file.write(f"Radius: {radius}\n")
        log_file.write(f"Max Depth: {max_depth}\n")
        log_file.write(f"Sigma: {sigma}\n")
        log_file.write(f"Noise Std Dev: {noise_std}\n")
        log_file.write(f"Analytical Removed: {analytical_volume}\n")
    
    # -------------------------
    # Ground Truth Transformation
    # -------------------------
    angle_x_deg, angle_y_deg, angle_z_deg = 0.4, 0.1, 17.5
    angle_x = np.deg2rad(angle_x_deg)
    angle_y = np.deg2rad(angle_y_deg)
    angle_z = np.deg2rad(angle_z_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x),  np.cos(angle_x)]])
    Ry = np.array([[ np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z),  np.cos(angle_z), 0],
                   [0, 0, 1]])
    R_gt = Rz @ Ry @ Rx
    tx, ty, tz = 10, -5, 0.2
    translation = np.array([tx, ty, tz])
    T = np.eye(4)
    T[:3, :3] = R_gt
    T[:3, 3] = translation
    T_gt = np.linalg.inv(T)
    np.savetxt(os.path.join(log_dir, "groundtruth.txt"), T_gt)
    
    # -------------------------
    # Deform Point Cloud and Add Noise
    # -------------------------
    d = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = d < radius
    deformation = max_depth * np.exp(- (d**2) / (2 * sigma**2))
    z_deformed = z.copy()
    z_deformed[mask] -= deformation[mask]
    
    x_noisy = x + np.random.normal(loc=0, scale=noise_std, size=x.shape)
    y_noisy = y + np.random.normal(loc=0, scale=noise_std, size=y.shape)
    z_noisy = z_deformed + np.random.normal(loc=0, scale=noise_std, size=z_deformed.shape)
    
    points = np.vstack((x_noisy, y_noisy, z_noisy)).T
    points_transformed = points @ R_gt.T + translation
    x_trans = points_transformed[:, 0]
    y_trans = points_transformed[:, 1]
    z_trans = points_transformed[:, 2]
    
    # Save modified LAS file
    new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    new_header.scales = las.header.scales
    new_header.offsets = las.header.offsets
    new_las = laspy.LasData(new_header)
    new_las.x = x_trans
    new_las.y = y_trans
    new_las.z = z_trans
    new_las.write(os.path.join(log_dir, "alt.las"))
    
    # -------------------------
    # Registration with EGS
    # -------------------------
    start_timestamp = datetime.datetime.now()
    print("Start timestamp:", start_timestamp)
    
    with open(os.path.join(log_dir, "EGS.txt"), "w") as log_file:
        log_file.write(f"VOXEL_SIZE: {voxel_size}\n")
    
    source = np.vstack((x_trans, y_trans, z_trans)).T
    target = np.vstack((x, y, z)).T
    pci = torch.from_numpy(source)
    pcj = torch.from_numpy(target)
    print("Centering PCI...")
    make_pci_posit_translation = torch.min(pci, axis=0)[0]
    pci = pci - make_pci_posit_translation
    print("Voxelizing PCI...")
    pci_voxel, NR_VOXELS_PCI = voxelize(pci, voxel_size, fill_positive=PV, fill_negative=NV)
    CENTRAL_VOXEL_PCI = torch.where(
        NR_VOXELS_PCI % 2 == 0,
        (NR_VOXELS_PCI / 2) - 1,
        torch.floor(NR_VOXELS_PCI / 2),
    ).int()
    central_voxel_center = CENTRAL_VOXEL_PCI * voxel_size + (0.5 * voxel_size)
    weight_to_fftconv3d = pci_voxel.type(torch.int32).to(device)[None, None, :, :, :]
    pp, pp_xyz = padding_options(padding, CENTRAL_VOXEL_PCI, NR_VOXELS_PCI)
    R_batch = load_rotations(rotation_choice=rotation_choice, rot_root_path='exhaustive-grid-search/data/rotations')
    my_data, my_dataloader = preprocess_pcj_B1(pcj, R_batch, voxel_size, pp, num_workers, PV, NV, PPV)
    
    maxes = []
    argmaxes = []
    shapes = []
    minimas = torch.empty(R_batch.shape[0], 3)
    
    print("Computing FFT Convolution...")
    with torch.no_grad():
        for ind_dataloader, (voxelized_pts_padded, mins, orig_input_shape) in enumerate(my_dataloader):
            minimas[ind_dataloader, :] = mins
            input_to_fftconv3d = voxelized_pts_padded.to(device)
            out = fft_conv(input_to_fftconv3d, weight_to_fftconv3d, bias=None)
            maxes.append(torch.max(out))
            argmaxes.append(torch.argmax(out))
            shapes.append(out.shape)
    
    m_index = torch.argmax(torch.stack(maxes))
    ind0, _, ind1, ind2, ind3 = unravel_index_pytorch(argmaxes[m_index], shapes[m_index])
    rotation_index = m_index * batch_size + ind0
    R_est = R_batch[rotation_index]
    
    t = torch.Tensor([
        -(pp_xyz[0] * voxel_size) + ((CENTRAL_VOXEL_PCI[0]) * voxel_size) + (ind1 * voxel_size) + (0.5 * voxel_size),
        -(pp_xyz[2] * voxel_size) + ((CENTRAL_VOXEL_PCI[1]) * voxel_size) + (ind2 * voxel_size) + (0.5 * voxel_size),
        -(pp_xyz[4] * voxel_size) + ((CENTRAL_VOXEL_PCI[2]) * voxel_size) + (ind3 * voxel_size) + (0.5 * voxel_size),
    ])
    center_pcj_translation = my_data.center
    make_pcj_posit_translation = minimas[rotation_index]
    estim_T_baseline = create_T_estim_matrix(center_pcj_translation, R_est, make_pcj_posit_translation,
                                              central_voxel_center, t, make_pci_posit_translation)
    print("Estimated transformation matrix:")
    print(estim_T_baseline)
    np.savetxt(os.path.join(log_dir, "estimated_EGS.txt"), estim_T_baseline.cpu().numpy())
    
    with open(os.path.join(log_dir, "GICP.txt"), "w") as log_file:
        log_file.write(f"QUANTILE_THR: {quantile_thr}\n")
    
    pci_np = copy.deepcopy(source)
    pcj_np = copy.deepcopy(target)
    pcj_np_estim = homo_matmul(pcj_np, estim_T_baseline.cpu().numpy())
    pci_o3d = torch.tensor(pci_np)  # using torch for consistency
    pcj_o3d = torch.tensor(pcj_np)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pci_np)
    dist, _ = neigh.kneighbors(pcj_np_estim)
    adaptive_thr = np.quantile(dist, quantile_thr)
    print("Adaptive threshold:", adaptive_thr)
    
    # Perform Generalized ICP registration
    import open3d as o3d  # open3d is only used for ICP registration
    pci_o3d_pcd = o3d.geometry.PointCloud()
    pci_o3d_pcd.points = o3d.utility.Vector3dVector(pci_np)
    pcj_o3d_pcd = o3d.geometry.PointCloud()
    pcj_o3d_pcd.points = o3d.utility.Vector3dVector(pcj_np)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                 relative_rmse=1e-6,
                                                                 max_iteration=max_iter)
    reg_o3d = o3d.pipelines.registration.registration_generalized_icp(
        pcj_o3d_pcd, pci_o3d_pcd, adaptive_thr, estim_T_baseline.cpu().numpy(), criteria=criteria)
    estim_T_baseline_refined = reg_o3d.transformation
    print("Refined transformation matrix:")
    print(estim_T_baseline_refined)
    np.savetxt(os.path.join(log_dir, "estimated_GICP.txt"), estim_T_baseline_refined)
    
    # -------------------------
    # Error Computation
    # -------------------------
    def compute_rre(T_est, T_gt):
        T_error = T_gt @ np.linalg.inv(T_est)
        R_error = T_error[:3, :3]
        cos_theta = (np.trace(R_error) - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        return np.degrees(theta_rad)

    def compute_rte(T_est, T_gt):
        T_error = T_gt @ np.linalg.inv(T_est)
        t_error = T_error[:3, 3]
        return np.linalg.norm(t_error)
    
    gt = T_gt
    rre = compute_rre(estim_T_baseline.cpu().numpy(), gt)
    rte = compute_rte(estim_T_baseline.cpu().numpy(), gt)
    rre_refined = compute_rre(estim_T_baseline_refined, gt)
    rte_refined = compute_rte(estim_T_baseline_refined, gt)
    print("Rotation error (deg):", rre)
    print("Translation error (m):", rte)
    print("Refined rotation error (deg):", rre_refined)
    print("Refined translation error (m):", rte_refined)
    
    with open(os.path.join(log_dir, "Result.txt"), "w") as log_file:
        log_file.write(f"Rotation error (deg): {rre}\n")
        log_file.write(f"Translation error (m): {rte}\n")
        log_file.write(f"Refined rotation error (deg): {rre_refined}\n")
        log_file.write(f"Refined translation error (m): {rte_refined}\n")
    
    # -------------------------
    # Change Detection and Volume Calculation
    # -------------------------
    # Transform target points using refined transformation
    points_all = np.vstack((x_trans, y_trans, z_trans)).T
    Rt = np.linalg.inv(estim_T_baseline_refined)
    rotation_matrix = Rt[:3, :3]
    translation_vec = Rt[:3, 3]
    points_transformed = points_all @ rotation_matrix.T + translation_vec
    x_refined = points_transformed[:, 0]
    y_refined = points_transformed[:, 1]
    z_refined = points_transformed[:, 2]
    
    # Read shapefile and mask stable parts
    import geopandas as gpd
    datasource = gpd.read_file(shp_file)
    pc_target = np.vstack((x, y, z)).T
    pc_target_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, z), crs="EPSG:4548")
    pts_in_poly_target_gdf = gpd.sjoin(pc_target_gdf, datasource, predicate="within")
    pc_stable_target = pc_target[pts_in_poly_target_gdf.index]
    
    pc_source = np.vstack((x_refined, y_refined, z_refined)).T
    pc_source_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_refined, y_refined, z_refined), crs="EPSG:4548")
    pts_in_poly_source_gdf = gpd.sjoin(pc_source_gdf, datasource, predicate="within")
    pc_stable_source = pc_source[pts_in_poly_source_gdf.index]
    print(f"Stable points: target {pc_stable_target.shape[0]}, source {pc_stable_source.shape[0]}")
    
    epoch_stable_target = py4dgeo.Epoch(pc_stable_target)
    epoch_stable_source = py4dgeo.Epoch(pc_stable_source)
    epoch_stable_source.build_kdtree()
    indices, distances = epoch_stable_source.kdtree.nearest_neighbors(epoch_stable_target.cloud, 1)
    distances = np.sqrt(distances)
    
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_stable_target, epoch_stable_source),
        corepoints=epoch_stable_target.cloud,
        normal_radii=(0.5,),
        cyl_radius=(0.5),
        max_distance=(5.0),
        registration_error=(0.0),
    )
    m3c2_distances_stableparts, uncertainties_stableparts = m3c2.run()
    reg_target_source = np.nanstd(m3c2_distances_stableparts)
    print(f"Registration error (stable parts): {reg_target_source:.3f} m")
    
    epoch_target = py4dgeo.Epoch(pc_target)
    epoch_source = py4dgeo.Epoch(pc_source)
    corepoints = epoch_target.cloud[::100]
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_target, epoch_source),
        corepoints=corepoints,
        normal_radii=(0.5,),
        cyl_radii=(0.5,),
        max_distance=(15.0),
        registration_error=(reg_target_source),
    )
    m3c2_distances, uncertainties = m3c2.run()
    change_sign = np.abs(m3c2_distances) > uncertainties["lodetection"]
    
    def is_inside_selected_hulls_vectorized(selected_hulls, pts_2d):
        union_hull = unary_union(selected_hulls)
        return contains(union_hull, pts_2d[:, 0], pts_2d[:, 1])
    
    points_refined_2d = np.vstack((x_refined, y_refined)).T
    points_raw_2d = np.vstack((x, y)).T
    sig_corepoints = corepoints[change_sign][:, :2]
    db = DBSCAN(eps=3.5, min_samples=10).fit(sig_corepoints)
    labels = db.labels_
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    hulls = []
    for lbl in unique_labels:
        cluster_points = sig_corepoints[labels == lbl]
        if len(cluster_points) >= 3:
            hull = MultiPoint(cluster_points).convex_hull
            hulls.append(hull)
    selected_hulls = [hulls[np.argmax([hull.area for hull in hulls])] ] if hulls else []
    
    if selected_hulls:
        inside_mask_refined = is_inside_selected_hulls_vectorized(selected_hulls, points_refined_2d)
        inside_mask_raw = is_inside_selected_hulls_vectorized(selected_hulls, points_raw_2d)
    else:
        inside_mask_refined = np.zeros(points_refined_2d.shape[0], dtype=bool)
        inside_mask_raw = np.zeros(points_raw_2d.shape[0], dtype=bool)
    
    points_refined = np.vstack((x_refined, y_refined, z_refined)).T
    points_raw = np.vstack((x, y, z)).T
    filtered_refined = points_refined[inside_mask_refined]
    filtered_raw = points_raw[inside_mask_raw]
    print(f"Filtered points: refined={len(filtered_refined)}, raw={len(filtered_raw)}")
    
    filtered_refined_x = filtered_refined[:, 0]
    filtered_refined_y = filtered_refined[:, 1]
    filtered_refined_z = filtered_refined[:, 2]
    filtered_raw_x = filtered_raw[:, 0]
    filtered_raw_y = filtered_raw[:, 1]
    filtered_raw_z = filtered_raw[:, 2]
    
    min_z_val = min(filtered_raw_z.min(), filtered_refined_z.min())
    z1_adj = filtered_raw_z - min_z_val
    z2_adj = filtered_refined_z - min_z_val
    x_min = min(filtered_raw_x.min(), filtered_refined_x.min())
    x_max = max(filtered_raw_x.max(), filtered_refined_x.max())
    y_min = min(filtered_raw_y.min(), filtered_refined_y.min())
    y_max = max(filtered_raw_y.max(), filtered_refined_y.max())
    grid_res = 0.07
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, grid_res),
                                 np.arange(y_min, y_max, grid_res))
    dem1 = griddata((filtered_raw_x, filtered_raw_y), z1_adj, (grid_x, grid_y), method='linear')
    dem2 = griddata((filtered_refined_x, filtered_refined_y), z2_adj, (grid_x, grid_y), method='linear')
    if selected_hulls:
        mask_grid = contains(unary_union(selected_hulls), grid_x, grid_y)
    else:
        mask_grid = np.ones_like(grid_x, dtype=bool)
    dem1[~mask_grid] = np.nan
    dem2[~mask_grid] = np.nan
    diff = dem2 - dem1
    diff[np.abs(diff) < reg_target_source] = 0
    cell_area = grid_res ** 2
    cut_volume = np.nansum(np.abs(diff[diff < 0])) * cell_area
    fill_volume = np.nansum(diff[diff > 0]) * cell_area
    net_volume = fill_volume - cut_volume
    print("Cut Volume (removed): {:.4f} m³".format(cut_volume))
    print("Fill Volume (added): {:.4f} m³".format(fill_volume))
    print("Net Volume Change: {:.4f} m³".format(net_volume))
    
    with open(os.path.join(log_dir, "change_result.txt"), "w") as log_file:
        log_file.write("Cut Volume (removed): {:.6f} m³\n".format(cut_volume))
        log_file.write("Fill Volume (added): {:.6f} m³\n".format(fill_volume))
        log_file.write("Net Volume Change: {:.6f} m³\n".format(net_volume))
    
    print("Analytical removed volume:", analytical_volume)
    finish_timestamp = datetime.datetime.now()
    elapsed_time = finish_timestamp - start_time
    elapsed_minutes = elapsed_time.total_seconds() / 60.0
    print("Elapsed time (min):", elapsed_minutes)
    
    volume_cut_acc = (analytical_volume + net_volume) / analytical_volume * 100
    print("Volume cut accuracy: {:.4f}%".format(volume_cut_acc))
    print("Noise std (mm):", noise_std * 1000)
    print("ICP_QUANTILE_THRESHOLD:", quantile_thr)
    print("Max depth:", max_depth)
    
    results = {
        "TE_Refined [m]": rte_refined,
        "RE_Refined [deg]": rre_refined,
        "Vol+_Est [m^3]": fill_volume,
        "Vol-_Est [m^3]": cut_volume,
        "Vol_Net_Est [m^3]": net_volume,
        "RE_Vol [%]": volume_cut_acc,
        "Time [min]": elapsed_minutes
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Sweep simulation parameters and output CSV results.")
    # Parameter sweep for noise_std and quantile threshold:
    parser.add_argument("--noise_start", type=float, required=True, help="Starting noise_std value in mm")
    parser.add_argument("--noise_step", type=float, required=True, help="Step for noise_std in mm")
    parser.add_argument("--noise_count", type=int, required=True, help="Number of noise_std steps")
    parser.add_argument("--quantile_start", type=float, required=True, help="Starting quantile threshold in percent")
    parser.add_argument("--quantile_step", type=float, required=True, help="Step for quantile threshold in percent")
    parser.add_argument("--quantile_count", type=int, required=True, help="Number of quantile steps")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Output CSV file")
    parser.add_argument("--base_log_dir", type=str, default="outputs", help="Base directory for log outputs")
    
    # Expose additional parameters from the original code:
    parser.add_argument("--input_file", type=str, default="/home/sam/Documents/datasets/WHU/3-Mountain/combined.las", help="Path to the input LAS file")
    parser.add_argument("--shp_file", type=str, default="/home/sam/Documents/datasets/WHU/3-Mountain/combined_stable.shp", help="Path to the shapefile")
    parser.add_argument("--radius", type=float, default=25, help="Radius of the mining region")
    parser.add_argument("--max_depth", type=float, default=15, help="Maximum depth at the center")
    parser.add_argument("--sigma", type=float, default=None, help="Spread of the deformation (if not provided, computed as radius/2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g. 'cuda:0')")
    parser.add_argument("--padding", type=str, default="same", help="Padding option for convolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--voxel_size", type=float, default=2.0, help="Voxel size")
    parser.add_argument("--PV", type=int, default=5, help="Fill positive value")
    parser.add_argument("--NV", type=int, default=-1, help="Fill negative value")
    parser.add_argument("--PPV", type=int, default=-1, help="PPV parameter")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of workers")
    parser.add_argument("--rotation_choice", type=str, default="AA_ICO162_S10_z_axis", help="Rotation choice")
    parser.add_argument("--icp_version", type=str, default="generalized", help="ICP version")
    parser.add_argument("--max_iter", type=int, default=2048, help="Max iterations for ICP")
    
    args = parser.parse_args()
    
    results_list = []
    for i in range(args.noise_count):
        noise_val_mm = args.noise_start + i * args.noise_step
        noise_std = noise_val_mm / 1000.0  # convert mm to m
        for j in range(args.quantile_count):
            quantile_val_percent = args.quantile_start + j * args.quantile_step
            quantile_thr = quantile_val_percent / 100.0  # convert percent to fraction
            print(f"Running simulation for noise_std = {noise_val_mm} mm, quantile_thr = {quantile_val_percent}%")
            sim_results = run_simulation(
                noise_std, quantile_thr, args.base_log_dir,
                args.input_file, args.shp_file, args.radius, args.max_depth, args.sigma,
                torch.device(args.device), args.padding, args.batch_size, args.voxel_size,
                args.PV, args.NV, args.PPV, args.num_workers, args.rotation_choice,
                args.icp_version, args.max_iter
            )
            result_entry = {
                "Noise [mm]": noise_val_mm,
                "q [%]": quantile_val_percent,
                "TE_Refined [m]": sim_results["TE_Refined [m]"],
                "RE_Refined [deg]": sim_results["RE_Refined [deg]"],
                "Vol+_Est [m^3]": sim_results["Vol+_Est [m^3]"],
                "Vol-_Est [m^3]": sim_results["Vol-_Est [m^3]"],
                "Vol_Net_Est [m^3]": sim_results["Vol_Net_Est [m^3]"],
                "RE_Vol [%]": sim_results["RE_Vol [%]"],
                "Time [min]": sim_results["Time [min]"]
            }
            results_list.append(result_entry)
    
    fieldnames = ["Noise [mm]", "q [%]", "TE_Refined [m]", "RE_Refined [deg]",
                  "Vol+_Est [m^3]", "Vol-_Est [m^3]", "Vol_Net_Est [m^3]", "RE_Vol [%]", "Time [min]"]
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results_list:
            writer.writerow(entry)
    
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
