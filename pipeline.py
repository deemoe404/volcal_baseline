import datetime
import os
import sys
from pathlib import Path

import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import geopandas as gpd
import py4dgeo
import py4dgeo.epoch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.vectorized import contains
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

current_dir = os.path.abspath(".")
submodule_dir = os.path.join(current_dir, "exhaustive-grid-search")
sys.path.append(submodule_dir)

from fft_conv_pytorch import fft_conv
from utils.pc_utils import voxelize, unravel_index_pytorch
from utils.data_utils import preprocess_pcj_B1
from utils.rot_utils import create_T_estim_matrix, load_rotations, homo_matmul
from utils.padding import padding_options


def estimate_avg_spacing(points, k=2):
    """
    Estimate the average spacing between points in a point cloud.
    :param points: (N, 3) numpy array of point coordinates.
    :param k: Number of nearest neighbors to query (k=2 means self and 1 neighbor).
    :return: Average distance to the 1st nearest neighbor (excluding self).
    """
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k)
    # distances[:, 0] are zeros (distance to self)
    avg_spacing = np.mean(distances[:, 1])
    return avg_spacing

def voxel_downsample(points, voxel_size):
    """
    Downsample the point cloud using a voxel grid.
    :param points: (N, 3) numpy array of point coordinates.
    :param voxel_size: Side length of the cubic voxel.
    :return: Downsampled point cloud as a numpy array.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(down_pcd.points)
    return down_points

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def adaptive_voxel_size(points, ref_ratio, init_voxel_size, kp=25, ki=15, lr=1):
    """
    Adaptively adjust voxel size so that the downsampling ratio (number of points after voxelization / original points)
    converges to a target reference ratio.
    :param points: (N, 3) numpy array of point coordinates.
    :param ref_ratio: Desired downsampling ratio (e.g., 0.1 for 10% of original points).
    :param init_voxel_size: Initial guess for the voxel size.
    :param num_iter: Number of iterations to run the control loop.
    :param lr: Learning rate for the proportional control.
    :return: Adapted voxel size.
    """
    voxel_size = init_voxel_size
    error_sum = 0.0
    scale = 1.0
    while True:
        down_points = voxel_downsample(points, voxel_size)
        error = ref_ratio - (len(down_points)/len(points))
        print(len(down_points)/len(points))
        if abs(error) < 0.001:
            break
        error_sum += error
        diff = kp * error + ki * error_sum
        scale = scale - lr * (sigmoid(diff) - 0.5)
        voxel_size = init_voxel_size * np.exp(scale)
        
        print(f"Voxel size: {voxel_size}, error: {error}, scale: {scale}")
    return voxel_size


def compute_rre(T_est, T_gt):
    """
    Compute the rotational relative error (RRE) between an estimated and a ground truth transformation.

    This function calculates the angular error between two 4x4 transformation matrices
    by computing the error transformation matrix, extracting its rotation component,
    and determining the angle (in degrees) that represents the rotation error.

    Parameters:
        T_est (array-like): The estimated 4x4 transformation matrix.
        T_gt (array-like): The ground truth 4x4 transformation matrix.

    Returns:
        float: The rotational error in degrees, computed as the angle corresponding to the rotation mismatch.
    """
    T_error = T_gt @ np.linalg.inv(T_est)
    R_error = T_error[:3, :3]
    cos_theta = (np.trace(R_error) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def compute_rte(T_est, T_gt):
    """
    Computes the relative translation error (RTE) between the estimated and ground truth transformation matrices.

    This function calculates the error transformation by multiplying the ground truth matrix (T_gt) with the inverse of the estimated matrix (T_est). It then extracts the translation component from the resulting error transformation and computes its Euclidean norm, which represents the magnitude of the translation error.

    Parameters:
        T_est (numpy.ndarray): A 4x4 transformation matrix representing the estimated pose.
        T_gt (numpy.ndarray): A 4x4 transformation matrix representing the ground truth pose.

    Returns:
        float: The Euclidean norm of the translation error between the estimated and ground truth poses.
    """
    T_error = T_gt @ np.linalg.inv(T_est)
    t_error = T_error[:3, 3]
    return np.linalg.norm(t_error)


def is_inside_selected_hulls_vectorized(selected_hulls, pts_2d):
    """
    Determines whether each point in a 2D array is located inside the union of the provided hulls.
    Parameters:
        selected_hulls (iterable): A collection of geometric hulls (e.g., Shapely Polygon objects)
                                   that define the areas of interest.
        pts_2d (numpy.ndarray): A 2D array of points with shape (n_points, 2) where each row represents
                                the (x, y) coordinates of a point.
    Returns:
        numpy.ndarray: A boolean mask array of shape (n_points,) where each element is True if the
                       corresponding point lies inside the union of the selected hulls, and False otherwise.
    """
    union_hull = unary_union(selected_hulls)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    mask = contains(union_hull, x, y)
    return mask


def get_transform_matrix(
    r_x: float, r_y: float, r_z: float, t_x: float, t_y: float, t_z: float
):
    """
    Create a transformation matrix from rotation angles and translation values.

    Parameters:
    r_x (float): Rotation angle around the x-axis in degrees.
    r_y (float): Rotation angle around the y-axis in degrees.
    r_z (float): Rotation angle around the z-axis in degrees.
    t_x (float): Translation along the x-axis in meters.
    t_y (float): Translation along the y-axis in meters.
    t_z (float): Translation along the z-axis in meters.

    Returns:
    np.ndarray: A 4x4 transformation matrix.
    """
    angle_x = np.deg2rad(r_x)
    angle_y = np.deg2rad(r_y)
    angle_z = np.deg2rad(r_z)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    T = np.array([[t_x], [t_y], [t_z]])
    T_est = np.eye(4)
    T_est[:3, :3] = R
    T_est[:3, 3] = T[:, 0]
    return T_est


def sample_range(range_val):
    """
    Sample a value from a given range.
    
    If range_val is a tuple of two numbers, sample uniformly from that interval.
    If range_val is a tuple of tuples/lists (i.e. multiple intervals),
    randomly choose one interval and sample uniformly from it.
    """
    # Check if the first element is a tuple/list: indicates multiple ranges
    if isinstance(range_val[0], (list, tuple)):
        chosen_range = range_val[np.random.randint(len(range_val))]
        return np.random.uniform(*chosen_range)
    else:
        return np.random.uniform(*range_val)

def get_random_transformation(angle_range_x=(-5, 5),
                              angle_range_y=(-5, 5),
                              angle_range_z=((-50, -40), (40, 50)),
                              translation_range=((-15, -5), (5, 15))):
    """
    Generate a random 4x4 transformation matrix with independent rotation and translation ranges.
    
    The range parameters can be provided as:
      - A tuple (min, max) for a single continuous range.
      - A tuple of tuples for a union of ranges (e.g., ((min1, max1), (min2, max2))).
    
    For example, to have z-axis rotation be either -45° ± 5° or +45° ± 5°:
      angle_range_z = ((-50, -40), (40, 50))
    
    Similarly, translation_range can be provided as a union of ranges.
    
    Returns:
      np.ndarray: A 4x4 transformation matrix computed by get_transform_matrix().
    """
    angle_x = sample_range(angle_range_x)
    angle_y = sample_range(angle_range_y)
    angle_z = sample_range(angle_range_z)
    
    translation_x = sample_range(translation_range)
    translation_y = sample_range(translation_range)
    translation_z = sample_range(translation_range)
    
    T = get_transform_matrix(
        angle_x, angle_y, angle_z,
        translation_x, translation_y, translation_z
    )
    return T


def apply_transformation(x: np.ndarray, y: np.ndarray, z: np.ndarray, T: np.ndarray):
    """
    Apply a transformation to a point cloud.

    Parameters:
    x (np.ndarray): X-coordinates of the point cloud.
    y (np.ndarray): Y-coordinates of the point cloud.
    z (np.ndarray): Z-coordinates of the point cloud.
    T (np.ndarray): A 4x4 transformation matrix.

    Returns:
    np.ndarray: Transformed X-coordinates
    np.ndarray: Transformed Y-coordinates
    np.ndarray: Transformed Z-coordinates
    """
    pts = np.vstack((x, y, z, np.ones_like(x)))
    pts_transformed = T @ pts
    x_transformed = pts_transformed[0, :]
    y_transformed = pts_transformed[1, :]
    z_transformed = pts_transformed[2, :]
    return x_transformed, y_transformed, z_transformed


def get_analytical_volume(s: float, r: float, d: float):
    """
    Calculate the analytical volume removed due to deformation.

    Parameters:
    s (float): Spread of the deformation (sigma).
    r (float): Radius of the mining region.
    d (float): Maximum depth at the center.

    Returns:
    float: Analytical volume removed (in cubic meters).
    """
    analytical_volume = 2 * np.pi * s**2 * d * (1 - np.exp(-(r**2) / (2 * s**2)))
    return analytical_volume


def apply_noise(x: np.ndarray, y: np.ndarray, z: np.ndarray, std: float):
    """
    Apply Gaussian noise to a point cloud.

    Parameters:
    x (np.ndarray): X-coordinates of the point cloud.
    y (np.ndarray): Y-coordinates of the point cloud.
    z (np.ndarray): Z-coordinates of the point cloud.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    np.ndarray: Noisy X-coordinates
    np.ndarray: Noisy Y-coordinates
    np.ndarray: Noisy Z-coordinates
    """
    noise_x = np.random.normal(loc=0, scale=std, size=x.shape)
    noise_y = np.random.normal(loc=0, scale=std, size=y.shape)
    noise_z = np.random.normal(loc=0, scale=std, size=z.shape)
    x_noisy = x + noise_x
    y_noisy = y + noise_y
    z_noisy = z + noise_z
    return x_noisy, y_noisy, z_noisy


def apply_deformation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c_x: float,
    c_y: float,
    r: float,
    d: float,
    s: float,
):
    """
    Apply a radial deformation to a point cloud.

    Parameters:
    x (np.ndarray): X-coordinates of the point cloud.
    y (np.ndarray): Y-coordinates of the point cloud.
    z (np.ndarray): Z-coordinates of the point cloud.
    c_x (float): Center x of the deformation.
    c_y (float): Center y of the deformation.
    r (float): Radius of the mining region.
    d (float): Maximum depth at the center.
    s (float): Spread of the deformation.

    Returns:
    np.ndarray: Deformed Z-coordinates
    np.ndarray: Mask of the deformed region
    """
    dist = np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
    mask = dist < r

    z_deformed = z.copy()
    deformation = d * np.exp(-(dist**2) / (2 * s**2))
    z_deformed[mask] -= deformation[mask]
    return z_deformed, mask


def EGS(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    padding: str,
    ppv: int,
    pv: int,
    nv: int,
    num_workers: int,
    rotation_choice: str,
    rotation_root_path: str,
    DEVICE: str = "cuda",
):
    """
    Exhaustive Grid Search (EGS) algorithm for point cloud registration.

    Parameters:
    source (np.ndarray): Source point cloud.
    target (np.ndarray): Target point cloud.
    voxel_size (float): Voxel size for voxelization.
    padding (str): Padding type for voxelization.
    ppv (int): Positive padding value for voxelization.
    pv (int): Positive voxel value for voxelization.
    nv (int): Negative voxel value for voxelization.
    num_workers (int): Number of workers for data loading.
    rotation_choice (str): Rotation choice for the EGS algorithm.
    rotation_root_path (str): Root path for the precomputed rotations matrix files.

    Returns:
    np.ndarray: Estimated transformation matrix.
    """
    pci = torch.from_numpy(target)
    pcj = torch.from_numpy(source)

    make_pci_posit_translation = torch.min(pci, axis=0)[0]
    pci = pci - make_pci_posit_translation

    pci_voxel, NR_VOXELS_PCI = voxelize(
        pci, voxel_size, fill_positive=pv, fill_negative=nv
    )

    CENTRAL_VOXEL_PCI = torch.where(
        NR_VOXELS_PCI % 2 == 0,
        (NR_VOXELS_PCI / 2) - 1,
        torch.floor(NR_VOXELS_PCI / 2),
    ).int()
    central_voxel_center = CENTRAL_VOXEL_PCI * voxel_size + (0.5 * voxel_size)
    weight_to_fftconv3d = pci_voxel.type(torch.int32).to(DEVICE)[None, None, :, :, :]

    pp, pp_xyz = padding_options(padding, CENTRAL_VOXEL_PCI, NR_VOXELS_PCI)
    R_batch = load_rotations(
        rotation_choice=rotation_choice, rot_root_path=rotation_root_path
    )
    my_data, my_dataloader = preprocess_pcj_B1(
        pcj, R_batch, voxel_size, pp, num_workers, pv, nv, ppv
    )

    maxes = []
    argmaxes = []
    shapes = []
    minimas = torch.empty(R_batch.shape[0], 3)

    torch.cuda.empty_cache()
    with torch.no_grad():
        for ind_dataloader, (voxelized_pts_padded, mins, orig_input_shape) in tqdm(
            enumerate(my_dataloader),
            total=len(my_dataloader),
            desc="FFT Convolution",
            smoothing=0.1,
        ):
            minimas[ind_dataloader, :] = mins
            input_to_fftconv3d = voxelized_pts_padded.to(DEVICE)
            out = fft_conv(input_to_fftconv3d, weight_to_fftconv3d, bias=None)
            maxes.append(torch.max(out))
            argmaxes.append(torch.argmax(out))
            shapes.append(out.shape)
            
            torch.cuda.empty_cache()

    m_index = torch.argmax(torch.stack(maxes))
    ind0, _, ind1, ind2, ind3 = unravel_index_pytorch(
        argmaxes[m_index], shapes[m_index]
    )
    rotation_index = m_index + ind0
    R = R_batch[rotation_index]

    t = torch.Tensor(
        [
            -(pp_xyz[0] * voxel_size)
            + ((CENTRAL_VOXEL_PCI[0]) * voxel_size)
            + (ind1 * voxel_size)
            + (0.5 * voxel_size),
            -(pp_xyz[2] * voxel_size)
            + ((CENTRAL_VOXEL_PCI[1]) * voxel_size)
            + (ind2 * voxel_size)
            + (0.5 * voxel_size),
            -(pp_xyz[4] * voxel_size)
            + ((CENTRAL_VOXEL_PCI[2]) * voxel_size)
            + (ind3 * voxel_size)
            + (0.5 * voxel_size),
        ]
    )

    center_pcj_translation = my_data.center
    make_pcj_posit_translation = minimas[rotation_index]
    estim_T_baseline = create_T_estim_matrix(
        center_pcj_translation,
        R,
        make_pcj_posit_translation,
        central_voxel_center,
        t,
        make_pci_posit_translation,
    )

    return estim_T_baseline.numpy()


def auto_GICP_TEST(
    source: np.ndarray,
    target: np.ndarray,
    T_init: np.ndarray,
    thr: float,
    neigh,
    max_iter: int = 2048,
):
    """
    Generalized Iterative Closest Point (GICP) algorithm for point cloud registration.

    Parameters:
    source (np.ndarray): Source point cloud.
    target (np.ndarray): Target point cloud.
    T_init (np.ndarray): Initial transformation matrix.
    thr (float): Threshold for auto max distance determination by quantile.
    max_iter (int): Maximum number of iterations.

    Returns:
    np.ndarray: Estimated transformation matrix.
    """
    source_np_estim = homo_matmul(source, T_init)

    dist, _ = neigh.kneighbors(source_np_estim)
    adaptive_thr = np.quantile(dist, thr)
    
    print(adaptive_thr)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iter,
    )

    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target)
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)
    reg_o3d = o3d.pipelines.registration.registration_generalized_icp(
        source_o3d, target_o3d, adaptive_thr, T_init, criteria=criteria
    )

    return reg_o3d.transformation


def auto_GICP(
    source: np.ndarray,
    target: np.ndarray,
    T_init: np.ndarray,
    thr: float,
    max_iter: int = 2048,
):
    """
    Generalized Iterative Closest Point (GICP) algorithm for point cloud registration.

    Parameters:
    source (np.ndarray): Source point cloud.
    target (np.ndarray): Target point cloud.
    T_init (np.ndarray): Initial transformation matrix.
    thr (float): Threshold for auto max distance determination by quantile.
    max_iter (int): Maximum number of iterations.

    Returns:
    np.ndarray: Estimated transformation matrix.
    """
    source_np_estim = homo_matmul(source, T_init)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(target)
    dist, _ = neigh.kneighbors(source_np_estim)
    adaptive_thr = np.quantile(dist, thr)
    
    print(adaptive_thr)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iter,
    )

    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target)
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)
    reg_o3d = o3d.pipelines.registration.registration_generalized_icp(
        source_o3d, target_o3d, adaptive_thr, T_init, criteria=criteria
    )

    return reg_o3d.transformation


def isolate_stable(cloud: np.ndarray, shp: gpd.GeoDataFrame, crs: str = "EPSG:4326"):
    """
    Isolate the stable region from a point cloud.

    Parameters:
    cloud (np.ndarray): Point cloud to isolate.
    shp (gpd.GeoDataFrame): Shapefile of the stable region.
    crs (str): Coordinate reference system.

    Returns:
    np.ndarray: Isolated point cloud.
    np.ndarray: Mask of the stable region
    """
    cloud_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cloud[:, 0], cloud[:, 1], cloud[:, 2]),
        crs=crs,
    )
    cloud_shp = gpd.sjoin(cloud_gdf, shp, predicate="within")
    result = cloud[cloud_shp.index]

    mask = np.ones(len(cloud[:, 0]), dtype=bool)
    mask[cloud_shp.index] = False

    return result, mask


from kneed import KneeLocator
from scipy import stats

def plot_k_distance_curve(cp: np.ndarray, cs: np.ndarray):
    """
    Plot the sorted 2nd nearest neighbor distances (k-distance curve)
    and mark the knee point used as the adaptive epsilon.
    
    Parameters:
        cp (np.ndarray): Core points of the point cloud.
        cs (np.ndarray): Boolean mask for significant changes.
    """
    # Select only the significant points
    sig_cp = cp[cs]
    
    # Compute distances to the 2nd nearest neighbor
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(sig_cp)
    distances, _ = neigh.kneighbors(sig_cp)
    second_nn_dists = distances[:, 1]
    
    # Sort distances to form the k-distance curve
    sorted_dists = np.sort(second_nn_dists)
    
    # Detect knee point using KneeLocator
    kneedle = KneeLocator(
        x=range(len(sorted_dists)), 
        y=sorted_dists, 
        curve="convex", 
        direction="increasing"
    )
    
    knee_index = kneedle.knee
    adaptive_eps = sorted_dists[knee_index] if knee_index is not None else None
    
    # Plot the k-distance curve
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_dists, label="2nd NN distances")
    
    if knee_index is not None:
        plt.axvline(x=knee_index, color='red', linestyle='--', label=f"Knee at index {knee_index}")
        plt.scatter(knee_index, adaptive_eps, color='red', 
                    label=f"Adaptive eps: {adaptive_eps:.2f}")
    
    plt.xlabel("Sorted Point Index")
    plt.ylabel("2nd Nearest Neighbor Distance")
    plt.title("k-Distance Graph for Knee Detection")
    plt.legend()
    plt.show()

def segment_changes_knee(cp: np.ndarray, cs: np.ndarray, min_samples: int = 10):
    """
    Segment changes in a point cloud using knee detection to automatically determine eps.
    
    Parameters:
        cp (np.ndarray): Core points of the point cloud.
        cs (np.ndarray): Boolean array indicating significant changes.
        min_samples (int): Minimum samples for DBSCAN.
        
    Returns:
        list: List of convex hulls (shapely.geometry.MultiPoint) for each cluster.
    """
    sig_cp = cp[cs][:, :2]
    
    # Compute distances to the second nearest neighbor
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(sig_cp)
    distances, _ = neigh.kneighbors(sig_cp)
    second_nn_dists = distances[:, 1]
    
    # Sort distances to form the k-distance curve
    sorted_dists = np.sort(second_nn_dists)
    
    # Use KneeLocator to find the knee point automatically
    kneedle = KneeLocator(
        x=range(len(sorted_dists)), 
        y=sorted_dists, 
        curve="convex", 
        direction="increasing"
    )
    
    adaptive_eps = sorted_dists[kneedle.knee] if kneedle.knee is not None else np.quantile(second_nn_dists, 0.9)
    print(f"Adaptive Epsilon (knee): {adaptive_eps:.4f}")
    
    # Cluster using DBSCAN on the 2D projection (e.g. x, y)
    sig_cp_2d = sig_cp[:, :2]
    db = DBSCAN(eps=adaptive_eps, min_samples=min_samples).fit(sig_cp_2d)
    labels = db.labels_
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    
    hulls = []
    for lbl in unique_labels:
        cluster_points = sig_cp_2d[labels == lbl]
        if len(cluster_points) >= 3:
            hull = MultiPoint(cluster_points).convex_hull
            hulls.append(hull)
    
    return hulls

def segment_changes(
    cp: np.ndarray, cs: np.ndarray, thr: float = -0.2, min_samples: int = 2
):
    """
    Segment changes in a point cloud.

    Parameters:
    cp (np.ndarray): Core points of the point cloud.
    cs (np.ndarray): Change sign of the core points.
    eps_thr (float): Epsilon threshold for DBSCAN.
    min_samples (int): Minimum number of samples for DBSCAN.

    Returns:
    list: List of hulls of the changes.
    """
    sig_cp_2d = cp[cs][:, :2]

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(sig_cp_2d)
    dist, _ = neigh.kneighbors(sig_cp_2d)
    second_nn_dists = dist[:, 1]
    
    avarage_eps = np.mean(second_nn_dists)
    divation = np.std(second_nn_dists)
    
    adaptive_eps = avarage_eps + thr * divation

    db = DBSCAN(eps=adaptive_eps, min_samples=min_samples).fit(sig_cp_2d)
    labels = db.labels_
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]

    hulls = []
    for lbl in unique_labels:
        cluster_points = sig_cp_2d[labels == lbl]
        if len(cluster_points) >= 3:
            hull = MultiPoint(cluster_points).convex_hull
            hulls.append(hull)

    return hulls


def reletive_DEM(
    cloud_1: np.ndarray,
    cloud_2: np.ndarray,
    grid_res: float = None,
    method: str = "linear",
    mask_hulls: list = None,
    grid_res_factor: float = 1.0,
):
    """
    Generate a pair of Digital Elevation Models (DEMs) from two point clouds.

    Parameters:
    cloud_1 (np.ndarray): First point cloud.
    cloud_2 (np.ndarray): Second point cloud.
    grid_res (float): Grid resolution for the DEMs.

    Returns:
    np.ndarray: First DEM.
    np.ndarray: Second DEM.
    """
    # Auto-compute grid resolution if not provided
    if grid_res is None:
        tree_1 = cKDTree(cloud_1[:, :2])
        distances_1, _ = tree_1.query(cloud_1[:, :2], k=2)
        median_nn_1 = np.median(distances_1[:, 1])
        grid_res_1 = grid_res_factor * median_nn_1
        
        tree_2 = cKDTree(cloud_2[:, :2])
        distances_2, _ = tree_2.query(cloud_2[:, :2], k=2)
        median_nn_2 = np.median(distances_2[:, 1])
        grid_res_2 = grid_res_factor * median_nn_2
        
        grid_res = max(grid_res_1, grid_res_2)

    min_z = min(cloud_1[:, 2].min(), cloud_2[:, 2].min())
    z1_adjusted = cloud_1[:, 2] - min_z
    z2_adjusted = cloud_2[:, 2] - min_z

    x_min = min(cloud_1[:, 0].min(), cloud_2[:, 0].min())
    x_max = max(cloud_1[:, 0].max(), cloud_2[:, 0].max())
    y_min = min(cloud_1[:, 1].min(), cloud_2[:, 1].min())
    y_max = max(cloud_1[:, 1].max(), cloud_2[:, 1].max())

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_res), np.arange(y_min, y_max, grid_res)
    )

    dem1 = griddata(
        (cloud_1[:, 0], cloud_1[:, 1]), z1_adjusted, (grid_x, grid_y), method=method
    )

    dem2 = griddata(
        (cloud_2[:, 0], cloud_2[:, 1]), z2_adjusted, (grid_x, grid_y), method=method
    )

    if mask_hulls is not None:
        union = unary_union(mask_hulls)
        mask_grid = contains(union, grid_x, grid_y)
        dem1[~mask_grid] = np.nan
        dem2[~mask_grid] = np.nan

    return dem1, dem2, grid_x, grid_y, grid_res


def calculate_volume(
    dem_before: np.ndarray,
    dem_after: np.ndarray,
    grid_res: float = 0.07,
    threshold: float = 0,
):
    """
    Calculate the volume difference between two DEMs.

    Parameters:
    dem_before (np.ndarray): DEM before deformation.
    dem_after (np.ndarray): DEM after deformation.
    grid_res (float): Grid resolution for the DEMs.
    threshold (float): Threshold for considering a change in DEM.

    Returns:
    float: Volume difference.
    """
    diff = dem_after - dem_before
    diff[np.abs(diff) < threshold] = 0

    cell_area = grid_res**2
    cut_volume = np.nansum(np.abs(diff[diff < 0])) * cell_area
    fill_volume = np.nansum(diff[diff > 0]) * cell_area
    net_volume = fill_volume - cut_volume

    return net_volume, cut_volume, fill_volume, diff
