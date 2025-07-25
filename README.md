# volcal_baseline

_A data-driven pipeline for large-scale open-pit-mine excavation monitoring from multi-temporal 3-D point clouds_

[![Journal](https://img.shields.io/badge/Journal-IET%20Image%20Processing-blue)](https://ietresearch.onlinelibrary.wiley.com/journal/17519659)
[![DOI](https://img.shields.io/badge/DOI-10.1049%2Fipr2.70130-blue)](https://doi.org/10.1049/ipr2.70130)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📢 **Announcements**
> - **[Jul 26, 2025]**: The pipeline is now available on **Android embedded devices** (e.g. DJI Remote).  
>   👉 [volcal_baseline_android](https://github.com/deemoe404/volcal_baseline_android)
> - **[Jul 25, 2025]**: The pipeline is now available as a **web service**.  
>   👉 [volcal_baseline_server](https://github.com/deemoe404/volcal_baseline_server)

`volcal_baseline` is the reference implementation that accompanies our paper  
**“A Data-driven solution for large-scale open-pit mines excavation monitoring based on 3-D point cloud”** published in _IET Image Processing_ (2025). The code turns two (or more) raw point-cloud epochs into sub-percent volumetric change estimates **without manual parameter tuning** by chaining:

1. **Exhaustive Grid Search (EGS)** coarse alignment  
2. **Generalised ICP (G-ICP)** fine alignment (adaptive correspondence distance)  
3. **M3C2** robust 3-D change detection with local LoD uncertainty  
4. **DEM-based cut / fill computation** with adaptive grid resolution  

Experiments on the WHU–TLS benchmark and a real UAV survey from Tongjiang quarry show **< 1 % volume error for volumes > 800 m³ and < 2 % under worst-case noise & mis-alignment**. Datasets we used are listed below, and the code is available as a Jupyter notebook for quick start.

| Dataset | Link | Notes |
|-|-|-|
| **WHU–TLS benchmark** | https://3s.whu.edu.cn/ybs/en/benchmark.htm | 115 terrestrial scans used for robustness tests |
| **Tongjiang Aerial (UAV)** | https://zenodo.org/records/15614501 | 4 missions over active limestone quarry |

You can download the pre-processed datasets & logs from the [Internet Archive](https://archive.org/details/volcal_baseline). The final folder structure should look like this:

```text
.
├── exhaustive-grid-search/    # submodule (coarse alignment)
├── py4dgeo/                   # submodule fork (M3C2 implementation)
├── datasets/                  # Pre-processed datasets
│   ├── WHU-TLS
│   └── Tongjiang_Aerial
├── sweep/
│   ├── batch_run.py           # Benchmarking script
│   ├── batch_visualise.py     # Benchmarking visualisation script
│   ├── exp_*.py               # Ablation study scripts
│   └── vis_*.py               # Ablation study visualisation scripts
├── pipeline.py                # Orchestrates the workflow
├── requirements.txt           # core Python deps
├── rotation_modifier.ipynb    # Jupyter notebook to build custom rotation presets
├── exp_WHU-TLS.ipynb          # WHU-TLS dataset
└── exp_Tongjiang_Aerial.ipynb # Tongjiang Aerial
```

---

## Quick start

For a quick start, clone the repository and install the dependencies using conda and pip. The code is tested with Python 3.11 and requires a capable GPU/iGPU for EGS. The following commands will set up the environment:

```bash
# Clone incl. submodules (EGS and Py4DGeo forks)
git clone --recursive https://github.com/deemoe404/volcal_baseline.git
cd volcal_baseline

# Create conda environment with Python 3.11
conda create -n volcal_env python=3.11 libstdcxx-ng -c conda-forge -y
conda activate volcal_env

# Install dependencies
pip install -r requirements.txt

cd py4dgeo
python -m pip install -v --editable .
```

Then, open the Jupyter notebooks `exp_WHU-TLS` or `exp_Tongjiang_Aerial` to run the pipeline on the corresponding datasets. The notebooks contain detailed instructions on how to run the pipeline and visualise the results.

## License

This project is released under the MIT License – see `LICENSE` for details.
