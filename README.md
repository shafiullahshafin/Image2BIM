# Image2BIM: Indoor Image-to-BIM Reconstruction

An end-to-end reconstruction pipeline that transforms indoor sensor scans into certified 3D IFC4 BIM models, multi-class LAS point clouds, and measured 2D architectural drawing sheets (PDF & PNG).

---

## Overview

Image2BIM automates the translation of indoor panoramic imagery and sensor data into building information models:

* **Canonical Centerline Partitions**: Eliminates duplicate overlapping walls across adjacent rooms by constructing single shared partition walls along their true median axes.
* **Bilateral Opening Visibility**: Generates through-wall boolean cuts (`IfcOpeningElement`), ensuring doors and windows are fully visible from both sides of the wall in standard viewers like BIMvision and Revit.
* **IFC4 Compliance**: Builds a valid spatial hierarchy (`IfcProject` → `IfcSite` → `IfcBuilding` → `IfcBuildingStorey` → Elements) with classified room spaces (`IfcSpace`) and polygonal floor/ceiling slabs (`IfcSlab`).
* **CAD Deliverables**: Generates 2D architectural floor plan sheets with charcoal wall poche, dimension chains, room areas, and door swing arcs.

---

## Input Data Specifications

For each scan station, the pipeline ingests:

| Input Data | Format | Description |
| :--- | :--- | :--- |
| **360° Panoramic Images** | `.png` / `.jpg` | Equirectangular RGB panoramas ($360^\circ \times 180^\circ$) capturing the full indoor sphere. |
| **Metric Depth Maps** | `.png` (16-bit) | Metric distance in millimeters from the camera center for every pixel. |
| **Camera Coordinates** | `.txt` | Camera station optical center translation `[X Y Z]` in millimeters. |
| **Surface Normal Maps** | `.png` | Pixel-wise surface normal vectors for projection density. |
| **CAD Annotations** | `annotations.json` | 3D bounding boxes and semantic labels for doors and windows. |

*Benchmark Dataset: Developed and evaluated on [Structured3D](https://structured3d-dataset.org/) ([GitHub](https://github.com/bertjiazheng/Structured3D)).*

### Standard Input Directory Layout

```
dataset/
└── <dataset_name>/
    └── <scene_id>/
        ├── annotations.json            # [Optional] 3D door/window annotations
        └── stations/
            ├── station_01/
            │   ├── rgb.png             # 360° Panoramic RGB image
            │   ├── depth.png           # 16-bit metric depth map (mm)
            │   ├── normal.png          # [Optional] Surface normals
            │   └── camera.txt          # Camera coordinates [X Y Z]
            └── station_02/
                └── ...
```

---

## Installation

### 1. Clone & Environment Setup

```bash
git clone https://github.com/shafiullahshafin/Image2BIM.git
cd Image2BIM

conda create -n image2bim python=3.10 -y
conda activate image2bim
```

### 2. Install Dependencies

```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Geometry, point clouds, BIM, and dependencies
pip install -r requirements.txt
```

*(Or manual installation: `pip install ifcopenshell shapely laspy reportlab matplotlib opencv-python Pillow numpy scipy einops`)*

### 3. Pretrained Weights

Place `checkpoint.pth` into `checkpoints/`:

* **Weights**: [Download HEAT Checkpoint](https://www.dropbox.com/scl/fi/57kxrtdwma8h9m2osnjn5/heat_checkpoints.zip?rlkey=77cso90mi4aroj4wpbh0w2tiv&st=k1oi772j&dl=0) *(extract `ckpts_heat_s3d_256/checkpoint.pth`)*

---

## Usage

All operations are executed via `run_pipeline.py`.

### Reconstruct a Single Scene

```powershell
python run_pipeline.py --scene_id 00000 --dataset structured3d
```

### Batch Reconstruct All Scenes

```powershell
python run_pipeline.py --all --dataset structured3d
```

### Run on a Custom Dataset

```powershell
python run_pipeline.py --scene_id room_01 --dataset custom_scans --raw_dir D:/Data/custom_scans
```

---

## CLI Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--scene_id` | `str` | `None` | Scene identifier to process (e.g. `00000`). |
| `--all` | `flag` | `False` | Batch processes all scenes detected in the dataset directory. |
| `--dataset` | `str` | `structured3d` | Name of the dataset folder under `dataset/` and `results/`. |
| `--raw_dir` | `str` | `None` | Path to custom input root (overrides `dataset/<dataset>`). |
| `--out_dir` | `str` | `None` | Path to custom output directory (overrides default results path). |
| `--ckpt_path` | `str` | `checkpoints/checkpoint.pth` | Path to HEAT neural model weights. |
| `--openings_source` | `str` | `dataset` | Source for doors/windows: `dataset` or `none`. |
| `--wall_height` | `float` | `2.80` | Default wall height in meters. |
| `--wall_thickness` | `float` | `0.20` | Exterior wall thickness in meters (median partitions scale to $0.12 - 0.15\,\text{m}$). |
| `--slab_thickness` | `float` | `0.15` | Floor and ceiling slab thickness in meters. |
| `--subsample` | `int` | `2` | Point cloud backprojection subsampling step. |

---

## Output Deliverables

Output files are organized under `results/<dataset>/scene_<scene_id>/` (e.g. `results/structured3d/scene_00000/`):

* **BIM Models**:
  * `<scene_id>_model.ifc`: Fully assembled IFC4 model containing walls, slabs, spaces, doors, and windows with through-wall cutouts.
* **Point Clouds**:
  * `<scene_id>_raw_scan.las`: Full 3D scan point cloud backprojected from panoramic depth.
  * `<scene_id>_predicted_walls.las`: Extracted wall boundary point cloud.
  * `<scene_id>_graph_corners.las`: Sampled corner junction pillar point cloud.
  * `<scene_id>_scan_and_walls_overlay.las`: Multi-class overlay combining raw scan, predicted walls, and corner junctions.
* **Architectural Drawings**:
  * `<scene_id>_floorplan_sheet.pdf`: High-resolution vector CAD drawing sheet with title block, room tags, dimensions, and door arcs.
  * `<scene_id>_floorplan_sheet.png`: High-resolution raster rendering for web and mobile viewers.
* **Structured Data**:
  * `<scene_id>_openings.json`: Extracted door and window 3D boundaries and positions.
  * `<scene_id>_floorplan.npy`: Vector planar graph and corner coordinates.
* **Metrics**:
  * `<scene_id>_metrics.json`: Precision, Recall, mean corner error (cm), and footprint match against CAD ground truth.

---

## References

* **Structured3D**: [Website](https://structured3d-dataset.org/) | [Paper](https://arxiv.org/abs/1908.00222)
* **HEAT**: [GitHub](https://github.com/woodfrog/heat) | [Paper](https://arxiv.org/abs/2203.12759)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For inquiries or to request the benchmark validation dataset, reach out to [m.shafiullah@hawarit.com](mailto:m.shafiullah@hawarit.com).
