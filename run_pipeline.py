"""
# ==============================================================================
# SCAN-TO-BIM PRODUCTION PIPELINE
# ==============================================================================
#
# QUICK COMMANDS TO RUN THIS PIPELINE:
#
# 1. Run on Scene 00000 using dataset sensors (Structured3D default):
#    python run_pipeline.py --scene_id 00000 --dataset structured3d
#
# 2. Run on all scenes:
#    python run_pipeline.py --all --dataset structured3d
#
# 3. Run on another dataset (e.g. matterport3d):
#    python run_pipeline.py --scene_id 00000 --dataset matterport3d
#
# 4. Custom output directory:
#    python run_pipeline.py --scene_id 00000 --dataset structured3d --out_dir ./results/my_custom_run
#
# ==============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    DATASETS_CONFIG,
    DATASET_DIR,
    DEFAULT_HEAT_CHECKPOINT,
    RESULTS_DIR,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WALL_HEIGHT,
    DEFAULT_WALL_THICKNESS,
    DEFAULT_SLAB_THICKNESS,
    DEFAULT_CEILING_TRANSPARENCY
)
from configs.dataset_adapter import DatasetAdapter
from utils.export_las import export_reconstructed_pointclouds
from utils.export_floorplan_pdf import generate_measured_floorplan_pdf
from pipeline import (
    run_stage1_sensors,
    run_stage2_pointcloud,
    run_stage3_projections,
    run_stage4_heat_floorplan,
    run_stage5_openings,
    run_stage6_ifc_assembly,
    run_stage7_benchmark
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Image2BIM: Sensor-Guided End-to-End Indoor BIM Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scene_id", type=str, default=None,
                        help="Single scene identifier to process (e.g. 00000 or scene_00000). To run the full dataset, omit or use --all / --scene_id all")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Process the FULL dataset across all available scenes")
    parser.add_argument("--dataset", type=str, default="structured3d",
                        help="Dataset name (maps to dataset/<dataset>/ and results/<dataset>/)")
    parser.add_argument("--mode", type=str, default="dataset_sensors", help=argparse.SUPPRESS)
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Custom path to dataset directory (overrides default dataset/<dataset>)")
    parser.add_argument("--ckpt_path", type=str, default=str(DEFAULT_HEAT_CHECKPOINT),
                        help="Path to HEAT model checkpoint (.pth)")
    parser.add_argument("--openings_source", type=str, default="dataset",
                        choices=["dataset", "none"],
                        help="Source of doors/windows: 'dataset' (reads dataset CAD annotations) or 'none'")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for generated BIM and scan artifacts")
    parser.add_argument("--wall_height", type=float, default=DEFAULT_WALL_HEIGHT,
                        help="Default wall height in meters")
    parser.add_argument("--wall_thickness", type=float, default=DEFAULT_WALL_THICKNESS,
                        help="Standard wall thickness in meters")
    parser.add_argument("--slab_thickness", type=float, default=DEFAULT_SLAB_THICKNESS,
                        help="Floor and ceiling slab thickness in meters")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Point cloud backprojection pixel subsampling factor")
    return parser.parse_args()


def get_adapter(args, clean_id):
    cfg = DATASETS_CONFIG.get(args.dataset, {})
    default_dataset_dir = str(DATASET_DIR / args.dataset)
    raw_dir = args.raw_dir if args.raw_dir else cfg.get("raw_dir", default_dataset_dir)
    adapter = DatasetAdapter(
        raw_dir=raw_dir,
        norm_dict_path=cfg.get("norm_dict_path"),
        normals_dir=cfg.get("normals_dir")
    )
    return adapter


def run_single_scene(clean_id: str, args):
    """Execute the full 7-stage pipeline for a single scene."""
    scene_name = f"scene_{clean_id}"

    # Results folder: default is results/<dataset>/scene_<id> (e.g. results/structured3d/scene_00000)
    out_dir = args.out_dir if args.out_dir else str(RESULTS_DIR / args.dataset / scene_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"STARTING SCAN-TO-BIM PIPELINE: {scene_name.upper()} (DATASET: {args.dataset.upper()})")
    print(f"Sensor Mode:     {args.mode}")
    print(f"Openings Source: {args.openings_source}")
    print(f"Output:          {out_dir}")
    print("=" * 80)

    # Initialize Adapter
    adapter = get_adapter(args, clean_id)

    # Print Dataset Source Paths (Images, Sensors, Annotations)
    scene_paths = adapter.get_scene_source_paths(clean_id)
    print("\nDATASET SOURCE LOCATIONS:")
    print(f"  Scene Root:      {scene_paths.get('scene_path')}")
    print(f"  CAD Annotation:  {scene_paths.get('annotations')}")
    stations = adapter.get_scene_stations(clean_id)
    print(f"  Stations Count:  {len(stations)} ({', '.join(stations)})")
    print("  Per-Station Image & Sensor Source Files:")
    for st in stations:
        st_paths = adapter.get_station_paths(clean_id, st)
        print(f"    - Station [{st}]:")
        print(f"        RGB:    {st_paths.get('rgb')}")
        print(f"        Depth:  {st_paths.get('depth')}")
        print(f"        Normal: {st_paths.get('normal')}")
        print(f"        Camera: {st_paths.get('camera')}")
    print("-" * 80)

    # STAGE 1: Sensor Ingestion (Dataset Sensors)
    sensor_data = run_stage1_sensors(adapter, clean_id, out_dir, mode=args.mode)

    # STAGE 2: 3D Point Cloud Synthesis (Raw Scan Before Processing)
    pts, bounds, raw_colors = run_stage2_pointcloud(sensor_data, out_dir, clean_id, subsample=args.subsample)

    # STAGE 3: 2D Top-Down Density & Normal Map Projections
    rgb_heat, norm_dict = run_stage3_projections(pts, bounds, adapter, clean_id, out_dir, res=DEFAULT_IMAGE_SIZE)

    # STAGE 4: HEAT Neural Transformer 2D Floorplan Reconstruction
    pg_data = run_stage4_heat_floorplan(rgb_heat, args.ckpt_path, out_dir, clean_id, image_size=DEFAULT_IMAGE_SIZE)

    # Dynamic Floor Elevation and Wall Height Calculation
    z_floor = float(np.percentile(pts[:, 2], 1)) if pts is not None and len(pts) > 0 else 0.0
    wall_height = args.wall_height

    # Check CAD annotations for exact ground truth ceiling height
    anno_gt = adapter.get_ground_truth(clean_id)
    if anno_gt and "junctions" in anno_gt and len(anno_gt["junctions"]) > 0:
        j_coords = np.array([j["coordinate"] for j in anno_gt["junctions"]], dtype=float)
        z_floor_anno = float(np.min(j_coords[:, 2])) / 1000.0
        z_ceil_anno = float(np.max(j_coords[:, 2])) / 1000.0
        if z_ceil_anno - z_floor_anno > 0.5:
            wall_height = round(z_ceil_anno - z_floor_anno, 2)
            z_floor = z_floor_anno
            print(f"  Exact CAD Wall Height Detected: {wall_height:.2f}m (Floor Z: {z_floor:.2f}m)")
    else:
        # Fallback to point cloud bounds
        z_ceil_cloud = float(np.percentile(pts[:, 2], 99))
        if z_ceil_cloud - z_floor > 1.0:
            wall_height = round(z_ceil_cloud - z_floor, 2)
            print(f"  PointCloud Estimated Wall Height: {wall_height:.2f}m (Floor Z: {z_floor:.2f}m)")

    # EXPORT AFTER-PROCESSING RECONSTRUCTED POINT CLOUDS
    print("\n" + "=" * 80)
    print("EXPORTING AFTER-PROCESSING RECONSTRUCTED POINT CLOUDS (LAS)")
    print("=" * 80)
    recon_clouds = export_reconstructed_pointclouds(
        pg_data, norm_dict, raw_pts=pts, raw_colors=raw_colors,
        out_dir=out_dir, scene_id=clean_id, wall_height=wall_height
    )

    # STAGE 5: Door & Window Extraction (Dataset Annotations)
    if args.openings_source != "none":
        openings = run_stage5_openings(
            adapter, clean_id, out_dir, pg_data, norm_dict,
            source=args.openings_source
        )
    else:
        openings = []
        print("\nSTAGE 5: SKIPPED (openings_source=none)")

    # Extract exact CAD geometry if present in dataset
    cad_geom = adapter.get_cad_geometry(clean_id)

    # STAGE 6: Production-Grade 3D IFC4 BIM Model Assembly
    out_ifc = run_stage6_ifc_assembly(
        pg_data, norm_dict, openings, out_dir, clean_id,
        wall_thickness=args.wall_thickness,
        slab_thickness=args.slab_thickness,
        wall_height=wall_height,
        ceiling_transparency=DEFAULT_CEILING_TRANSPARENCY,
        z_floor=z_floor,
        cad_geom=cad_geom
    )

    # STAGE 7: Benchmark Performance Against CAD Ground Truth
    bench_metrics = run_stage7_benchmark(adapter, clean_id, pg_data, norm_dict, openings, out_dir)

    # EXPORT ARCHITECTURAL 2D MEASURED FLOORPLAN DRAWING SHEET (PDF + PNG)
    print("\n" + "=" * 80)
    print("EXPORTING ARCHITECTURAL 2D MEASURED FLOORPLAN DRAWING SHEET (PDF)")
    print("=" * 80)
    pdf_path = os.path.join(out_dir, f"{clean_id}_floorplan_sheet.pdf")
    try:
        generate_measured_floorplan_pdf(
            pg_data=pg_data,
            norm_dict=norm_dict,
            openings=openings,
            out_pdf_path=pdf_path,
            scene_id=clean_id,
            drawing_title="Ground Floor Plan",
            cad_geom=cad_geom
        )
        print(f"  Architectural Measured 2D PDF:     {pdf_path}")
    except Exception as e:
        print(f"  [WARNING] Could not export PDF drawing sheet: {e}")
        pdf_path = None

    raw_las_path = os.path.join(out_dir, f"{clean_id}_raw_scan.las")
    folder_prefix = scene_name
    viz_path = os.path.join(out_dir, f"{folder_prefix}_images", f"{clean_id}_predicted_floorplan.png")
    npy_path = os.path.join(out_dir, f"{clean_id}_floorplan.npy")

    print("\n" + "#" * 80)
    print("SCAN-TO-BIM PIPELINE EXECUTION COMPLETE!")
    print(f"  [1] Final 3D BIM Model:            {out_ifc}")
    print(f"  [2] Architectural 2D PDF Drawing:  {pdf_path}")
    print(f"  [3] Raw Scan (Before Processing):  {raw_las_path}")
    print(f"  [4] Reconstructed Walls Cloud:     {recon_clouds['walls_las']}")
    print(f"  [5] Corner Junctions Cloud:        {recon_clouds['corners_las']}")
    print(f"  [6] Multi-Class Overlay Cloud:     {recon_clouds['overlay_las']}")
    print(f"  [7] 2D CAD Floorplan:              {viz_path}")
    print(f"  [8] 2D Vector Graph (.npy):        {npy_path}")
    print("#" * 80 + "\n")

    return {
        "scene_id": clean_id,
        "ifc_path": out_ifc,
        "pdf_path": pdf_path,
        "raw_las": raw_las_path,
        "overlay_las": recon_clouds["overlay_las"],
        "npy_path": npy_path,
        "wall_height": wall_height,
        "z_floor": z_floor,
        "metrics": bench_metrics
    }


def main():
    import time
    args = parse_args()

    is_full_dataset = args.all or (args.scene_id is not None and args.scene_id.lower() == "all")

    # Batch mode: run all scenes of the full dataset
    if is_full_dataset:
        cfg = DATASETS_CONFIG.get(args.dataset, {})
        default_dataset_dir = Path(DATASET_DIR) / args.dataset
        raw_dir = Path(args.raw_dir) if args.raw_dir else Path(cfg.get("raw_dir", default_dataset_dir))

        if not raw_dir.exists():
            print(f"[ERROR] Dataset directory does not exist: {raw_dir}")
            sys.exit(1)

        scenes = sorted([
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and (d.name.startswith("scene_") or d.name.isdigit())
        ])

        if not scenes:
            print(f"[ERROR] No scenes found in {raw_dir}")
            sys.exit(1)

        print("\n" + "=" * 95)
        print(f"BATCH SCAN-TO-BIM PIPELINE: {args.dataset.upper()} ({len(scenes)} scenes)")
        print(f"Mode:     {args.mode}")
        print(f"Dataset:  {raw_dir}")
        print(f"Results:  {RESULTS_DIR / args.dataset}")
        print("=" * 95 + "\n")

        summary_records = []
        total_start = time.time()

        for idx, sc in enumerate(scenes, 1):
            clean_id = sc.replace("scene_", "")
            print(f"\n>>>>> [{idx}/{len(scenes)}] SCENE: {sc.upper()} <<<<<")
            t0 = time.time()
            try:
                res = run_single_scene(clean_id, args)
                status = "SUCCESS"
                ifc_p = Path(res["ifc_path"])
                ifc_kb = (ifc_p.stat().st_size / 1024) if ifc_p.exists() else 0.0
                ifc_str = str(ifc_p)
                pdf_str = str(res.get("pdf_path", "N/A"))
            except Exception as e:
                print(f"[ERROR] Failed on scene {sc}: {e}")
                status = "FAILED"
                ifc_kb = 0.0
                ifc_str = "N/A"
                pdf_str = "N/A"
            elapsed = time.time() - t0
            summary_records.append({
                "scene": sc,
                "status": status,
                "time_sec": round(elapsed, 1),
                "ifc_kb": round(ifc_kb, 1),
                "ifc_path": ifc_str,
                "pdf_path": pdf_str
            })

        total_time = time.time() - total_start
        print("\n" + "=" * 115)
        print("BATCH RECONSTRUCTION SUMMARY TABLE")
        print("=" * 115)
        print(f"{'#':<4} | {'Scene ID':<15} | {'Status':<10} | {'IFC Size':<12} | {'Time (s)':<10} | {'2D PDF Drawing'}")
        print("-" * 115)
        success_count = sum(1 for r in summary_records if r["status"] == "SUCCESS")
        for idx, r in enumerate(summary_records, 1):
            print(f"{idx:<4} | {r['scene']:<15} | {r['status']:<10} | {r['ifc_kb']:>8.1f} KB | {r['time_sec']:>8.1f} s | {r['pdf_path']}")
        print("-" * 115)
        print(f"Finished: {success_count}/{len(scenes)} Scenes Generated in {total_time/60:.2f} minutes.")
        print("=" * 95 + "\n")

    else:
        # Single scene mode (defaults to 00000 if not specified)
        target_scene = args.scene_id if args.scene_id else "00000"
        clean_id = f"{int(target_scene):05d}" if target_scene.isdigit() else target_scene.replace("scene_", "")
        run_single_scene(clean_id, args)


if __name__ == "__main__":
    main()

