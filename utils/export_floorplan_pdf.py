"""
Architectural 2D Measured Floorplan PDF Generator.
Produces high-precision vector CAD floor plan drawing sheets (PDF + PNG)
matching professional architectural standards (Title Block, Legend, North Arrow,
Dual-Tier Dimension Chains in mm, Scale Bar, Wall/Door/Window symbols).
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon, MultiPoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.planar_graph import get_regions_from_pg


def generate_measured_floorplan_pdf(
    pg_data: Dict[str, np.ndarray],
    norm_dict: Dict[str, Any],
    openings: List[Dict[str, Any]],
    out_pdf_path: str,
    scene_id: str,
    drawing_title: str = "Ground Floor Plan",
    cad_geom: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an architectural 2D floorplan drawing sheet (PDF & PNG) with exact measurements.
    If cad_geom is provided (e.g. from Structured3D), renders 100% exact CAD geometry.
    """
    min_c = np.array(norm_dict["min_coords"], dtype=float)
    max_c = np.array(norm_dict["max_coords"], dtype=float)
    x_span = max_c[0] - min_c[0]
    y_span = max_c[1] - min_c[1]

    use_cad = bool(cad_geom and cad_geom.get("has_cad"))

    if use_cad:
        cad_rooms = cad_geom.get("rooms", [])
        outwall = cad_geom.get("outwall", {})
        ow_coords = outwall.get("coords")
        ow_mm = ow_coords * 1000.0 if ow_coords is not None else None

        rooms_mm = []
        total_area_m2 = 0.0
        for rm in cad_rooms:
            r_mm = rm["coords"] * 1000.0
            rooms_mm.append({
                "coords_mm": r_mm,
                "name": rm["name"],
                "area_m2": rm["area_m2"],
                "centroid_mm": (rm["centroid"][0] * 1000.0, rm["centroid"][1] * 1000.0)
            })
            total_area_m2 += rm["area_m2"]

        all_x = []
        all_y = []
        if ow_mm is not None:
            all_x.extend(ow_mm[:, 0])
            all_y.extend(ow_mm[:, 1])
        else:
            for rm in rooms_mm:
                all_x.extend(rm["coords_mm"][:, 0])
                all_y.extend(rm["coords_mm"][:, 1])
    else:
        # Convert corners from normalized 256x256 pixel grid to physical millimeters
        c_px = pg_data["corners"]
        c_mm = np.zeros_like(c_px, dtype=float)
        c_mm[:, 0] = c_px[:, 0] / 256.0 * x_span + min_c[0]
        c_mm[:, 1] = c_px[:, 1] / 256.0 * y_span + min_c[1]

        # Extract topological room polygons
        raw_regions = get_regions_from_pg(pg_data, corner_sorted=True)
        rooms_mm = []
        total_area_m2 = 0.0

        for r_px in raw_regions:
            if len(r_px) >= 3:
                r_mm = np.zeros_like(r_px, dtype=float)
                r_mm[:, 0] = r_px[:, 0] / 256.0 * x_span + min_c[0]
                r_mm[:, 1] = r_px[:, 1] / 256.0 * y_span + min_c[1]
                poly = Polygon(r_mm / 1000.0)  # in meters
                if poly.is_valid and poly.area > 0.5:
                    rooms_mm.append({
                        "coords_mm": r_mm,
                        "poly_m": poly,
                        "name": f"ROOM {len(rooms_mm) + 1}",
                        "area_m2": poly.area,
                        "centroid_mm": (poly.centroid.x * 1000.0, poly.centroid.y * 1000.0)
                    })
                    total_area_m2 += poly.area

        # Fallback to convex hull if no closed cycles found
        if not rooms_mm and len(c_mm) >= 3:
            hull = MultiPoint(c_mm / 1000.0).convex_hull
            if hull.geom_type == "Polygon":
                h_mm = np.array(hull.exterior.coords) * 1000.0
                rooms_mm.append({
                    "coords_mm": h_mm,
                    "poly_m": hull,
                    "name": "ROOM",
                    "area_m2": hull.area,
                    "centroid_mm": (hull.centroid.x * 1000.0, hull.centroid.y * 1000.0)
                })
                total_area_m2 = hull.area

        ow_mm = None
        all_x = []
        all_y = []
        for rm in rooms_mm:
            all_x.extend(rm["coords_mm"][:, 0])
            all_y.extend(rm["coords_mm"][:, 1])
        if not all_x and len(c_mm) > 0:
            all_x = list(c_mm[:, 0])
            all_y = list(c_mm[:, 1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    dx = max_x - min_x
    dy = max_y - min_y

    # Calculate scale factor for title block
    max_extent_m = max(dx, dy) / 1000.0
    scale_label = "1:50" if max_extent_m <= 8.5 else "1:100"

    # Create Matplotlib Figure (A4 Landscape aspect ratio: 11.69 x 8.27 inches)
    fig = plt.figure(figsize=(11.69, 8.27), dpi=300, facecolor='white')

    # Border axes covering entire sheet [0, 1] x [0, 1]
    border_ax = fig.add_axes([0, 0, 1, 1], frame_on=False)
    border_ax.set_xlim(0, 1)
    border_ax.set_ylim(0, 1)
    border_ax.axis('off')

    # Sheet outer frame
    sheet_rect = patches.Rectangle(
        (0.035, 0.035), 0.93, 0.93,
        linewidth=1.2, edgecolor='#111111', facecolor='none'
    )
    border_ax.add_patch(sheet_rect)

    # Main floorplan drawing axes
    main_ax = fig.add_axes([0.13, 0.16, 0.57, 0.72])
    main_ax.set_aspect('equal')
    main_ax.axis('off')

    # 1. DRAW WALL POCHE & ROOM SPACES
    if use_cad and ow_mm is not None:
        # Fill building envelope with architectural wall poche
        main_ax.add_patch(MplPolygon(ow_mm, closed=True, facecolor='#565656', edgecolor='#111111', lw=2.0, zorder=1))
        # Cutout rooms with crisp white floors and wall outlines
        for rm in rooms_mm:
            main_ax.add_patch(MplPolygon(rm["coords_mm"], closed=True, facecolor='#FFFFFF', edgecolor='#111111', lw=1.8, zorder=2))
            cx, cy = rm["centroid_mm"]
            fsize = 9.5 if rm["area_m2"] >= 6.0 else (8.0 if rm["area_m2"] >= 2.0 else 7.0)
            main_ax.text(
                cx, cy, f"{rm['name']}\n{rm['area_m2']:.1f} m²",
                color='#111111', fontsize=fsize, fontweight='bold',
                ha='center', va='center', zorder=4, fontfamily='sans-serif'
            )
    else:
        for idx, rm in enumerate(rooms_mm):
            r_pts = rm["coords_mm"]
            room_poly = MplPolygon(r_pts, closed=True, facecolor='#565656', edgecolor='#111111', linewidth=2.0, zorder=2)
            main_ax.add_patch(room_poly)
            cx, cy = rm["centroid_mm"]
            fsize = 11 if rm["area_m2"] >= 6.0 else (9 if rm["area_m2"] >= 2.0 else 7.5)
            main_ax.text(
                cx, cy, rm["name"], color='white', fontsize=fsize, fontweight='bold',
                ha='center', va='center', zorder=4, fontfamily='sans-serif'
            )
        for e in pg_data.get("edges", []):
            p1 = c_mm[e[0]]
            p2 = c_mm[e[1]]
            main_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#111111', linewidth=2.5, solid_capstyle='butt', zorder=3)

    # 2. DRAW OPENINGS (Doors and Windows)
    for op in openings:
        op_type = op.get("type", "door").lower()
        center_m = op.get("center", [0, 0, 0])
        cx_mm = center_m[0] * 1000.0
        cy_mm = center_m[1] * 1000.0
        w_mm = float(op.get("width", 0.8)) * 1000.0
        thick_mm = float(op.get("thickness", 0.20)) * 1000.0
        ang_rad = float(op.get("angle_rad", 0.0))
        ang_deg = float(op.get("angle_deg", math.degrees(ang_rad)))
        swing_sign = float(op.get("swing_sign", 1.0))

        u = np.array([math.cos(ang_rad), math.sin(ang_rad)])
        n = np.array([-math.sin(ang_rad), math.cos(ang_rad)])

        # Corner for opening rectangle
        c0_x = cx_mm - (w_mm / 2.0) * u[0] - (thick_mm / 2.0) * n[0]
        c0_y = cy_mm - (w_mm / 2.0) * u[1] - (thick_mm / 2.0) * n[1]

        # White cutout over wall poche
        rect = Rectangle((c0_x, c0_y), w_mm, thick_mm, angle=ang_deg,
                         facecolor='white', edgecolor='none', zorder=5)
        main_ax.add_patch(rect)

        # Opening jamb lines across wall thickness
        p_jamb1_a = np.array([c0_x, c0_y])
        p_jamb1_b = p_jamb1_a + thick_mm * n
        p_jamb2_a = np.array([c0_x, c0_y]) + w_mm * u
        p_jamb2_b = p_jamb2_a + thick_mm * n

        main_ax.plot([p_jamb1_a[0], p_jamb1_b[0]], [p_jamb1_a[1], p_jamb1_b[1]], color='#111111', lw=1.2, zorder=6)
        main_ax.plot([p_jamb2_a[0], p_jamb2_b[0]], [p_jamb2_a[1], p_jamb2_b[1]], color='#111111', lw=1.2, zorder=6)

        if op_type == "window":
            # Window frame centerline
            mid_a = (p_jamb1_a + p_jamb1_b) / 2.0
            mid_b = (p_jamb2_a + p_jamb2_b) / 2.0
            main_ax.plot([mid_a[0], mid_b[0]], [mid_a[1], mid_b[1]], color='#111111', lw=1.0, zorder=7)
            # Glazing frame edges
            main_ax.plot([p_jamb1_a[0], p_jamb2_a[0]], [p_jamb1_a[1], p_jamb2_a[1]], color='#666666', lw=0.6, zorder=7)
            main_ax.plot([p_jamb1_b[0], p_jamb2_b[0]], [p_jamb1_b[1], p_jamb2_b[1]], color='#666666', lw=0.6, zorder=7)

        elif op_type == "door":
            # Inward door swing: leaf & smooth quarter-circle arc
            n_sw = n * swing_sign
            hinge = (p_jamb1_a + p_jamb1_b) / 2.0
            leaf_end = hinge + w_mm * n_sw

            # Door leaf
            main_ax.plot([hinge[0], leaf_end[0]], [hinge[1], leaf_end[1]], color='#111111', lw=1.6, zorder=7)

            # Smooth quarter-circle swing arc
            t_vals = np.linspace(0, math.pi / 2.0, 25)
            arc_pts = np.array([hinge + w_mm * (math.cos(tv) * u + math.sin(tv) * n_sw) for tv in t_vals])
            main_ax.plot(arc_pts[:, 0], arc_pts[:, 1], color='#111111', lw=1.0, ls='-', zorder=7)

    # 4. ARCHITECTURAL DUAL-TIER DIMENSION CHAINS (Left & Bottom)
    # Helper functions for drawing dimension lines with witness lines and arrows
    def draw_dim_h(x1, x2, y_dim, y_witness_start, text, offset_text=70.0):
        gap = 80.0
        ext = 140.0
        # Witness lines extending down from wall corner to dimension line
        main_ax.plot([x1, x1], [y_witness_start - gap, y_dim - ext], color='#111111', linewidth=0.6, zorder=7)
        main_ax.plot([x2, x2], [y_witness_start - gap, y_dim - ext], color='#111111', linewidth=0.6, zorder=7)
        # Dimension line
        main_ax.plot([x1, x2], [y_dim, y_dim], color='#111111', linewidth=0.75, zorder=7)
        # Inward-pointing arrows at extremities
        arr_len = min(abs(x2 - x1) * 0.14, 110.0)
        main_ax.annotate('', xy=(x1, y_dim), xytext=(x1 + arr_len, y_dim),
                         arrowprops=dict(arrowstyle='->', color='#111111', lw=0.75), zorder=8)
        main_ax.annotate('', xy=(x2, y_dim), xytext=(x2 - arr_len, y_dim),
                         arrowprops=dict(arrowstyle='->', color='#111111', lw=0.75), zorder=8)
        # Dimension value in mm
        main_ax.text((x1 + x2) / 2.0, y_dim + offset_text, str(text),
                     color='#111111', fontsize=8.5, ha='center', va='bottom',
                     fontfamily='sans-serif', zorder=9)

    def draw_dim_v(y1, y2, x_dim, x_witness_start, text, offset_text=70.0):
        gap = 80.0
        ext = 140.0
        # Witness lines extending left from wall corner to dimension line
        main_ax.plot([x_witness_start - gap, x_dim - ext], [y1, y1], color='#111111', linewidth=0.6, zorder=7)
        main_ax.plot([x_witness_start - gap, x_dim - ext], [y2, y2], color='#111111', linewidth=0.6, zorder=7)
        # Dimension line
        main_ax.plot([x_dim, x_dim], [y1, y2], color='#111111', linewidth=0.75, zorder=7)
        # Inward-pointing arrows
        arr_len = min(abs(y2 - y1) * 0.14, 110.0)
        main_ax.annotate('', xy=(x_dim, y1), xytext=(x_dim, y1 + arr_len),
                         arrowprops=dict(arrowstyle='->', color='#111111', lw=0.75), zorder=8)
        main_ax.annotate('', xy=(x_dim, y2), xytext=(x_dim, y2 - arr_len),
                         arrowprops=dict(arrowstyle='->', color='#111111', lw=0.75), zorder=8)
        # Dimension value in mm (rotated 90 degrees)
        main_ax.text(x_dim - offset_text, (y1 + y2) / 2.0, str(text),
                     color='#111111', fontsize=8.5, ha='right', va='center',
                     rotation=90, fontfamily='sans-serif', zorder=9)

    # Cluster corner coordinates along X and Y to define architectural projection steps
    def cluster_coords(coords, tol=140.0):
        sorted_c = sorted(coords)
        clusters = []
        for c in sorted_c:
            if not clusters:
                clusters.append([c])
            elif abs(c - np.mean(clusters[-1])) <= tol:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [float(np.mean(cl)) for cl in clusters]

    x_keys = cluster_coords(all_x, tol=140.0)
    y_keys = cluster_coords(all_y, tol=140.0)

    # Filter out minuscule step artifacts (< 350mm)
    x_steps = [x_keys[0]]
    for xk in x_keys[1:]:
        if xk - x_steps[-1] >= 350.0:
            x_steps.append(xk)
    if x_keys[-1] - x_steps[-1] > 0 and x_steps[-1] != x_keys[-1]:
        x_steps[-1] = x_keys[-1]

    y_steps = [y_keys[0]]
    for yk in y_keys[1:]:
        if yk - y_steps[-1] >= 350.0:
            y_steps.append(yk)
    if y_keys[-1] - y_steps[-1] > 0 and y_steps[-1] != y_keys[-1]:
        y_steps[-1] = y_keys[-1]

    # Find corner witness line start coordinates
    corner_pts_mm = np.column_stack([all_x, all_y]) if len(all_x) > 0 else np.zeros((0, 2))

    # For each x_step, find minimum Y among corners near that x
    def get_y_anchor(x_val):
        near_pts = [pt[1] for pt in corner_pts_mm if abs(pt[0] - x_val) < 250.0]
        return min(near_pts) if near_pts else min_y

    # For each y_step, find minimum X among corners near that y
    def get_x_anchor(y_val):
        near_pts = [pt[0] for pt in corner_pts_mm if abs(pt[1] - y_val) < 250.0]
        return min(near_pts) if near_pts else min_x

    dim_margin = max(dx, dy) * 0.08
    y_dim_tier1 = min_y - dim_margin
    y_dim_tier2 = min_y - dim_margin * 2.2

    x_dim_tier1 = min_x - dim_margin
    x_dim_tier2 = min_x - dim_margin * 2.2

    # Bottom Chain: Tier 1 (Segment lengths)
    for i in range(len(x_steps) - 1):
        x1, x2 = x_steps[i], x_steps[i+1]
        seg_len = int(round(x2 - x1))
        y_w = min(get_y_anchor(x1), get_y_anchor(x2))
        draw_dim_h(x1, x2, y_dim_tier1, y_w, seg_len)

    # Bottom Chain: Tier 2 (Overall length)
    total_w = int(round(max_x - min_x))
    draw_dim_h(min_x, max_x, y_dim_tier2, min_y, total_w)

    # Left Chain: Tier 1 (Segment lengths)
    for i in range(len(y_steps) - 1):
        y1, y2 = y_steps[i], y_steps[i+1]
        seg_len = int(round(y2 - y1))
        x_w = min(get_x_anchor(y1), get_x_anchor(y2))
        draw_dim_v(y1, y2, x_dim_tier1, x_w, seg_len)

    # Left Chain: Tier 2 (Overall height)
    total_h = int(round(max_y - min_y))
    draw_dim_v(min_y, max_y, x_dim_tier2, min_x, total_h)

    # Set viewport limits with generous margin for dimension strings
    pad_left = dim_margin * 3.5
    pad_right = dim_margin * 1.0
    pad_bottom = dim_margin * 3.5
    pad_top = dim_margin * 1.5

    main_ax.set_xlim(min_x - pad_left, max_x + pad_right)
    main_ax.set_ylim(min_y - pad_bottom, max_y + pad_top)

    # 5. NORTH ARROW (Top Right)
    arrow_x = 0.65
    arrow_y = 0.88
    border_ax.text(arrow_x, arrow_y + 0.04, 'N', fontsize=12, fontweight='bold', ha='center', va='center')
    border_ax.annotate('', xy=(arrow_x, arrow_y + 0.035), xytext=(arrow_x, arrow_y - 0.035),
                       arrowprops=dict(arrowstyle='->', color='#111111', lw=1.2))

    # 6. LEGEND (Top Right)
    leg_x = 0.70
    leg_y = 0.86
    border_ax.text(leg_x, leg_y, 'LEGEND', fontsize=11, fontweight='bold', ha='left', va='center')

    legend_items = [
        ("WALL", "rect_fill"),
        ("DOOR", "door_icon"),
        ("WINDOW", "window_icon"),
        ("COLUMN", "column_icon"),
        ("DIMENSION (mm)", "dim_icon")
    ]

    row_y = leg_y - 0.032
    for label, icon_type in legend_items:
        ix = leg_x
        iy = row_y
        if icon_type == "rect_fill":
            border_ax.add_patch(Rectangle((ix, iy - 0.008), 0.022, 0.016, facecolor='#565656', edgecolor='#111111', lw=0.8))
        elif icon_type == "door_icon":
            border_ax.plot([ix, ix], [iy - 0.008, iy + 0.008], color='#111111', lw=1.0)
            arc = Arc((ix, iy - 0.008), width=0.032, height=0.032, angle=0, theta1=0, theta2=90, color='#111111', lw=0.8)
            border_ax.add_patch(arc)
        elif icon_type == "window_icon":
            border_ax.add_patch(Rectangle((ix, iy - 0.007), 0.022, 0.014, facecolor='white', edgecolor='#111111', lw=0.8))
            border_ax.plot([ix, ix + 0.022], [iy, iy], color='#111111', lw=0.6)
        elif icon_type == "column_icon":
            circle = patches.Circle((ix + 0.011, iy), 0.007, facecolor='none', edgecolor='#111111', lw=0.8)
            border_ax.add_patch(circle)
        elif icon_type == "dim_icon":
            border_ax.plot([ix, ix + 0.022], [iy, iy], color='#111111', lw=0.8)
            border_ax.plot([ix, ix], [iy - 0.006, iy + 0.006], color='#111111', lw=0.8)
            border_ax.plot([ix + 0.022, ix + 0.022], [iy - 0.006, iy + 0.006], color='#111111', lw=0.8)

        border_ax.text(ix + 0.032, iy, label, fontsize=9, ha='left', va='center', fontfamily='sans-serif')
        row_y -= 0.028

    # 7. TITLE BLOCK / METADATA TABLE (Bottom Right)
    tbl_x = 0.70
    tbl_y = 0.28
    tbl_w = 0.23
    tbl_h = 0.12
    row_h = tbl_h / 4.0

    border_ax.add_patch(Rectangle((tbl_x, tbl_y), tbl_w, tbl_h, facecolor='none', edgecolor='#111111', lw=1.0))
    col1_w = 0.09
    border_ax.plot([tbl_x + col1_w, tbl_x + col1_w], [tbl_y, tbl_y + tbl_h], color='#111111', lw=0.8)

    table_data = [
        ("DRAWING", drawing_title),
        ("SCALE", scale_label),
        ("UNITS", "mm"),
        ("AREA", f"{total_area_m2:.2f} m2")
    ]

    for r_i, (k, v) in enumerate(table_data):
        curr_y = tbl_y + tbl_h - (r_i + 1) * row_h
        if r_i < 3:
            border_ax.plot([tbl_x, tbl_x + tbl_w], [curr_y, curr_y], color='#111111', lw=0.8)
        border_ax.text(tbl_x + 0.008, curr_y + row_h / 2.0, k, fontsize=8.5, ha='left', va='center', fontfamily='sans-serif')
        border_ax.text(tbl_x + col1_w + 0.008, curr_y + row_h / 2.0, v, fontsize=8.5, ha='left', va='center', fontfamily='sans-serif')

    # 8. GRAPHIC SCALE BAR (Bottom Left)
    sb_x = 0.15
    sb_y = 0.075
    sb_seg_w = 0.07
    num_segs = 4
    seg_h = 0.012

    for s_i in range(num_segs):
        fc = '#111111' if s_i % 2 == 0 else 'white'
        border_ax.add_patch(Rectangle((sb_x + s_i * sb_seg_w, sb_y), sb_seg_w, seg_h,
                                      facecolor=fc, edgecolor='#111111', lw=0.8))
        border_ax.text(sb_x + s_i * sb_seg_w, sb_y - 0.012, str(s_i), fontsize=8, ha='center', va='top')
    border_ax.text(sb_x + num_segs * sb_seg_w, sb_y - 0.012, str(num_segs), fontsize=8, ha='center', va='top')
    border_ax.text(sb_x + num_segs * sb_seg_w + 0.01, sb_y + seg_h / 2.0, 'metres', fontsize=8, ha='left', va='center')

    # Save vector PDF and companion high-resolution PNG
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    fig.savefig(out_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)

    out_png_path = str(Path(out_pdf_path).with_suffix('.png'))
    fig.savefig(out_png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    return out_pdf_path

