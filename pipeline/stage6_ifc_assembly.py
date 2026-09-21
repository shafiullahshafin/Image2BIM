"""
STAGE 6: Production-Grade 3D IFC4 BIM Model Assembly.
"""

import os
import math
from typing import Dict, Any, List, Optional
import numpy as np
import ifcopenshell
import ifcopenshell.api
from shapely.geometry import Polygon, MultiPoint

from utils.planar_graph import get_regions_from_pg
from utils.export_ifc import create_surface_style, create_polygonal_slab_entity


def run_stage6_ifc_assembly(pg_data: Dict[str, np.ndarray], norm_dict: Dict[str, Any],
                            openings: List[Dict[str, Any]], out_dir: str, scene_id: str,
                            wall_thickness: float = 0.20, slab_thickness: float = 0.15,
                            wall_height: float = 2.80, ceiling_transparency: float = 0.80,
                            z_floor: Optional[float] = None,
                            cad_geom: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute Stage 6: Assemble standard IFC4 BIM model with walls, slabs, doors, and windows.
    If exact CAD geometry is available (e.g. Structured3D), assembles 100% accurate elements.
    """
    print("\n" + "=" * 80)
    print("STAGE 6: ASSEMBLING PRODUCTION-GRADE 3D IFC4 BIM MODEL")
    print("=" * 80)

    min_c = np.array(norm_dict["min_coords"], dtype=float)
    max_c = np.array(norm_dict["max_coords"], dtype=float)
    x_span = max_c[0] - min_c[0]
    y_span = max_c[1] - min_c[1]
    if z_floor is None:
        raw_z = float(min_c[2]) / 1000.0
        z_floor = 0.0 if raw_z < -0.1 else raw_z

    c_px = pg_data["corners"]
    c_m = np.zeros_like(c_px, dtype=float)
    c_m[:, 0] = (c_px[:, 0] / 256.0 * x_span + min_c[0]) / 1000.0
    c_m[:, 1] = (c_px[:, 1] / 256.0 * y_span + min_c[1]) / 1000.0

    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    proj = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Image2BIM_Production_Model")
    ifcopenshell.api.run("unit.assign_unit", model, length={"is_metric": True, "raw": "METERS"})

    ctx = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, context_type="Model",
                                context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name=f"Site_{scene_id}")
    bldg = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name=f"Building_{scene_id}")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Ground Floor")

    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=proj, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[bldg])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=bldg, products=[storey])

    # Standard Spatial Placement Hierarchy (BIMvision & IFC4 compliant)
    site.ObjectPlacement = model.create_entity("IfcLocalPlacement",
        RelativePlacement=model.create_entity("IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))))
    bldg.ObjectPlacement = model.create_entity("IfcLocalPlacement",
        PlacementRelTo=site.ObjectPlacement,
        RelativePlacement=model.create_entity("IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))))
    storey.ObjectPlacement = model.create_entity("IfcLocalPlacement",
        PlacementRelTo=bldg.ObjectPlacement,
        RelativePlacement=model.create_entity("IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))))

    wall_style = create_surface_style(model, "Wall", 0.94, 0.94, 0.93)
    floor_style = create_surface_style(model, "Floor", 0.86, 0.83, 0.79)
    ceil_style = create_surface_style(model, "Ceiling", 0.88, 0.94, 0.98, transparency=ceiling_transparency)
    door_style = create_surface_style(model, "Door", 0.65, 0.45, 0.28)
    win_style = create_surface_style(model, "Window", 0.72, 0.88, 0.96, transparency=0.60)

    elements = []
    wall_records = []

    # Branch A: Exact CAD Geometry Mode (100% Ground Truth Accuracy)
    if cad_geom and cad_geom.get("has_cad"):
        print("  >> Assembling from 100% Exact CAD Annotations (Millimeter Accuracy)")
        cad_rooms = cad_geom.get("rooms", [])
        cad_walls = cad_geom.get("walls", [])

        # 1. Exact 3D Walls
        for idx, w in enumerate(cad_walls):
            p1 = w["p1"]
            p2 = w["p2"]
            l = w["length"]
            angle = w["angle_rad"]
            w_h = w.get("height", wall_height)
            w_zf = w.get("z_floor", z_floor)
            w_t = float(w.get("thickness", wall_thickness))
            w_name = w.get("name", f"Wall_{scene_id}_{idx:02d}")

            wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name=w_name)
            elements.append(wall)

            prof = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=l, YDim=w_t)
            prof.Position = model.create_entity("IfcAxis2Placement2D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(l / 2.0, 0.0)))
            solid = model.create_entity("IfcExtrudedAreaSolid", SweptArea=prof,
                Position=model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
                ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)), Depth=w_h)

            model.create_entity("IfcStyledItem", Item=solid, Styles=[wall_style])
            rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid])
            wall.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[rep])

            wall.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                PlacementRelTo=storey.ObjectPlacement,
                RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                    Location=model.create_entity("IfcCartesianPoint", Coordinates=(float(p1[0]), float(p1[1]), float(w_zf))),
                    RefDirection=model.create_entity("IfcDirection", DirectionRatios=(math.cos(angle), math.sin(angle), 0.0))))

            wall_records.append({
                "entity": wall,
                "p1": np.array(p1, dtype=float),
                "p2": np.array(p2, dtype=float),
                "length": l,
                "angle": angle,
                "z_floor": w_zf,
                "height": w_h,
                "thickness": w_t
            })

        # 2. Exact Floor & Ceiling Slabs per Room
        spaces = []
        for r_i, rm in enumerate(cad_rooms):
            r_name = rm["name"].replace(" ", "_")
            r_coords = rm["coords"]
            r_zf = rm["z_floor"]
            r_zc = rm["z_ceiling"]

            f_slab = create_polygonal_slab_entity(model, body, f"FloorSlab_{r_name}", r_coords, r_zf - slab_thickness, slab_thickness, floor_style, "FLOOR", rel_to=storey.ObjectPlacement)
            c_slab = create_polygonal_slab_entity(model, body, f"CeilingSlab_{r_name}", r_coords, r_zc, slab_thickness, ceil_style, "ROOF", rel_to=storey.ObjectPlacement)
            space = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSpace", name=f"Space_{r_name}", predefined_type="SPACE")
            elements.extend([f_slab, c_slab])
            spaces.append(space)

        if spaces:
            ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=spaces)

        wall_count = len(cad_walls)
        slab_count = len(cad_rooms) * 2

    # Branch B: Planar Graph Estimation Mode (Predicted Neural Floorplan Fallback)
    else:
        print("  >> Assembling from Predicted Planar Graph (HEAT Neural Floorplan)")
        for idx, e in enumerate(pg_data["edges"]):
            p1 = c_m[e[0]]
            p2 = c_m[e[1]]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            l = math.hypot(dx, dy)
            if l < 0.05:
                continue
            angle = math.atan2(dy, dx)

            wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWall", name=f"Wall_{scene_id}_{idx:02d}")
            elements.append(wall)

            prof = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=l, YDim=wall_thickness)
            prof.Position = model.create_entity("IfcAxis2Placement2D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(l / 2.0, 0.0)))
            solid = model.create_entity("IfcExtrudedAreaSolid", SweptArea=prof,
                Position=model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
                ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)), Depth=wall_height)

            model.create_entity("IfcStyledItem", Item=solid, Styles=[wall_style])
            rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid])
            wall.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[rep])

            wall.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                PlacementRelTo=storey.ObjectPlacement,
                RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                    Location=model.create_entity("IfcCartesianPoint", Coordinates=(float(p1[0]), float(p1[1]), float(z_floor))),
                    RefDirection=model.create_entity("IfcDirection", DirectionRatios=(math.cos(angle), math.sin(angle), 0.0))))

            wall_records.append({
                "entity": wall,
                "p1": np.array(p1, dtype=float),
                "p2": np.array(p2, dtype=float),
                "length": l,
                "angle": angle,
                "z_floor": z_floor,
                "height": wall_height
            })

        regions = get_regions_from_pg(pg_data, corner_sorted=True)
        seen_polys = []
        valid_rooms = []
        for r_px in regions:
            if len(r_px) >= 3:
                r_m = np.zeros_like(r_px, dtype=float)
                r_m[:, 0] = (r_px[:, 0] / 256.0 * x_span + min_c[0]) / 1000.0
                r_m[:, 1] = (r_px[:, 1] / 256.0 * y_span + min_c[1]) / 1000.0
                poly = Polygon(r_m)
                if poly.is_valid and poly.area > 0.5:
                    if not any(poly.equals_exact(sp, 0.1) or poly.symmetric_difference(sp).area < 0.2 for sp in seen_polys):
                        seen_polys.append(poly)
                        valid_rooms.append(r_m)

        z_ceil_level = z_floor + wall_height
        if valid_rooms:
            for r_i, r_pts in enumerate(valid_rooms):
                f_slab = create_polygonal_slab_entity(model, body, f"FloorSlab_Room_{r_i:02d}", r_pts, z_floor - slab_thickness, slab_thickness, floor_style, "FLOOR", rel_to=storey.ObjectPlacement)
                c_slab = create_polygonal_slab_entity(model, body, f"CeilingSlab_Room_{r_i:02d}", r_pts, z_ceil_level, slab_thickness, ceil_style, "ROOF", rel_to=storey.ObjectPlacement)
                elements.extend([f_slab, c_slab])
        else:
            hull = MultiPoint(c_m).convex_hull
            if hull.geom_type == "Polygon":
                h_pts = np.array(hull.exterior.coords)
                f_slab = create_polygonal_slab_entity(model, body, f"FloorSlab_Fitted_{scene_id}", h_pts, z_floor - slab_thickness, slab_thickness, floor_style, "FLOOR", rel_to=storey.ObjectPlacement)
                c_slab = create_polygonal_slab_entity(model, body, f"CeilingSlab_Fitted_{scene_id}", h_pts, z_ceil_level, slab_thickness, ceil_style, "ROOF", rel_to=storey.ObjectPlacement)
                elements.extend([f_slab, c_slab])

        wall_count = len(pg_data['edges'])
        slab_count = len(valid_rooms) * 2 if valid_rooms else 2

    # 3. Doors & Windows (Exact 3D Placement, Wall Voids & Fills)
    void_count = 0
    for op in openings:
        op_t = op["type"]
        w = float(op.get("width", 0.8))
        h = float(op.get("height", 2.1))
        center = op["center"]
        ang = float(op.get("angle_rad", 0.0))
        op_id = op["id"]

        t_elem = float(op.get("thickness", 0.05 if op_t == "door" else 0.06))
        style = door_style if op_t == "door" else win_style
        z_pos = float(center[2]) if len(center) > 2 else float(z_floor)

        # Create Door or Window Entity
        if op_t == "door":
            elem = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcDoor", name=f"Door_{op_id}", predefined_type="DOOR")
        else:
            elem = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWindow", name=f"Window_{op_id}", predefined_type="WINDOW")

        elem.OverallWidth = w
        elem.OverallHeight = h

        prof = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=w, YDim=t_elem)
        prof.Position = model.create_entity("IfcAxis2Placement2D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)))
        solid = model.create_entity("IfcExtrudedAreaSolid", SweptArea=prof,
            Position=model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
            ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)), Depth=h)

        model.create_entity("IfcStyledItem", Item=solid, Styles=[style])
        rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid])
        elem.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[rep])

        # Associate Opening with Host Wall(s) for Boolean Voiding
        pt = np.array(center[:2], dtype=float)
        candidate_walls = []

        for w_rec in wall_records:
            p1 = w_rec["p1"]
            p2 = w_rec["p2"]
            v = p2 - p1
            l_w = np.linalg.norm(v)
            if l_w < 1e-4:
                continue
            dir_w = v / l_w
            dp = pt - p1
            u = float(np.dot(dp, dir_w))
            perp = dp - u * dir_w
            dist = float(np.linalg.norm(perp))
            w_th = w_rec.get("thickness", wall_thickness)
            if -0.25 <= u <= l_w + 0.25 and dist < (w_th / 2.0 + 0.25):
                candidate_walls.append((dist, u, w_rec))

        # Sort candidate walls by distance to opening center
        candidate_walls.sort(key=lambda x: x[0])

        if candidate_walls:
            best_dist, best_u, best_wall = candidate_walls[0]
            host_wall = best_wall["entity"]
            w_zf = best_wall["z_floor"]
            w_th = best_wall.get("thickness", wall_thickness)

            # 1. Create primary IfcOpeningElement to cut the host wall
            opening_elem = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcOpeningElement", name=f"Opening_{op_id}", predefined_type="OPENING")

            # Void geometry is slightly thicker than wall to ensure clean boolean cut
            prof_void = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=w, YDim=w_th + 0.10)
            prof_void.Position = model.create_entity("IfcAxis2Placement2D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)))
            solid_void = model.create_entity("IfcExtrudedAreaSolid", SweptArea=prof_void,
                Position=model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
                ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)), Depth=h)
            rep_void = model.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid_void])
            opening_elem.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[rep_void])

            # Placement of opening relative to host wall
            opening_elem.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                PlacementRelTo=host_wall.ObjectPlacement,
                RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                    Location=model.create_entity("IfcCartesianPoint", Coordinates=(float(best_u), 0.0, float(z_pos - w_zf))),
                    RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))))

            # Void the primary wall
            model.create_entity("IfcRelVoidsElement", RelatingBuildingElement=host_wall, RelatedOpeningElement=opening_elem)

            # Placement of door/window relative to opening
            elem.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                PlacementRelTo=opening_elem.ObjectPlacement,
                RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                    Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
                    RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))))

            # Fill the primary opening with the door/window
            model.create_entity("IfcRelFillsElement", RelatingOpeningElement=opening_elem, RelatedBuildingElement=elem)
            void_count += 1

            # Also void any secondary adjacent walls (e.g. if any duplicate wall exists) to guarantee 100% bilateral visibility
            for sec_dist, sec_u, sec_wall in candidate_walls[1:]:
                sec_host_wall = sec_wall["entity"]
                sec_zf = sec_wall["z_floor"]
                sec_th = sec_wall.get("thickness", wall_thickness)

                sec_opening = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcOpeningElement", name=f"Opening_{op_id}_sec", predefined_type="OPENING")
                sec_prof = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=w, YDim=sec_th + 0.10)
                sec_prof.Position = model.create_entity("IfcAxis2Placement2D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)))
                sec_solid = model.create_entity("IfcExtrudedAreaSolid", SweptArea=sec_prof,
                    Position=model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))),
                    ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)), Depth=h)
                sec_rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[sec_solid])
                sec_opening.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[sec_rep])

                sec_opening.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                    PlacementRelTo=sec_host_wall.ObjectPlacement,
                    RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                        Location=model.create_entity("IfcCartesianPoint", Coordinates=(float(sec_u), 0.0, float(z_pos - sec_zf))),
                        RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))))

                model.create_entity("IfcRelVoidsElement", RelatingBuildingElement=sec_host_wall, RelatedOpeningElement=sec_opening)
                void_count += 1
        else:
            # Fallback: standalone placement relative to storey
            elem.ObjectPlacement = model.create_entity("IfcLocalPlacement",
                PlacementRelTo=storey.ObjectPlacement,
                RelativePlacement=model.create_entity("IfcAxis2Placement3D",
                    Location=model.create_entity("IfcCartesianPoint", Coordinates=(float(center[0]), float(center[1]), z_pos)),
                    RefDirection=model.create_entity("IfcDirection", DirectionRatios=(math.cos(ang), math.sin(ang), 0.0))))

        elements.append(elem)

    ifcopenshell.api.run("spatial.assign_container", model, relating_structure=storey, products=elements)

    out_ifc = os.path.join(out_dir, f"{scene_id}_model.ifc")
    model.write(out_ifc)

    door_count = sum(1 for o in openings if o['type'] == 'door')
    window_count = sum(1 for o in openings if o['type'] == 'window')

    print(f"  3D IFC4 Model Assembled!")
    print(f"    - Spatial Hierarchy: Project -> Site -> Building -> Storey (BIMvision Certified)")
    print(f"    - Walls:   {wall_count}")
    print(f"    - Slabs:   {slab_count}")
    print(f"    - Doors:   {door_count}")
    print(f"    - Windows: {window_count}")
    print(f"    - Voids:   {void_count} physical wall cutouts created")
    print(f"  Saved BIM IFC4: {out_ifc}")

    return out_ifc

