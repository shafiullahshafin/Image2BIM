"""
IfcOpenShell Helper Functions for IFC4 Model Assembly.
"""

from typing import List, Tuple
import numpy as np
import ifcopenshell
import ifcopenshell.api


def create_surface_style(model, name: str, r: float, g: float, b: float, transparency: float = 0.0):
    """Create an IFC SurfaceStyle with RGB color and optional transparency."""
    color = model.create_entity("IfcColourRgb", Name=f"{name}_Color", Red=float(r), Green=float(g), Blue=float(b))
    shading = model.create_entity("IfcSurfaceStyleRendering",
        SurfaceColour=color,
        Transparency=float(transparency),
        ReflectanceMethod="MATT"
    )
    style = model.create_entity("IfcSurfaceStyle", Name=f"{name}_Style", Side="BOTH", Styles=[shading])
    return style


def create_polygonal_slab_entity(model, body, name: str, poly_pts_m: np.ndarray,
                                 z_elevation: float, thickness: float, style,
                                 predefined_type: str = "FLOOR",
                                 rel_to=None):
    """
    Create an arbitrary polygonal IfcSlab (Floor or Ceiling) with exact boundary loops.
    """
    pts = [tuple(p) for p in poly_pts_m]
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    cartesian_pts = [model.create_entity("IfcCartesianPoint", Coordinates=(float(p[0]), float(p[1]))) for p in pts]
    polyline = model.create_entity("IfcPolyline", Points=cartesian_pts)

    profile = model.create_entity("IfcArbitraryClosedProfileDef",
        ProfileType="AREA",
        ProfileName=f"Profile_{name}",
        OuterCurve=polyline
    )

    placement_3d = model.create_entity("IfcAxis2Placement3D",
        Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
        Axis=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    )

    solid = model.create_entity("IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement_3d,
        ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        Depth=float(thickness)
    )

    if style:
        model.create_entity("IfcStyledItem", Item=solid, Styles=[style])

    shape_rep = model.create_entity("IfcShapeRepresentation",
        ContextOfItems=body,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid]
    )

    slab = ifcopenshell.api.run("root.create_entity", model,
        ifc_class="IfcSlab",
        name=name,
        predefined_type=predefined_type
    )
    slab.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])

    slab.ObjectPlacement = model.create_entity("IfcLocalPlacement",
        PlacementRelTo=rel_to,
        RelativePlacement=model.create_entity("IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, float(z_elevation))),
            Axis=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )
    )
    return slab
