"""
Data Collection Script for Mesh Deformation Learning
-----------------------------------------------------
This script extracts vertex positions and deformation deltas from a Maya scene,
preparing them for use in training a machine learning model.

Responsibilities:
- Gathers vertex positions and deformation data from a deformed mesh.
- Saves structured training data for downstream model training.
"""


from maya.api import OpenMaya as om2
from maya import cmds
from typing import List, Tuple
import numpy as np
import random
import torch

def get_point_positions_and_deltas(
        mesh_name: str,
        deformer_name: str,
        base_mesh_name: str
        ) -> Tuple[List[om2.MPoint], List[Tuple[float, float, float]], float]:
    """
    Extracts the positions of mesh points, their deformation deltas, and the inflation factor from the Maya scene.

    Args:
        mesh_name (str): Name of the mesh in the Maya scene.
        deformer_name (str): Name of the deformer applied to the mesh.
        base_mesh_name (str): Name of the base mesh to use for comparison.

    Returns:
        tuple: A tuple containing:
            - original_points (list of MPoint): The original positions of the points in the mesh.
            - deltas (list of tuple): The deformation deltas for each point (delta_x, delta_y, delta_z).
            - inflate_amount (float): The inflation factor applied by the deformer.
    """
    selection_list = om2.MSelectionList()
    selection_list.add(mesh_name)
    mesh_obj = selection_list.getDagPath(0)
    mesh_fn = om2.MFnMesh(mesh_obj)
    deformed_points = mesh_fn.getPoints(om2.MSpace.kWorld)

    base_selection = om2.MSelectionList()
    base_selection.add(base_mesh_name)
    base_mesh_obj = base_selection.getDagPath(0)
    base_mesh_fn = om2.MFnMesh(base_mesh_obj)
    original_points = base_mesh_fn.getPoints(om2.MSpace.kWorld)

    selection_list.add(deformer_name)
    deformer_obj = selection_list.getDependNode(1)
    deformer_fn = om2.MFnDependencyNode(deformer_obj)
    inflate_attr = deformer_fn.attribute("inflateAmount")
    inflate_amount_plug = om2.MPlug(deformer_obj, inflate_attr)
    inflate_amount: float = inflate_amount_plug.asFloat()

    deltas: List[Tuple[float, float, float]] = [
        (deformed_points[i].x - original_points[i].x,
        deformed_points[i].y - original_points[i].y,
        deformed_points[i].z - original_points[i].z)
        for i in range(len(deformed_points))
    ]

    return original_points, deltas, inflate_amount


# Open the Maya file
cmds.file("P:/Code/github/pixolusion/deformerPlayground/data/torusTest.mb", open=True, force=True)

# Configuration
mesh_name = "pTorusShape1"
deformer_name = "inflateDeformer1"
base_mesh_name = "pTorusShape1Orig"
start_frame = 1
end_frame = 120

X_train_list = []
y_train_list = []
points, deltas, inflation_factor = get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)

# Add explicit zero-inflation data
for i, point in enumerate(points):
    X_train_list.append([point.x, point.y, point.z, 0.0])  # Inflation factor 0
    y_train_list.append([0.0, 0.0, 0.0])  # No deformation

for frame in range(start_frame, end_frame + 1):
    cmds.currentTime(frame)
    points, deltas, inflation_factor = get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)
    for i, point in enumerate(points):
        X_train_list.append([point.x, point.y, point.z, inflation_factor])
        y_train_list.append([deltas[i][0], deltas[i][1], deltas[i][2]])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

torch.save((X_train, y_train), "P:/Code/github/pixolusion/deformerPlayground/data/training_data.pt")
print("Data collection complete. Saved training data.")
