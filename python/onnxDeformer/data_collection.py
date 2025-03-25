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
import argparse
import numpy as np
import torch


def get_point_positions_and_deltas(
    mesh_name: str, deformer_name: str, base_mesh_name: str
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
    deformed_points = mesh_fn.getPoints()

    base_selection = om2.MSelectionList()
    base_selection.add(base_mesh_name)
    base_mesh_obj = base_selection.getDagPath(0)
    base_mesh_fn = om2.MFnMesh(base_mesh_obj)
    original_points = base_mesh_fn.getPoints()
    original_normals = base_mesh_fn.getNormals()

    selection_list.add(deformer_name)
    deformer_obj = selection_list.getDependNode(1)
    deformer_fn = om2.MFnDependencyNode(deformer_obj)
    inflate_attr = deformer_fn.attribute("inflateAmount")
    inflate_amount_plug = om2.MPlug(deformer_obj, inflate_attr)
    inflate_amount: float = inflate_amount_plug.asFloat()

    deltas: List[Tuple[float, float, float]] = [
        (
            deformed_points[i].x - original_points[i].x,
            deformed_points[i].y - original_points[i].y,
            deformed_points[i].z - original_points[i].z,
        )
        for i in range(len(deformed_points))
    ]

    return original_points, original_normals, deltas, inflate_amount


def collect_data_from_maya(
    mesh_name: str,
    deformer_name: str,
    base_mesh_name: str,
    start_frame: int,
    end_frame: int,
):
    """Collects training data from Maya by iterating over frames."""
    X_train_list, y_train_list = [], []

    for frame in range(start_frame, end_frame + 1):
        cmds.currentTime(frame)
        orig_points, original_normals, deltas, inflation_factor = (
            get_point_positions_and_deltas(mesh_name, deformer_name, base_mesh_name)
        )
        for i, point in enumerate(orig_points):
            X_train_list.append(
                [
                    point.x,
                    point.y,
                    point.z,
                    original_normals[i].x,
                    original_normals[i].y,
                    original_normals[i].z,
                    inflation_factor,
                ]
            )
            y_train_list.append([deltas[i][0], deltas[i][1], deltas[i][2]])

    return np.vstack(X_train_list), np.vstack(y_train_list)  # Stack into single arrays


def save_training_data(X_train: np.ndarray, y_train: np.ndarray, file_path: str):
    """Saves training data to a file."""
    torch.save((X_train, y_train), file_path)


def collect(file_path):
    cmds.loadPlugin("inflateDeformerCPP")
    transform = cmds.polyTorus(ch=False)[0]
    shape = cmds.listRelatives(transform)[0]
    deformer_name = cmds.deformer(type="inflateDeformer")[0]
    cmds.currentTime(-50)
    cmds.setAttr(f"{deformer_name}.ia", -0.5)
    cmds.setKeyframe(f"{deformer_name}.ia")
    cmds.currentTime(0)
    cmds.setAttr(f"{deformer_name}.ia", 0)
    cmds.setKeyframe(f"{deformer_name}.ia")
    cmds.currentTime(100)
    cmds.setAttr(f"{deformer_name}.ia", 1)
    cmds.setKeyframe(f"{deformer_name}.ia")
    x_train, y_train = collect_data_from_maya(
        shape, deformer_name, f"{shape}Orig", -100, 100
    )
    save_training_data(x_train, y_train, file_path)
    print(f"\nSaved training data to: {file_path}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and save a model for mesh deformation prediction."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output data file. (.pt)"
    )
    args = parser.parse_args()
    training_file = args.output_file.replace("\\", "/")
    print("Collecting")
    collect(training_file)


if __name__ == "__main__":
    import maya.standalone

    maya.standalone.initialize(name="python")
    main()
    maya.standalone.uninitialize()
