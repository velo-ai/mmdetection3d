import pathlib
import json
from typing import Dict, Any
import numpy as np
from spatialmath import SE3, SO3
from mmengine import dump

# This takes in a velo dataset and generates the pkl files with the annotations
# It also generates those pesky binary pcd files


world_t_world_mmdet = SE3(SO3.TwoVectors(x="y", y="-x"))

def lidar_transform_points(points_b: np.array, a_t_b: SE3) -> np.array:
    points_b_h = np.concatenate((
        points_b,
        np.ones((points_b.shape[0], 1))
    ), axis=1)

    a_t_b_mat = np.array(a_t_b)

    points_a_h = np.transpose(a_t_b_mat @ np.transpose(points_b_h))

    points_a = points_a_h[:, :3] / points_a_h[:, 3][:, np.newaxis]

    return points_a

def get_info(root_dir: pathlib.Path, seqs: [str]) -> [Any]:
    infos = []

    for seq in seqs:
        seq_dir = root_dir / seq

        meta_path = seq_dir / "meta.json"

        pcd_rot_dir = seq_dir / "pcd_rot"
        pcd_rot_dir.mkdir(exist_ok=True)

        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        for frame in meta["frames"]:
            pcd_in_path = (seq_dir / "pcd" / frame).with_suffix(".bin")
            pcd_path = (seq_dir / "pcd_rot" / frame).with_suffix(".bin")

            with open(pcd_in_path, "rb") as fp:
                pcd_in_data = fp.read()

            points = np.frombuffer(pcd_in_data, dtype=np.float32).copy().reshape((-1, 4))
            points[:, 0:3] = lidar_transform_points(points[:, 0:3], world_t_world_mmdet.inv())

            with open(pcd_path, "wb") as fp:
                fp.write(points.tobytes())

            relative_pcd_bin_path = (pathlib.Path(seq_dir.name) / "pcd_rot" / frame).with_suffix(".bin")

            annot_path = (seq_dir / "annot" / frame).with_suffix(".json")

            with open(annot_path, "r") as fp:
                annot = json.load(fp)
            
            poses = [SE3(np.array(cuboid["pose"]).reshape((4, 4))) for cuboid in annot["cuboids"]]
            poses = [world_t_world_mmdet.inv() * pose for pose in poses]

            name = np.array([cuboid["label"] for cuboid in annot["cuboids"]])
            location = np.array([pose.t for pose in poses])
            dimensions = np.array([
                np.array(cuboid["dimensions"]) for cuboid in annot["cuboids"]
            ])
            yaw = np.array([pose.rpy()[2] for pose in poses])

            frame_info = {
                "point_cloud": {
                    "num_features": 4,
                    "path": str(relative_pcd_bin_path),
                },
                "annotations": {
                    "name": name,
                    "location": location,
                    "dimensions": dimensions,
                    "yaw": yaw,
                },
            }

            infos.append(frame_info)

    info = {
        "metainfo": {},
        "data_list": infos,
    }

    return info


def create_velo_infos(root_dir: str):
    root_dir = pathlib.Path(root_dir)

    splits_path = root_dir / "splits.json"
    with open(splits_path, "r") as fp:
        splits = json.load(fp)

    train_info = get_info(
        root_dir,
        seqs=splits["train"],
    )
    train_info_path = root_dir / "velo_infos_train.pkl"
    dump(train_info, train_info_path)

    val_info = get_info(
        root_dir,
        seqs=splits["val"],
    )
    val_info_path = root_dir / "velo_infos_val.pkl"
    dump(val_info, val_info_path)

    test_info = get_info(
        root_dir,
        seqs=splits["test"],
    )
    test_info_path = root_dir / "velo_infos_test.pkl"
    dump(test_info, test_info_path)


