import pathlib
import mmcv
import json
from typing import Dict, Any
import numpy as np
import pypcd4
from spatialmath import SE3

# This takes in a velo dataset and generates the pkl files with the annotations
# It also generates those pesky binary pcd files




def get_info(root_dir: pathlib.Path, seqs: [str]) -> [Any]:
    info = []

    for seq in seqs:
        seq_dir = root_dir / seq

        meta_path = seq_dir / "meta.json"

        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        for frame in meta["frames"]:
            relative_pcd_bin_path = (pathlib.Path("pcd") / frame).with_suffix(".bin")
            annot_path = (seq_dir / "annot" / frame).with_suffix(".json")

            with open(annot_path, "r") as fp:
                annot = json.load(fp)
            
            poses = [SE3(np.array(cuboid["pose"]).reshape((4, 4))) for cuboid in annot["cuboids"]]

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
                }
            }

            info.append(frame_info)

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
    mmcv.dump(train_info, train_info_path)

    val_info = get_info(
        root_dir,
        seqs=splits["val"],
    )
    val_info_path = root_dir / "velo_infos_val.pkl"
    mmcv.dump(val_info, val_info_path)

    test_info = get_info(
        root_dir,
        seqs=splits["test"],
    )
    test_info_path = root_dir / "velo_infos_test.pkl"
    mmcv.dump(test_info, test_info_path)


