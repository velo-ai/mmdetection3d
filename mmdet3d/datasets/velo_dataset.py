import numpy as np
from typing import Optional, List, Union, Callable
import pathlib
import copy

from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.box_3d_mode import Box3DMode


# See https://mmdetection3d.readthedocs.io/en/dev-1.x/advanced_guides/customize_dataset.html

@DATASETS.register_module()
class VeloDataset(BaseDataset):
    METAINFO = {
        'classes': ('pedestrian', 'cyclist', 'car', 'truck', 'motorcycle')
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = "",
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 box_type_3d: dict = "LiDAR",
                 backend_args: Optional[dict] = None,
                 test_mode: bool = False):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode
        )

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """

        input = self.get_data_info(index)
        input = copy.deepcopy(input)

        input["box_type_3d"] = LiDARInstance3DBoxes
        input["box_mode_3d"] = Box3DMode.LIDAR

        example = self.pipeline(input)

        return example

    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information.
        """

        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info

    def parse_ann_info(self, info: dict) -> Union[dict, None]:
        annotations = info["annotations"]

        if len(annotations) == 0:
            return None

        if annotations["location"].shape[0] > 0:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                np.concatenate((
                    annotations["location"],
                    annotations["dimensions"],
                    annotations["yaw"][:, np.newaxis],
                ), axis=-1),
                box_dim=7,
                origin=(0.5, 0.5, 0.5)
            )
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                np.zeros((0, 7)),
                box_dim=7,
                origin=(0.5, 0.5, 0.5)
            )

        gt_labels_3d = np.array([
            self.METAINFO["classes"].index(label) for label in annotations["name"]
        ]).astype(np.int64)
        gt_names = annotations["name"]

        ann_info = {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_names": gt_names,
        }

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        info["lidar_points"] = {
            "num_pts_feats": info["point_cloud"]["num_features"],
            "lidar_path": str(pathlib.Path(self.data_root) / info["point_cloud"]["path"])
        }

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        else:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
