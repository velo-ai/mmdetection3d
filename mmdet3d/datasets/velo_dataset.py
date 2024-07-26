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
        'classes': ('Pedestrian', 'Cyclist', 'Car', 'Truck', 'Motorcycle')
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = "",
                 pipeline: List[Union[dict, Callable]] = [],
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

        locations = np.array([annotation["location"] for annotation in annotations])
        dimensions = np.array([annotation["dimensions"] for annotation in annotations])
        yaw = np.array([annotation["yaw"] for annotation in annotations])

        gt_bboxes_3d = LiDARInstance3DBoxes(
            np.concatenate((
                locations,
                dimensions,
                yaw[:, np.newaxis],
            ), axis=-1),
            box_dim=7,
            origin=(0.5, 0.5, 0.5)
        )

        gt_labels_3d = np.array([
            self.CLASSES.index(annotation["label"]) for annotation in annotations
        ]).astype(np.int64)
        gt_names = np.array([
            annotation["label"] for annotation in annotations
        ])

        ann_info = {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_names": gt_names,
        }

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        info["num_pts_feats"] = info["lidar_points"]["num_features"]
        info["lidar_path"] = str(pathlib.Path(self.data_root) / info["point_cloud"]["path"])

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)

        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
