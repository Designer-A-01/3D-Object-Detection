import pyquaternion

from typing import Dict, Optional, Sequence, Union
from nuscenes.utils.data_classes import Box as NuScenesBox

import tensorrt as trt
import torch
import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
from mmdet3d.apis.inference import show_result_meshlab

import argparse

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')

DefaultAttribute = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('engine', help='checkpoint file')
    parser.add_argument('--samples', default=500, help='samples to benchmark')
    parser.add_argument('--postprocessing', action='store_true')
    args = parser.parse_args()
    return args


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')
    
class TRTWrapper(torch.nn.Module):

    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        bindings = [None] * (len(self._input_names) + len(self._output_names))

        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))            
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.zeros(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def get_quaternion_from_euler(e):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll = e[0]
    pitch = e[1]
    yaw = e[2]

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]


""" 
jeho
Insta360 frame에 대한 bbox outputs formatting
""" 
def format_single_bbox_insta360(results, data):
    nusc_annos = {}
    mapped_class_names = CLASSES

    print('Start to convert detection format...')
    det = results[0] # single inference result
    boxes = det['boxes_3d'].tensor.numpy()
    scores = det['scores_3d'].numpy()
    labels = det['labels_3d'].numpy()
    
    ego2global_trans = data['ego2global'][:3, -1].cpu().numpy().tolist()
    ego2global_rot = data['ego2global'][:3, :3].cpu().numpy().tolist()
    r = R.from_matrix(ego2global_rot)
    quat = r.as_quat()
    quat_xyz = quat[:3]
    quat_w = [quat[-1]]
    quat = np.concatenate([quat_w, quat_xyz])
    
    trans = ego2global_trans
    rot = pyquaternion.Quaternion(quat)

    # trans = self.data_infos[sample_idx]['cams'][self.ego_cam]['ego2global_translation']
    # rot = self.data_infos[sample_idx]['cams'][self.ego_cam]['ego2global_rotation']
    # rot = pyquaternion.Quaternion(rot)
    
    annos = list()
    for i, box in enumerate(boxes):
        name = mapped_class_names[labels[i]]
        center = box[:3]
        wlh = box[[4, 3, 5]]
        box_yaw = box[6]
        box_vel = box[7:].tolist()
        box_vel.append(0)
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
        nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
        
        # bbox from ego to global
        nusc_box.rotate(rot)
        nusc_box.translate(trans)
        
        if np.sqrt(nusc_box.velocity[0]**2 + nusc_box.velocity[1]**2) > 0.2:
            if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = DefaultAttribute[name]
                
        nusc_anno = dict(
            # sample_token=sample_token,
            translation=nusc_box.center.tolist(),
            size=nusc_box.wlh.tolist(),
            rotation=nusc_box.orientation.elements.tolist(),
            velocity=nusc_box.velocity[:2],
            detection_name=name,
            detection_score=float(scores[i]),
            attribute_name=attr,
        )
        annos.append(nusc_anno)
    
    # nusc_annos[sample_idx].extend(annos)
    # nusc_annos[sample_idx] = annos
    
    # nusc_submissions = {
    #     'meta': self.modality,
    #     'results': nusc_annos,
    # }

    return annos