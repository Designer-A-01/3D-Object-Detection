import io
import os
import time
from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch
import torch.onnx
from mmcv import Config, imdenormalize
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
from mmdet3d.apis.inference import show_result_meshlab

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import argparse

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes

from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from pyquaternion.quaternion import Quaternion

import cv2
import numpy as np
import matplotlib.cm as cm

import mmcv
import ctypes

from trt_tools import *

views = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

class Frame_uInt8(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int)]

class Frame_float(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int)]


def initFrameData(device):
    tangent_intrinsics = {'CAM_FRONT_LEFT': [[1.31669199e+03, 0.00000000e+00, 7.71567974e+02], # Tangent location 0 (leftmost)
                                            [0.00000000e+00, 1.30594375e+03, 4.27529182e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                        'CAM_FRONT': [[1.32277551e+03, 0.00000000e+00, 7.56801337e+02], # Tangent location 1
                                        [0.00000000e+00, 1.31076362e+03, 4.17111552e+02],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                        'CAM_FRONT_RIGHT': [[1.31303854e+03, 0.00000000e+00, 7.16879740e+02], # Tangent location 2
                                            [0.00000000e+00, 1.30008012e+03, 4.21897818e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                        'CAM_BACK_LEFT': [[1.31538668e+03, 0.00000000e+00, 7.62655552e+02], # Tangent location 3
                                            [0.00000000e+00, 1.30582175e+03, 4.21392564e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                        'CAM_BACK': [[1.37645753e+03, 0.00000000e+00, 7.33078005e+02], # Tangent location 4
                                    [0.00000000e+00, 1.36126888e+03, 4.13681559e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                        'CAM_BACK_RIGHT': [[1.33525055e+03, 0.00000000e+00, 7.02715248e+02], # Tangent location 5
                                            [0.00000000e+00, 1.32071237e+03, 4.12790092e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]}

    tangent_intrinsics_list = []
    for cam in views:
        tangent_intrinsics_list.append(torch.Tensor(tangent_intrinsics[cam]))
    tangent_intrinsics = torch.stack(tangent_intrinsics_list)

    # sensor2ego rotation 변경!
    sensor2ego_rot_eulers = {'CAM_BACK_LEFT': [-90.0, 0.0, 240.0],
                            'CAM_BACK': [-90.0, 0.0, 180.0],
                            'CAM_BACK_RIGHT': [-90.0, 0.0, 120.0],
                            'CAM_FRONT_LEFT': [-90.0, 0.0, 60.0],
                            'CAM_FRONT': [-90.0, 0.0, 0.0],
                            'CAM_FRONT_RIGHT': [-90.0, 0.0, -60.0]}
    
    cur_data = dict()

    sensor2ego_trans = [0.0, 0.0, 1.5]

    # sensor2ego matrix
    sensor2ego_mats = []
    for cam in views:
        sensor2ego_degrees = sensor2ego_rot_eulers[cam]
        sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
        sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
        w, x, y, z = sensor2ego_q

        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(sensor2ego_trans)

        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        
        sensor2ego_mats.append(sensor2ego)
    sensor2ego_mats = torch.stack(sensor2ego_mats)

    # initial ego pose matrix
    w, x, y, z = np.array([1.0, 0.0, 0.0, 0.0])
    ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
    ego2global_tran = torch.Tensor(np.array([0.0, 0.0, 0.0]))
    cur_ego2global = ego2global_rot.new_zeros((4, 4))
    cur_ego2global[3, 3] = 1
    cur_ego2global[:3, :3] = ego2global_rot
    cur_ego2global[:3, -1] = ego2global_tran

    cur_data['sensor2ego_mats'] = sensor2ego_mats.to(device)
    cur_data['ego2global'] = cur_ego2global.to(device)
    cur_data['intrinsics'] = tangent_intrinsics.to(device)

    return cur_data


def extractBEVFeature(cfg, cur_data, device):
    # size data
    tangent_h = 396 # 900 # 396
    tangent_w = 704 # 1600 # 704
    H, W = cfg.data_config['src_size']
    fH, fW = cfg.data_config['input_size'] # 256, 704
    newH, newW = tangent_h, tangent_w
    crop_h = int((1 - np.mean(cfg.data_config['crop_h'])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    tangent_crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    
    # post-homography transformation
    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)

    resize = float(fW) / float(W)
    resize += cfg.data_config.get('resize_test', 0.0)
    rotate = 0

    post_rot *= resize
    post_tran -= torch.Tensor(tangent_crop[:2])

    rot_h = rotate / 180 * np.pi
    A = torch.Tensor([[np.cos(rot_h), np.sin(rot_h)], [-np.sin(rot_h), np.cos(rot_h)]])
    b = torch.Tensor([tangent_crop[2] - tangent_crop[0], tangent_crop[3] - tangent_crop[1]]) / 2
    b = A.matmul(-b) + b

    post_rot2 = A.matmul(post_rot)
    post_tran2 = A.matmul(post_tran) + b
    post_tran = torch.zeros(3) # for convenience, make augmentation matrices 3x3
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2

    post_rot = post_rot.to(device)
    post_tran = post_tran.to(device)

    rots, trans = [], []
    post_trans, post_rots, intrins = [], [], [] # only used for BEV feature extraction, not used in BEV alignment  
    
    for cam_idx, cam_name in enumerate(cfg.data_config['cams']):
        # current sensor to ego matrix
        sweepsensor2sweepego_mat = cur_data['sensor2ego_mats'][cam_idx]
        rot = sweepsensor2sweepego_mat[:3, :3]
        tran = sweepsensor2sweepego_mat[:3, 3]

        rots.append(rot)
        trans.append(tran)
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        intrins.append(cur_data['intrinsics'][cam_idx])
    
    dummy = torch.zeros((0, 0))
    rots = torch.stack(rots).unsqueeze(0).to(device) # [1, 6, 3, 3]
    trans = torch.stack(trans).unsqueeze(0).to(device) # [1, 6, 3]
    post_trans = torch.stack(post_trans).unsqueeze(0).to(device) # [1, 6, 3] 
    post_rots = torch.stack(post_rots).unsqueeze(0).to(device) # [1, 6, 3, 3]
    intrins = torch.stack(intrins).unsqueeze(0).to(device) # [1, 6, 3, 3]
    bda = torch.eye(3).unsqueeze(0).to(device) # TODO

    return [dummy, rots, trans, intrins, post_rots, post_trans, bda]




def main():

    ## Preprocessing
    n_patch = 6
    device = "cuda:0"

    input_video = "/home/mobed/Workspace/jetson_player/omni3DOD/input/daejeon_road_scene.insv"

    # Load library (5ms)

    clib = ctypes.CDLL("./omni3DOD/build/libomni3DOD.so")

    print("Successfully loaded c++ plugins for omni3DOD")

    # Define the return type of the C function as an array
    clib.convertFEV2TP.restype = ctypes.POINTER(Frame_float)

    # Allocating Host and Device memory (8ms)
    clib.initHostDeviceMemory()    # 8ms

    load_tensorrt_plugin()

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # build tensorrt model
    trt_model = TRTWrapper(args.engine, [f'output_{i}' for i in range(36)])


    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ]
    

    
    
    # Set cameras
    threshold = 0.35
    box_vis_level = BoxVisibility.ANY

    # Video processing start
    video = cv2.VideoCapture(input_video)

    # Get the frame rate and the dimensions of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_interval = round(1000 / fps)


    # imsize = (704, 256)
    imsize = (1600, 900)
    channels = 3
    tangent_height = 256
    tangent_width = 704

    # Initialize camera pose related data
    cur_data = initFrameData(device) # {"ego2global", "sensor2ego_mats", "intrinsics"}
    meta_inputs = extractBEVFeature(cfg, cur_data, device)
    meta_outputs = model.get_bev_pool_input(meta_inputs)
    metas = dict(
        ranks_bev=meta_outputs[0].int().contiguous(),
        ranks_depth=meta_outputs[1].int().contiguous(),
        ranks_feat=meta_outputs[2].int().contiguous(),
        interval_starts=meta_outputs[3].int().contiguous(),
        interval_lengths=meta_outputs[4].int().contiguous())

    # Preprocess transformation matrix for postprocessing
    """ Box coord. transformation: global => ego => sensor => image """
    ego2global_translation = cur_data['ego2global'][:3, -1].cpu().numpy().tolist()
    ego2global_rotation = cur_data['ego2global'][:3, :3].cpu().numpy()
    r = R.from_matrix(ego2global_rotation)
    quat = r.as_quat()
    quat_xyz = quat[:3]
    quat_w = [quat[-1]]
    ego2global_rotation = np.concatenate([quat_w, quat_xyz]).tolist()

    sensor2ego_rot_list = []
    sensor2ego_trans_list = []
    intrinsic_list = []

    for i, cam in enumerate(views):
        sensor2ego_trans = cur_data['sensor2ego_mats'][i][:3, -1].cpu().numpy().tolist()

        sensor2ego_rot = cur_data['sensor2ego_mats'][i][:3, :3].cpu().numpy().tolist()
        r = R.from_matrix(sensor2ego_rot)
        quat = r.as_quat()
        quat_xyz = quat[:3]
        quat_w = [quat[-1]]
        sensor2ego_rot = np.concatenate([quat_w, quat_xyz]).tolist()

        intrinsic = cur_data['intrinsics'][i].cpu().numpy()
        intrinsic = np.array(intrinsic)

        sensor2ego_rot_list.append(sensor2ego_rot)
        sensor2ego_trans_list.append(sensor2ego_trans)
        intrinsic_list.append(intrinsic)



    # @@@@@@@@@@ Real-time inference start @@@@@@@@@@
    while video.isOpened():
        # Read the current frame (BGR)
        ret, frame = video.read()   

        if ret:

            # Convert to RGB and float type
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)

            # Create a Frame object and convert to Tangent patches (18~27ms)
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            frame_obj = Frame_float()
            frame_obj.data = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            frame_obj.width = frame.shape[1]
            frame_obj.height = frame.shape[0]


            frames_ptr = clib.convertFEV2TP(ctypes.byref(frame_obj)) # BGR

            # frame_tensor = torch.zeros((n_patch, channels, tangent_height, tangent_width))
            # frame_tensor.data_ptr() = frames_ptr


            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            print("time (ms): ", elapsed * 1000) # @@@@@@@@@@@@@@@@@@@@@@@@@


            # Detecting and visualizing (100~120ms)
            frames = []
            vis_frames = []
            for i in range(n_patch):
                frame_data = np.ctypeslib.as_array(frames_ptr[i].data, shape=(frames_ptr[i].height, frames_ptr[i].width, 3))
                vis_frames.append(frame_data)

                frame_data = np.transpose(frame_data, [2, 0, 1])
                frames.append(frame_data)


            frames = np.array(frames)
            frames = frames[[2,3,4,5,0,1],:,:,:]
            frames = torch.from_numpy(frames)
            frames = frames.to(device)  # bottleneck
            

            # Perform BEVDet object detection
            frames = frames.squeeze(0).contiguous()
            trt_output = trt_model.forward(dict(img=frames, **metas))  # 20~30ms
            trt_output = [trt_output[f'output_{i}'] for i in range(36)]
        
            # Convert box results into insta360 format
            pred = model.result_deserialize(trt_output)
            img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
            bbox_list = model.pts_bbox_head.get_bboxes(
                pred, img_metas, rescale=True)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]

            result = format_single_bbox_insta360(bbox_results, cur_data)            
            


            for i, cam in enumerate(views):

                # patch = frames[i,:,:,:].permute([1, 2, 0]).numpy()
                # patch = patch*255
                # patch = patch.astype(np.uint8)
                # print(patch.dtype)
                # patch = mmcv.imdenormalize(patch, 
                #                 np.array(img_conf['img_mean'], np.float32), # TODO check: img_mean, img_std?
                #                 np.array(img_conf['img_std'], np.float32), False)
                # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                patch = vis_frames[(i+2)%6]
                patch = mmcv.imdenormalize(patch, 
                                np.array(img_conf['img_mean'], np.float32), # TODO check: img_mean, img_std?
                                np.array(img_conf['img_std'], np.float32), True)
                # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = patch.astype(np.uint8)

                patch = cv2.resize(patch, (1600, 900), cv2.INTER_LINEAR)


                for box_dict in result:
                    # print(box_dict)
                    if box_dict['detection_score'] >= threshold and box_dict['detection_name'] in show_classes:
                        box = Box(
                            box_dict['translation'],
                            box_dict['size'],
                            Quaternion(box_dict['rotation']),
                            name=box_dict['detection_name']
                        )
                        
                        # box를 global => ego로 이동
                        box.translate(-np.array(ego2global_translation))
                        box.rotate(Quaternion(ego2global_rotation).inverse)

                        # box를 ego => camera로 이동
                        box.translate(-np.array(sensor2ego_trans_list[i]))
                        box.rotate(Quaternion(sensor2ego_rot_list[i]).inverse)
                        
                        if box_in_image(box, intrinsic_list[i], imsize, vis_level=box_vis_level):
                            c=cm.get_cmap('tab10')(show_classes.index(box.name))
                            print(box)
                            print(box.name)

                            # box를 camera => image로 이동해서 render
                            box.render_cv2(patch, view=intrinsic_list[i], normalize=True, colors=(c, c, c))

                cv2.imshow("Frame", patch)

                if (cam == 'CAM_BACK'):
                    # Finish streaming if the user input ESC
                    if cv2.waitKey(100) == 27:
                        break
                
        else:
            break
    



if __name__ == "__main__":
    main()
