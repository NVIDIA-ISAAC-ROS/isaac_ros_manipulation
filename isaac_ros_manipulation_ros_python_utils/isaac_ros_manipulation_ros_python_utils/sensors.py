# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
# flake8: noqa: I100
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    WorkflowType, CameraType, TrackingType, DepthType
)
import isaac_ros_manipulation_ros_python_utils.constants as constants

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import (
    IncludeLaunchDescription,
)


def get_calibration_parameters(workflow_type: WorkflowType,
                               use_sim_time: bool,
                               camera_type: CameraType,
                               setup: str,
                               num_cameras: int,
                               gripper_type: str,
                               robot_type: str = 'UR',
                               ) -> List[Node]:
    """
    Add a node to publish calibration parameters to the static transforms.

    Args
    ----
        workflow_config (WorkflowConfig): Workflow config

    Returns
    -------
        List[Node]: List of nodes

    """
    nodes = []
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )
    if not use_sim_time:
        tracking_type_str = str(TrackingType.NONE)
        if workflow_type == WorkflowType.POSE_TO_POSE:
            tracking_type_str = str(TrackingType.POSE_TO_POSE)
        elif workflow_type == WorkflowType.OBJECT_FOLLOWING:
            tracking_type_str = str(TrackingType.OBJECT_FOLLOWING)
        elif workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
            tracking_type_str = str(TrackingType.NONE)

        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/static_transforms.launch.py']
            ),
            launch_arguments={
                'broadcast_world_base_link': 'False',
                'camera_type': str(camera_type),
                'tracking_type': tracking_type_str,
                'calibration_name': setup,
                'gripper_type': gripper_type,
                'robot_type': robot_type,
                'num_cameras': str(num_cameras)
            }.items(),
        ))
    else:
        print('No calibration specific things needed for Isaac Sim')
    return nodes


def get_camera_nodes(num_cameras: str, setup: str,
                     camera_type: CameraType,
                     depth_type: str,
                     depth_engine_file_path: str,
                     enable_dnn_depth_in_realsense: bool,
                     workflow_type: WorkflowType) -> List[Node]:
    """
    Return nodes that enable camera streams onto ROS topics.

    It will either run the hawk or realsense node, else it will return nothing for Isaac Sim.

    Args
    ----
        num_cameras (int): Number of cameras
        setup (str): Setup name
        camera_type (CameraType): Camera type
        depth_type (str): Depth estimation type
        depth_engine_file_path (str): Engine file path for the selected depth type
        enable_dnn_depth_in_realsense (bool): Whether to enable DNN depth in realsense

    Returns
    -------
        List[Node]: List of camera based nodes (returns ESS as well for Hawk based setup)

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )

    # Isaac sim needs no camera nodes, Isaac sim will publish rectified imagers
    camera_nodes = []

    if workflow_type in (
        WorkflowType.PICK_AND_PLACE,
        WorkflowType.DISPLAYPORT_CABLE_INSERTION,
    ):
        # FoundationPose uses RealSense depth in these workflows.
        enable_depth = False
    else:
        # For object following, we currently have to use depth from realsense due to lack of
        # synchronization in the drop node. This is because FoundationPose expects exact
        # timestamp matching between color, depth and segmentation mask. The realsense driver
        # emits infra1/infra2 at slight timing offsets from the rgb imager. A potential
        # solution is to overwrite timestamps to be the same (like we do in Manipulator servers)
        # for close enough samples. Or make FoundationPose able to accept messages with some
        # maximum threshold timestamp difference.
        enable_depth = True

    if camera_type == CameraType.REALSENSE:
        # For FoundationStereo, use lower dropped_fps to run cameras at 1 Hz to reduce system stress
        is_foundation_stereo = depth_type in (
            str(DepthType.FOUNDATION_STEREO_LOW_RES),
            str(DepthType.FOUNDATION_STEREO_HIGH_RES))
        dropped_fps_value = '14' if (
            enable_dnn_depth_in_realsense and is_foundation_stereo) else '10'

        camera_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_files_include_dir, '/realsense.launch.py']),
            launch_arguments={
                'num_cameras': str(num_cameras),
                'camera_ids_config_name': setup,
                'enable_dnn_depth_in_realsense': 'True' if enable_dnn_depth_in_realsense else 'False',
                'enable_depth': 'True' if enable_depth else 'False',
                'dropped_fps': dropped_fps_value
            }.items())
        )

        if enable_dnn_depth_in_realsense:
            for i in range(int(num_cameras)):
                if depth_type == str(DepthType.ESS_LIGHT) or depth_type == str(DepthType.ESS_FULL):
                    camera_nodes.append(IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [launch_files_include_dir, '/ess.launch.py']),
                        launch_arguments={
                            'camera_type': str(camera_type),
                            'depth_type': depth_type,
                            'ess_engine_file_path': depth_engine_file_path,
                            'camera_namespace': f'camera_{i+1}',
                            'left_image_raw_topic': 'infra1/image_rect_raw_drop',
                            'left_camera_info_topic': 'infra1/camera_info_drop',
                            'right_image_raw_topic': 'infra2/image_rect_raw_drop',
                            'right_camera_info_topic': 'infra2/camera_info_drop',
                            'depth_output_topic': 'depth_image',
                            'rgb_output_topic': 'rgb/image_rect_color',
                            'rgb_camera_info_output_topic': 'rgb/camera_info',
                            'input_image_height': str(constants.REALSENSE_IMAGE_HEIGHT),
                            'input_image_width': str(constants.REALSENSE_IMAGE_WIDTH),
                        }.items()))
                elif depth_type in (str(DepthType.FOUNDATION_STEREO_LOW_RES),
                                    str(DepthType.FOUNDATION_STEREO_HIGH_RES)):
                    camera_nodes.append(IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [launch_files_include_dir, '/foundationstereo.launch.py']),
                        launch_arguments={
                            'camera_type': str(camera_type),
                            'depth_type': depth_type,
                            'foundationstereo_engine_file_path': depth_engine_file_path,
                            'camera_namespace': f'camera_{i+1}',
                            'left_image_raw_topic': 'infra1/image_rect_raw_drop',
                            'left_camera_info_topic': 'infra1/camera_info_drop',
                            'right_image_raw_topic': 'infra2/image_rect_raw_drop',
                            'right_camera_info_topic': 'infra2/camera_info_drop',
                            'depth_output_topic': 'depth_image',
                            'rgb_output_topic': 'rgb/image_rect_color',
                            'rgb_camera_info_output_topic': 'rgb/camera_info',
                            'input_image_height': str(constants.REALSENSE_IMAGE_HEIGHT),
                            'input_image_width': str(constants.REALSENSE_IMAGE_WIDTH),
                        }.items()))
                else:
                    raise ValueError(
                        f'DNN-enhanced depth type {depth_type} not supported for realsense')
    elif camera_type == CameraType.HAWK:
        camera_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_files_include_dir, '/hawk.launch.py']),
            launch_arguments={
                'num_cameras': str(num_cameras),
                'camera_ids_config_name': setup
            }.items()
        ))
        for i in range(int(num_cameras)):
            camera_name_prefix = f'hawk_camera_{i+1}'
            if depth_type == str(DepthType.ESS_LIGHT) or depth_type == str(DepthType.ESS_FULL):
                camera_nodes.append(IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([launch_files_include_dir, '/ess.launch.py']),
                    launch_arguments={
                        'camera_type': str(camera_type),
                        'depth_type': depth_type,
                        'ess_engine_file_path': depth_engine_file_path,
                        'left_image_raw_topic': 'left/image_raw_drop',
                        'left_camera_info_topic': 'left/camera_info_drop',
                        'right_image_raw_topic': 'right/image_raw_drop',
                        'right_camera_info_topic': 'right/camera_info_drop',
                        'depth_output_topic': 'depth_image',
                        'rgb_output_topic': 'rgb/image_rect_color',
                        'rgb_camera_info_output_topic': 'rgb/camera_info',
                        'camera_namespace': camera_name_prefix,
                        'input_image_height': str(constants.HAWK_IMAGE_HEIGHT),
                        'input_image_width': str(constants.HAWK_IMAGE_WIDTH)
                    }.items()))
            elif depth_type in (str(DepthType.FOUNDATION_STEREO_LOW_RES),
                                str(DepthType.FOUNDATION_STEREO_HIGH_RES)):
                camera_nodes.append(IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        [launch_files_include_dir, '/foundationstereo.launch.py']),
                    launch_arguments={
                        'camera_type': str(camera_type),
                        'depth_type': depth_type,
                        'foundationstereo_engine_file_path': depth_engine_file_path,
                        'left_image_raw_topic': 'left/image_raw_drop',
                        'left_camera_info_topic': 'left/camera_info_drop',
                        'right_image_raw_topic': 'right/image_raw_drop',
                        'right_camera_info_topic': 'right/camera_info_drop',
                        'depth_output_topic': 'depth_image',
                        'rgb_output_topic': 'rgb/image_rect_color',
                        'rgb_camera_info_output_topic': 'rgb/camera_info',
                        'camera_namespace': camera_name_prefix,
                        'input_image_height': str(constants.HAWK_IMAGE_HEIGHT),
                        'input_image_width': str(constants.HAWK_IMAGE_WIDTH),
                    }.items()))
            else:
                raise ValueError(
                    f'DNN-enhanced depth type {depth_type} not supported for hawk')
    else:
        print('No camera specific nodes needed for Isaac Sim')
    return camera_nodes
