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

"""
This file is the main meta launch file that launches manipulator-supported workflows.

This file launches 3 other launch files:

1. Sensor launch file which exposes RGB camera streams, camera info streams (intrinsic), and
   depth via regular SGM-based (with infrared emitter) for RealSense or ESS (DNN) for stereo
   cameras.
   Calibration-specific information of the setup (camera w.r.t robot base).
   For sim-based setup, we get those streams from Isaac Sim.
2. Robot driver launch file includes driver nodes for the UR robot and Robotiq 2F gripper that
   expose joint states and load the robot description files. For sim, it includes topic-based
   control of the robot in sim through ros2_control.
3. Core launch file. This includes cuMotion for GPU-accelerated motion planning, nvblox for
   3D reconstruction of the environment, and either DOPE or the combination of RT-DETR and
   FoundationPose for object detection and pose estimation.
"""

import os
from typing import List

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params
from isaac_ros_manipulation_ros_python_utils.launch_utils import get_str_variable

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter


def _is_manual_displayport_workflow(params):
    return (
        str(params.get('workflow_type', '')).upper() == 'DISPLAYPORT_CABLE_INSERTION' and
        str(params.get('displayport_insertion_use_manual_poses', '')).lower() == 'true'
    )


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    """
    Door into the core of Isaac Manipulator nodes.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        List[Node]: List of nodes

    """
    # The "workflow configuration" is a YAML file that determines which Isaac Manipulator
    # reference workflow is to be run as well as the detailed configuration of that workflow.
    manipulator_workflow_config_path = get_str_variable(context, 'manipulator_workflow_config')
    params = load_yaml_params(manipulator_workflow_config_path)

    isaac_ros_manipulation_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch')

    sensor_nodes = []
    if not _is_manual_displayport_workflow(params):
        # This launch file may be replaced with one for a different sensor setup,
        # so long as it provides RGB and depth.
        sensor_nodes = [IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                isaac_ros_manipulation_workflow_bringup_include_dir,
                '/sensors/cameras.launch.py',
            ]),
            launch_arguments={key: str(value) for key, value in params.items()}.items())]

    # Routes to the robot-specific driver launch file based on 'robot_launch_file_path'
    # in the workflow configuration YAML. This enables support for different robots
    # without modifying this file.
    driver_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_ros_manipulation_workflow_bringup_include_dir, '/drivers.launch.py']
        ),
        launch_arguments={
            'manipulator_workflow_config': manipulator_workflow_config_path,
        }.items())

    # The "core" Manipulator launch file launches nodes for cuMotion, nvblox, and either
    # DOPE or the combination of RT-DETR and FoundationPose.
    core_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_ros_manipulation_workflow_bringup_include_dir, '/workflows/core.launch.py']),
        launch_arguments={key: str(value) for key, value in params.items()}.items())
    manipulator_init_nodes = []
    if params['use_sim_time'] == 'true':
        manipulator_init_nodes += [SetParameter(name='use_sim_time', value=True)]

    return (
        manipulator_init_nodes + sensor_nodes + [core_nodes] + [driver_nodes]
    )


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'manipulator_workflow_config',
            description='Path to the yaml file which hosts configuration values for the'
                        'manipulator workflow.'
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
