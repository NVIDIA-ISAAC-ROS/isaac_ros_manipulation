# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
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

import importlib
import os
from typing import List, Union

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_ros_python_utils.config import CoreConfig
import isaac_ros_manipulation_ros_python_utils.constants as constants
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    CameraType, PoseEstimationType, WorkflowType
)
# flake8: noqa: I100
from isaac_ros_manipulation_ros_python_utils.perception import (
    get_dope_nodes, get_foundation_pose_nodes, get_object_detection_servers,
    get_object_selection_server
)
import isaac_ros_manipulation_ros_python_utils.workflows as workflow_utils_ext
from isaac_ros_manipulation_ros_python_utils.displayport_insertion import (
    get_displayport_insertion_nodes, get_displayport_insertion_orchestrator
)

from launch.actions import (
    IncludeLaunchDescription, Shutdown
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder


def get_visualization_actions(
    core_config: CoreConfig, moveit_config: MoveItConfigsBuilder
) -> List[Union[Node, IncludeLaunchDescription]]:
    """
    Return visualization actions based on config flags.

    Independently checks enable_rviz_visualization and enable_foxglove_visualization
    so callers don't need to gate on either flag.
    """
    actions = []

    if core_config.enable_rviz_visualization and not core_config.headless:
        rviz_params = [
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ]
        if core_config.use_sim_time:
            rviz_params.append({'use_sim_time': core_config.use_sim_time})

        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', core_config.rviz_config_file],
            parameters=rviz_params,
        ))

    if core_config.enable_foxglove_visualization:
        launch_files_include_dir = os.path.join(
            get_package_share_directory('isaac_ros_manipulation_bringup'),
            'launch', 'visualization'
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/foxglove_bridge.launch.py']
            ),
        ))

    return actions


def get_pose_estimation_nodes(core_config: CoreConfig) -> List[Node] | None:
    """
    Get pose estimation nodes, return None if not applicable.

    Args
    ----
        core_config (CoreConfig): Core config

    Returns
    -------
        List[Node]: List of pose estimation nodes

    """
    if core_config.workflow_config.workflow_type != WorkflowType.POSE_TO_POSE:
        if core_config.pose_estimation_config.pose_estimation_type == \
                PoseEstimationType.FOUNDATION_POSE:
            return get_foundation_pose_nodes(
                camera_config=core_config.camera_config,
                workflow_type=core_config.workflow_config.workflow_type,
                pose_estimation_config=core_config.pose_estimation_config
            )
        if core_config.pose_estimation_config.pose_estimation_type == PoseEstimationType.DOPE:
            if core_config.workflow_config.workflow_type == WorkflowType.OBJECT_FOLLOWING:
                return get_dope_nodes(camera_config=core_config.camera_config,
                                      dope_config=core_config.pose_estimation_config,
                                      workflow_type=core_config.workflow_config.workflow_type)
            else:
                raise NotImplementedError('Dope is not yet supported for pick and place')
    return []


def _get_internal_workflows(workflow_type: WorkflowType):
    """
    Import the launch helpers for workflows kept for internal use only.

    Raises
    ------
        NotImplementedError: If the workflow is not part of this release

    """
    module_name = 'isaac_ros_manipulation_ros_python_utils.internal_workflows'
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise NotImplementedError(
            f'Workflow type {workflow_type.value} is not part of this release; use '
            f'{WorkflowType.DISPLAYPORT_CABLE_INSERTION.value}') from exc


def get_workflow_nodes(core_config: CoreConfig) -> List[Node]:
    """
    Get workflow nodes for supported modes.

    Args
    ----
        workflow_config (WorkflowConfig): Workflow config

    Returns
    -------
        List[Node]: List of Nodes

    """
    workflow_nodes = []
    if core_config.workflow_config.workflow_type == WorkflowType.POSE_TO_POSE:
        workflow_nodes = _get_internal_workflows(WorkflowType.POSE_TO_POSE).get_pose_to_pose(
            workflow_config=core_config.workflow_config)
    elif core_config.workflow_config.workflow_type in (
        WorkflowType.PICK_AND_PLACE,
        WorkflowType.DISPLAYPORT_CABLE_INSERTION,
    ):
        if core_config.workflow_config.workflow_type == WorkflowType.PICK_AND_PLACE:
            workflow_nodes = _get_internal_workflows(
                WorkflowType.PICK_AND_PLACE).get_multi_object_pick_and_place(
                orchestration_config=core_config.workflow_config.orchestration_config)
        elif core_config.workflow_config.workflow_type == \
                WorkflowType.DISPLAYPORT_CABLE_INSERTION:
            workflow_nodes = workflow_utils_ext.get_pick_and_place_orchestrator(
                core_config.workflow_config)
            workflow_nodes += get_displayport_insertion_nodes(
                core_config.workflow_config, use_sim_time=core_config.use_sim_time)
            workflow_nodes += get_displayport_insertion_orchestrator(
                core_config.workflow_config)
        if not core_config.workflow_config.use_ground_truth_pose_in_sim and not getattr(
                core_config.workflow_config, 'use_manual_poses', False):
            workflow_nodes += get_object_detection_servers(
                camera_config=core_config.camera_config,
                pose_estimation_config=core_config.pose_estimation_config
            )
            workflow_nodes += get_object_selection_server(
                object_selection_config=core_config.object_selection_config
            )
            if core_config.pose_estimation_config.pose_estimation_type == \
                    PoseEstimationType.FOUNDATION_POSE:
                workflow_nodes += get_foundation_pose_nodes(
                    core_config.camera_config,
                    workflow_type=core_config.workflow_config.workflow_type,
                    pose_estimation_config=core_config.pose_estimation_config)
            elif core_config.pose_estimation_config.pose_estimation_type == \
                    PoseEstimationType.DOPE:
                raise NotImplementedError("Dope is not supported for pick and place")
    elif core_config.workflow_config.workflow_type == WorkflowType.OBJECT_FOLLOWING:
        workflow_nodes = _get_internal_workflows(
            WorkflowType.OBJECT_FOLLOWING).get_object_following(
            workflow_config=core_config.workflow_config)
        workflow_nodes += get_pose_estimation_nodes(core_config)
    else:
        raise NotImplementedError(f'Workflow type {core_config.workflow_config.workflow_type}'
                                  'not supported !!')

    return workflow_nodes


def get_manipulation_container(core_config: CoreConfig) -> Node:
    """
    Return manipulation container that allows multiple nodes to run in a single process.

    Args
    ----
        workflow_config (WorkflowConfig): Config for workflow

    Returns
    -------
        Node: Manipulation container

    """
    env_variables = dict(os.environ)

    if core_config.enable_cuda_mps:
        env_variables.update({
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': core_config.cuda_mps_active_thread_percentage_container,
            'CUDA_MPS_PIPE_DIRECTORY': core_config.cuda_mps_pipe_directory,
            'CUDA_MPS_CLIENT_PRIORITY': core_config.cuda_mps_client_priority_container
        })

    if not core_config.enable_nsight_profiling:
        manipulation_container = Node(
            name=constants.MANIPULATOR_CONTAINER_NAME,
            package='rclcpp_components',
            executable='component_container_mt',
            arguments=['--ros-args', '--log-level', core_config.log_level],
            parameters=[{'use_sim_time': core_config.use_sim_time}],
            on_exit=Shutdown(),
            env=env_variables
        )
    else:
        if core_config.enable_system_wide_profiling:
            # This requires a sudo command so that Nsight can see multiple processes
            prefix = os.path.join(
                get_package_share_directory('isaac_ros_manipulation_bringup'),
                'params', 'nsys_wrapper_for_system_visibility.sh'
            )
        else:
            prefix = f'nsys profile --trace=osrt,nvtx,cuda --delay' \
                f'{core_config.delay_to_start_nsight}' \
                f'--duration {core_config.nsight_profile_duration}  --stats=true ' \
                f'-o {core_config.nsight_profile_output_file_path}'

        manipulation_container = Node(
            name=constants.MANIPULATOR_CONTAINER_NAME,
            package='rclcpp_components',
            executable='component_container_mt',
            arguments=['--ros-args', '--log-level', core_config.log_level],
            parameters=[{'use_sim_time': core_config.use_sim_time}],
            prefix=prefix,
            sigterm_timeout="1000",
            on_exit=Shutdown(),
        )

    return manipulation_container


def get_joint_state_publisher(use_sim_time: bool) -> Node:
    """
    Return joint state publisher that publishes TF of joints of a robot arm.

    Args
    ----
        workflow_config (WorkflowConfig): Workflow config

    Returns
    -------
        Node: Joint state publisher node

    """
    return Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        on_exit=Shutdown(),
    )


def get_cumotion_node(
    workflow_type: WorkflowType,
    camera_type: CameraType,
    xrdf_file_path: str,
    urdf_file_path: str,
    distance_threshold: float,
    num_cameras: int,
    time_sync_slop: str,
    use_sim_time: bool,
    setup: str,
    read_esdf_world: bool,
    core_config: CoreConfig
) -> Node:
    """
    Get cumotion node.

    Args
    ----
        camera_type (CameraType): The camera type to be used
        asset_name (str): Asset to be moved

    Returns
    -------
        Node: The cumotion launch node

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )
    use_sim_time_str = 'true' if use_sim_time else 'false'
    tool_frame = getattr(core_config.workflow_config, 'gripper_frame', 'gripper_frame')
    enable_object_attachment = 'true' if workflow_type in (
        WorkflowType.PICK_AND_PLACE,
        WorkflowType.DISPLAYPORT_CABLE_INSERTION,
    ) else 'false'
    enable_dnn_depth_in_realsense = 'true' if \
        core_config.depth_estimation_config.enable_dnn_depth_in_realsense else 'false'

    if workflow_type in (
        WorkflowType.PICK_AND_PLACE,
        WorkflowType.DISPLAYPORT_CABLE_INSERTION,
    ):
        object_attachment_type = core_config.workflow_config.object_attachment_type.value
    else:
        object_attachment_type = ''

    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/cumotion.launch.py']
        ),
        launch_arguments={
            'use_sim_time': use_sim_time_str,
            'tool_frame': tool_frame,
            'camera_type': str(camera_type),
            'joint_states_topic': core_config.cumotion_config.cumotion_joint_states_topic,
            'time_sync_slop': time_sync_slop,
            'distance_threshold': str(distance_threshold),
            'time_dilation_factor': str(core_config.workflow_config.time_dilation_factor),
            'urdf_file_path': urdf_file_path,
            'xrdf_file_path': xrdf_file_path,
            # This allows for nvblox to work well with Isaac Sim and not reject depth frames
            # from robot segmentor due to DDS drops/fluctuations.
            'qos_setting': 'DEFAULT' if use_sim_time else 'SENSOR_DATA',
            'enable_object_attachment': enable_object_attachment,
            'object_attachment_type': object_attachment_type,
            'workspace_bounds_name': setup,
            'num_cameras': str(num_cameras),
            'enable_cuda_mps': 'true' if core_config.enable_cuda_mps else 'false',
            'cuda_mps_pipe_directory': core_config.cuda_mps_pipe_directory,
            'cuda_mps_client_priority_robot_segmenter':
                core_config.cuda_mps_client_priority_robot_segmenter,
            'cuda_mps_active_thread_percentage_robot_segmenter':
                core_config.cuda_mps_active_thread_percentage_robot_segmenter,
            'cuda_mps_client_priority_planner':
                core_config.cuda_mps_client_priority_planner,
            'cuda_mps_active_thread_percentage_planner':
                core_config.cuda_mps_active_thread_percentage_planner,
            'enable_dnn_depth_in_realsense': enable_dnn_depth_in_realsense,
            'nvblox_global_frame': core_config.nvblox_global_frame,
            'moveit_collision_objects_scene_file': core_config.cumotion_config.moveit_collision_objects_scene_file,
            'read_esdf_world': 'true' if read_esdf_world else 'false',
        }.items(),
    )

    return cumotion_launch


def get_nvblox_node(camera_type: str, use_sim_time: bool,
                    setup: str, num_cameras: str,
                    enable_dnn_depth_in_realsense: bool,
                    nvblox_global_frame: str) -> Node:
    """
    Get nvblox node.

    Args
    ----
        camera_type (CameraType): The camera type to be used

    Returns
    -------
        Node: The nvblox launch nodes

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )
    nvblox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_files_include_dir, '/nvblox.launch.py']),
        launch_arguments={
            'camera_type': str(camera_type),
            'use_sim_time': 'true' if use_sim_time else 'false',
            'workspace_bounds_name': setup,
            'num_cameras': str(num_cameras),
            'enable_dnn_depth_in_realsense': 'true' if enable_dnn_depth_in_realsense else 'false',
            'nvblox_global_frame': nvblox_global_frame,
        }.items(),
    )
    return nvblox_launch
