# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Union

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils.config import (
    DisplayPortInsertionConfig,
    get_home_joint_positions_from_env_yaml,
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import RobotType
from launch.actions import IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from sensor_msgs.msg import JointState
import yaml


def get_displayport_insertion_nodes(
    displayport_config: DisplayPortInsertionConfig,
    use_sim_time: bool,
) -> List[Union[Node, IncludeLaunchDescription]]:
    """Launch the DisplayPort insertion policy through Isaac ROS Deploy."""
    if not displayport_config.requires_deploy:
        return []

    if displayport_config.use_task_space_policy:
        return _get_task_space_insertion_nodes(displayport_config)

    launch_file_path = os.path.join(
        get_package_share_directory('isaac_ros_deploy_reference_applications'),
        'launch',
        'displayport_insertion_deploy_controllers.launch.py')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file_path),
            launch_arguments={
                'robot_sn': displayport_config.driver_config.robot_sn,
                'config_path': displayport_config.deploy_config_path,
                'goal_pose_topic': displayport_config.goal_pose_topic,
                'decimation': str(displayport_config.deploy_decimation),
                'inference_controller_name':
                    displayport_config.deploy_inference_controller_name,
                'safety_controller_name':
                    displayport_config.deploy_safety_controller_name,
                'start_inactive': 'true',
                'controller_manager_timeout_s':
                    str(displayport_config.deploy_controller_switch_timeout_sec),
                'hardware_command_interfaces': os.environ.get(
                    'DISPLAYPORT_INSERTION_DEPLOY_HARDWARE_COMMAND_INTERFACES',
                    '["position"]'),
                'blend_ratio': '0.0',
                'max_blend_ratio_speed':
                    str(displayport_config.deploy_max_blend_ratio_speed),
                'use_sim_time': 'true' if use_sim_time else 'false',
            }.items(),
        )
    ]


def _get_task_space_insertion_nodes(
    displayport_config: DisplayPortInsertionConfig,
) -> List[Union[Node, IncludeLaunchDescription]]:
    """
    Launch the task-space policy graph that commands the Flexiv Cartesian sink.

    Unlike the joint-space path, this brings up the pose source, input builder,
    Triton, the pose-delta command builder/decoder, and the Flexiv passthrough
    rather than ros2_control Deploy controllers. The graph comes up idle: it
    holds observations until the orchestrator publishes a socket pose, and the
    passthrough forwards nothing until the orchestrator opens its enable gate.
    """
    launch_file_path = os.path.join(
        get_package_share_directory('isaac_ros_deploy_reference_applications'),
        'launch',
        'displayport_task_space_flexiv_robot.launch.py')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file_path),
            launch_arguments={
                'robot_sn': displayport_config.driver_config.robot_sn,
                'config_path': displayport_config.deploy_config_path,
                'socket_pose_topic': displayport_config.task_space_socket_pose_topic,
                'adapter_enable_topic': displayport_config.task_space_enable_topic,
                # The orchestrator owns both gates, so nothing moves at bring-up.
                'adapter_enabled_on_start': 'false',
                'require_socket_pose': 'true',
                'tcp_offset': displayport_config.task_space_tcp_offset,
                'adapter_tcp_offset':
                    displayport_config.task_space_adapter_tcp_offset,
                'action_blend_ratio':
                    str(displayport_config.task_space_action_blend_ratio),
                'publish_rate': str(displayport_config.goal_pose_publish_rate_hz),
                'base_frame': displayport_config.driver_config.base_frame,
                # The driver is in a joint mode at bring-up, so impedance cannot be set
                # here; the orchestrator sets it after the controller switch.
                'set_cartesian_impedance_on_start': 'false',
            }.items(),
        )
    ]


def parse_joint_state_from_yaml(
    yaml_file_path: str,
    use_sim_time: bool,
    joint_names: List[str],
) -> JointState:
    """Parse initial joint positions from a policy env.yaml file."""
    del use_sim_time  # retained for signature compatibility; not needed here
    with open(yaml_file_path, 'r') as file:
        env_yaml = yaml.load(file, Loader=yaml.UnsafeLoader)

    home_target_position = get_home_joint_positions_from_env_yaml(env_yaml, yaml_file_path)
    if len(home_target_position) != len(joint_names):
        raise ValueError(
            f'Home joint positions in {yaml_file_path} has {len(home_target_position)} values '
            f'but robot has {len(joint_names)} joints.')

    joint_state = JointState()
    joint_state.position = home_target_position
    joint_state.velocity = [0.0] * len(home_target_position)
    joint_state.effort = [0.0] * len(home_target_position)
    joint_state.name = list(joint_names)
    return joint_state


def get_displayport_insertion_orchestrator(
        displayport_config: DisplayPortInsertionConfig) -> List[Node]:
    """Launch the DisplayPort cable insertion orchestrator."""
    if displayport_config.use_sim_time:
        camera_tf_name = 'front_stereo_camera_left'
    else:
        camera_tf_name = 'camera_1_color_optical_frame'

    if displayport_config.driver_config.robot_type == RobotType.UR:
        socket_pose_base_frame = 'base'
    else:
        socket_pose_base_frame = displayport_config.driver_config.base_frame

    use_manual_poses = displayport_config.use_manual_poses

    return [
        Node(
            package='isaac_ros_manipulation_contact_rich_insertion',
            executable='displayport_insertion_orchestrator.py',
            name='displayport_insertion_orchestrator',
            parameters=[{
                'use_sim_time': displayport_config.use_sim_time,
                'wait_for_point_topic': not use_manual_poses,
                'point_topic_name_as_trigger': 'input_points_debug',
                'max_timeout_time_for_action_call':
                    displayport_config.timeout_for_insertion_action_call,
                'is_segment_anything_segmentation_enabled': not use_manual_poses,
                'use_manual_poses': use_manual_poses,
                'manual_plug_pose': displayport_config.manual_plug_pose,
                'manual_socket_pose': displayport_config.manual_socket_pose,
                'plug_orientation_correction': displayport_config.plug_orientation_correction,
                'socket_orientation_correction':
                    displayport_config.socket_orientation_correction,
                'socket_mate_offset': displayport_config.socket_mate_offset,
                'socket_mesh_file_path': displayport_config.socket_mesh_file_path,
                'cable_mesh_file_path': displayport_config.cable_mesh_file_path,
                'manual_grasp_mode': displayport_config.manual_grasp_mode,
                'camera_prim_name_in_tf': camera_tf_name,
                'run_rl_inference': displayport_config.run_rl_inference,
                'goal_pose_topic': displayport_config.goal_pose_topic,
                'deploy_inference_controller_name':
                    displayport_config.deploy_inference_controller_name,
                'deploy_safety_controller_name':
                    displayport_config.deploy_safety_controller_name,
                'deploy_normal_controller_name':
                    displayport_config.deploy_normal_controller_name,
                'deploy_controller_switch_timeout_sec':
                    displayport_config.deploy_controller_switch_timeout_sec,
                'deploy_blend_ratio': displayport_config.deploy_blend_ratio,
                'deploy_max_blend_ratio_speed':
                    displayport_config.deploy_max_blend_ratio_speed,
                'deploy_pre_goal_wait_sec': displayport_config.deploy_pre_goal_wait_sec,
                'deploy_external_policy': displayport_config.deploy_external_policy,
                'goal_pose_publish_rate_hz': displayport_config.goal_pose_publish_rate_hz,
                'use_joint_space_planner': displayport_config.use_joint_space_planner,
                'target_joint_state_for_place_pose':
                    displayport_config.home_joint_state.position,
                'output_dir': displayport_config.output_dir,
                'offset_for_hover_pose': displayport_config.offset_for_hover_pose,
                'hover_pose_position': displayport_config.hover_pose_position,
                'override_hover_position': displayport_config.override_hover_position,
                'hover_pose_orientation': displayport_config.hover_pose_orientation,
                'base_frame': displayport_config.driver_config.base_frame,
                'socket_pose_base_frame': socket_pose_base_frame,
                'socket_tf_frame': displayport_config.socket_tf_frame,
                'gripper_frame': displayport_config.driver_config.gripper_frame,
                'grasp_frame': displayport_config.driver_config.grasp_frame,
                'gripper_action_name': displayport_config.driver_config.gripper_action_name,
                'gripper_open_position': displayport_config.driver_config.gripper_open_position,
                'gripper_close_position': displayport_config.driver_config.gripper_close_position,
                'gripper_max_effort': displayport_config.driver_config.gripper_max_effort,
                'gripper_settle_time_sec':
                    displayport_config.driver_config.gripper_settle_time_sec,
                'pre_deploy_close_gripper':
                    displayport_config.pre_deploy_close_gripper,
                'pre_deploy_gripper_close_position':
                    displayport_config.pre_deploy_gripper_close_position,
                'post_deploy_open_gripper':
                    displayport_config.post_deploy_open_gripper,
                'post_deploy_pre_open_wait_sec':
                    displayport_config.post_deploy_pre_open_wait_sec,
                'post_deploy_open_wait_sec':
                    displayport_config.post_deploy_open_wait_sec,
                'post_deploy_retract_to_hover':
                    displayport_config.post_deploy_retract_to_hover,
                'post_deploy_retreat_lift_distance':
                    displayport_config.post_deploy_retreat_lift_distance,
                'enable_controller_switching': displayport_config.enable_controller_switching,
                'use_task_space_policy': displayport_config.use_task_space_policy,
                'task_space_socket_pose_topic': (
                    displayport_config.task_space_socket_pose_topic
                    if displayport_config.use_task_space_policy else ''),
                'task_space_enable_topic': (
                    displayport_config.task_space_enable_topic
                    if displayport_config.use_task_space_policy else ''),
                'task_space_cartesian_controller_name': (
                    displayport_config.task_space_cartesian_controller_name
                    if displayport_config.use_task_space_policy else ''),
                'task_space_cartesian_impedance_service_name':
                    displayport_config.task_space_cartesian_impedance_service_name,
                'task_space_cartesian_stiffness':
                    displayport_config.task_space_cartesian_stiffness,
                'task_space_cartesian_damping_ratio':
                    displayport_config.task_space_cartesian_damping_ratio,
                'task_space_release_after_insertion':
                    displayport_config.task_space_release_after_insertion,
                'task_space_return_home_after_insertion':
                    displayport_config.task_space_return_home_after_insertion,
                'task_space_home_pose': displayport_config.task_space_home_pose,
                'time_dilation_factor': displayport_config.time_dilation_factor,
                'joint_names': displayport_config.driver_config.joint_names,
                'joint_limits_file_path':
                    displayport_config.driver_config.joint_limits_file_path,
            }],
            output='screen',
            on_exit=Shutdown(),
        )
    ]
