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

from isaac_ros_launch_utils import GroupAction

from isaac_ros_manipulation_flexiv_driver_utils import (
    FlexivDriverUtils, FlexivRizonDriverConfig,
    get_grav_gripper_node, get_isaac_sim_joint_parser_node,
)
from isaac_ros_manipulation_ros_python_utils.config import CoreConfig
from isaac_ros_manipulation_ros_python_utils.core import (
    get_visualization_actions
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)


def launch_setup(context, *args, **kwargs):
    driver_config = FlexivRizonDriverConfig(context)
    flexiv = FlexivDriverUtils(driver_config)

    manipulator_init_nodes = []
    core_config = CoreConfig(context)
    if driver_config.use_sim_time:
        manipulator_init_nodes.append(
            get_isaac_sim_joint_parser_node(driver_config.use_sim_time))
        manipulator_init_nodes.append(
            get_grav_gripper_node(driver_config.use_sim_time))
    manipulator_init_nodes.append(flexiv.get_robot_state_publisher())
    ros2_control_nodes = flexiv.get_robot_control_nodes()
    move_group_node, moveit_config = flexiv.get_moveit_group_node()
    manipulator_init_nodes.extend(
        get_visualization_actions(
            core_config=core_config,
            moveit_config=moveit_config
        )
    )
    return manipulator_init_nodes + ros2_control_nodes + [move_group_node]


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'log_level',
            default_value='error',
            description='Log level of the container.',
            choices=['debug', 'info', 'warn', 'error']
        ),
        DeclareLaunchArgument(
            'rizon_type',
            description='Type of Flexiv Rizon robot.',
            choices=['Rizon4', 'Rizon4s', 'Rizon4M', 'Rizon4R',
                     'Rizon10', 'Rizon10s'],
        ),
        DeclareLaunchArgument(
            'controller_spawner_timeout',
            default_value='10',
            description='Timeout used when spawning controllers.',
        ),
        DeclareLaunchArgument(
            'joint_limits_file_path',
            description='Joint limits file path',
        ),
        DeclareLaunchArgument(
            'kinematics_file_path',
            description='Kinematics file path',
        ),
        DeclareLaunchArgument(
            'moveit_controllers_file_path',
            description='MoveIt controller config file path',
        ),
        DeclareLaunchArgument(
            'ros2_controllers_file_path',
            description='ROS2 control controller config file path',
        ),
        DeclareLaunchArgument(
            'workflow_type',
            choices=['POSE_TO_POSE', 'PICK_AND_PLACE',
                     'OBJECT_FOLLOWING', 'DISPLAYPORT_CABLE_INSERTION'],
            description='Type of workflow to run',
        ),
        DeclareLaunchArgument(
            'robot_type',
            choices=['UR', 'FLEXIV'],
            description='Robot family used to drive TF frame prefix and '
                        'arm joint name derivation in shared launch utilities.',
        ),
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )

    return LaunchDescription(launch_args + [group_action])
