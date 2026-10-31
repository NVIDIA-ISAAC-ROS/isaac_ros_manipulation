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

"""
Real-robot Flexiv Rizon driver launch with MoveIt + cuMotion.

Reuses configs from the flexiv_ros2 third-party package:
  - URDF from flexiv_description (with FlexivHardwareInterface or mock)
  - SRDF, kinematics, joint_limits from flexiv_moveit_config
  - ros2_control controllers from flexiv_bringup
  - Gripper launch from flexiv_gripper

Adds cuMotion planning pipeline from isaac_ros_cumotion_moveit on top of
the MoveIt move_group node, and publishes the static TFs the Isaac
Manipulation stack expects.
"""

from isaac_ros_manipulation_flexiv_driver_utils import (
    FlexivDriverUtils, FlexivRizonDriverConfig,
)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def launch_setup(context, *args, **kwargs):
    driver_config = FlexivRizonDriverConfig(context)
    flexiv = FlexivDriverUtils(driver_config)

    # Pre-process cuMotion URDF/XRDF to match Flexiv's <robot_sn>_ prefixing.
    flexiv.apply_real_cumotion_urdf_prefix()

    # Always-on nodes: state publisher, control nodes (non-deferred), static TFs.
    always_on = [flexiv.get_robot_state_publisher()]
    always_on += flexiv.get_robot_control_nodes()
    always_on += flexiv.get_real_static_tfs()

    # Deferred nodes -- built here but added to the launch description only
    # through the event handlers below, which sequence the bring-up.
    move_group_node, moveit_config = flexiv.get_moveit_group_node()
    rviz_nodes = flexiv.get_real_rviz_nodes(moveit_config)
    gripper_launch = flexiv.get_real_gripper_launch()

    event_handlers = flexiv.get_real_event_handlers(
        move_group_node=move_group_node,
        rviz_nodes=rviz_nodes,
        gripper_launch=gripper_launch,
    )
    return always_on + event_handlers


def generate_launch_description():
    declared_arguments = [
        # Core robot identity.
        DeclareLaunchArgument(
            'rizon_type',
            description='Type of the Flexiv Rizon robot.',
            default_value='Rizon4s',
            choices=[
                'Rizon4', 'Rizon4M', 'Rizon4R', 'Rizon4s',
                'Rizon10', 'Rizon10s',
            ],
        ),
        DeclareLaunchArgument(
            'robot_sn',
            description='Serial number of the robot to connect to.',
        ),
        DeclareLaunchArgument(
            'robot_type',
            default_value='FLEXIV',
            choices=['UR', 'FLEXIV'],
            description='Robot family used to drive TF frame prefix and '
                        'arm joint name derivation in shared launch utilities.',
        ),
        DeclareLaunchArgument(
            'gripper_type',
            default_value='grav',
            choices=['grav'],
            description='Type of gripper mounted on the Flexiv Rizon.',
        ),

        # Real-robot runtime args.
        DeclareLaunchArgument(
            'rdk_control_mode',
            default_value='joint_position',
            description='RDK control mode.',
            choices=['joint_position', 'joint_impedance'],
        ),
        DeclareLaunchArgument(
            'runtime_cartesian_switching',
            default_value='false',
            description='Prepare Cartesian mode at activation so a task-space '
                        'policy can take Cartesian command authority at runtime '
                        'without restarting the driver. Zeroes the force/torque '
                        'sensor at bring-up, so nothing may be in contact with '
                        'the robot.',
        ),
        DeclareLaunchArgument(
            'load_gripper',
            default_value='false',
            description='Load the Flexiv Grav gripper.',
        ),
        DeclareLaunchArgument(
            'gripper_name',
            default_value='Flexiv-GN01',
            description='Name of the gripper.',
        ),
        DeclareLaunchArgument(
            'load_mounted_ft_sensor',
            default_value='false',
            description='Load the mounted force torque sensor.',
        ),
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Use mock hardware instead of real robot.',
        ),
        DeclareLaunchArgument(
            'fake_sensor_commands',
            default_value='false',
            description='Enable fake sensor command interfaces.',
        ),

        # RViz gating.
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Start RViz automatically (MoveIt config).',
        ),
        DeclareLaunchArgument(
            'enable_rviz_visualization',
            default_value='false',
            description='Launch workflow visualization RViz.',
        ),
        DeclareLaunchArgument(
            'rviz_config_file',
            default_value='',
            description='Path to the RViz config file for workflow visualization.',
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
