#!/usr/bin/env python3

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
"""Orchestration node for DisplayPort cable insertion in Isaac Manipulator."""

import math
import os
from threading import Event
import time
from typing import Any, Dict, Iterable, List, Tuple

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from controller_manager_msgs.srv import SwitchController
try:
    from flexiv_msgs.srv import SetCartesianImpedance
except ImportError:  # pragma: no cover - optional outside the Flexiv workspace
    SetCartesianImpedance = None
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_manipulation_interfaces.action import (
    AddSegmentationMask, DisplayPortCableInsertion, GetObjectPose, GetObjects, PickAndPlace
)
from isaac_ros_manipulation_interfaces.srv import (
    AddMeshToObject, AssignNameToObject, ClearObjects
)
import isaac_ros_manipulation_ros_python_utils.geometry as geometry_utils
import isaac_ros_manipulation_ros_python_utils.planning_utils as planning_utils
from moveit_msgs.action import ExecuteTrajectory
import numpy as np
from rcl_interfaces.msg import Parameter as RCLPYParameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as RCLPYNode
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header, String
import tf2_ros


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


class IsaacRosManipulationDisplayPortInsertionOrchestrator(RCLPYNode):
    """
    Orchestrator for DisplayPort cable insertion.

    This node performs socket and cable perception using SAM with initial hints,
    updates Foundation Pose with mesh files, executes pick-and-place to a hover
    pose above the socket, and optionally runs Deploy insertion.
    """

    def __init__(self):
        super().__init__('displayport_insertion_orchestrator')

        # use_sim_time is auto-declared by rclpy when passed via launch params.
        self.declare_parameter('wait_for_point_topic', False)
        self.declare_parameter('point_topic_name_as_trigger', '')
        self.declare_parameter('max_timeout_time_for_action_call', 10.0)
        self.declare_parameter('is_segment_anything_segmentation_enabled', False)
        self.declare_parameter('socket_mesh_file_path', '')
        self.declare_parameter('cable_mesh_file_path', '')
        self.declare_parameter('manual_grasp_mode', True)
        self.declare_parameter('use_manual_poses', False)
        self.declare_parameter('manual_plug_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('manual_socket_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('plug_orientation_correction', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('socket_orientation_correction', [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('socket_mate_offset', [0.0, 0.0, 0.0])
        self.declare_parameter('camera_prim_name_in_tf', 'camera_color_optical_frame')
        self.declare_parameter('run_rl_inference', False)
        self.declare_parameter('use_joint_space_planner', False)
        self.declare_parameter('target_joint_state_for_place_pose',
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('output_dir', ISAAC_ROS_WS)
        self.declare_parameter('offset_for_hover_pose', 0.08)
        self.declare_parameter('hover_pose_position', [0.0, 0.0, 0.0])
        self.declare_parameter('override_hover_position', False)
        self.declare_parameter('hover_pose_orientation', [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('socket_pose_base_frame', 'base_link')
        self.declare_parameter('socket_tf_frame', 'displayport_socket_frame')
        self.declare_parameter('gripper_frame', 'gripper_frame')
        self.declare_parameter('grasp_frame', 'grasp_frame')
        self.declare_parameter('gripper_action_name', '/robotiq_gripper_controller/gripper_cmd')
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('gripper_close_position', 0.65)
        self.declare_parameter('gripper_max_effort', 10.0)
        self.declare_parameter('gripper_settle_time_sec', 0.0)
        self.declare_parameter('pre_deploy_close_gripper', False)
        self.declare_parameter('pre_deploy_gripper_close_position', -0.065)
        self.declare_parameter('post_deploy_open_gripper', False)
        self.declare_parameter('post_deploy_pre_open_wait_sec', 0.0)
        self.declare_parameter('post_deploy_open_wait_sec', 0.0)
        self.declare_parameter('post_deploy_retract_to_hover', False)
        self.declare_parameter('post_deploy_retreat_lift_distance', 0.0)
        self.declare_parameter('enable_controller_switching', True)
        self.declare_parameter('use_task_space_policy', False)
        self.declare_parameter('task_space_socket_pose_topic', '')
        self.declare_parameter('task_space_enable_topic', '')
        self.declare_parameter('task_space_cartesian_controller_name', '')
        self.declare_parameter('task_space_cartesian_impedance_service_name', '')
        self.declare_parameter('task_space_cartesian_stiffness', [0.0])
        self.declare_parameter('task_space_cartesian_damping_ratio', [0.0])
        self.declare_parameter('task_space_release_after_insertion', False)
        self.declare_parameter('task_space_return_home_after_insertion', False)
        self.declare_parameter('task_space_home_pose', [0.0])
        self.declare_parameter('task_space_enable_after_cycles', 2)
        self.declare_parameter('time_dilation_factor', 0.6)
        self.declare_parameter('joint_names', [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint',
        ])
        self.declare_parameter('joint_limits_file_path', '')
        self.declare_parameter('goal_pose_topic', '/displayport_insertion/goal_pose')
        self.declare_parameter('deploy_inference_controller_name', 'deploy_inference_controller')
        self.declare_parameter('deploy_safety_controller_name', 'deploy_safety_controller')
        self.declare_parameter('deploy_normal_controller_name', 'rizon_arm_controller')
        self.declare_parameter('deploy_controller_switch_timeout_sec', 10.0)
        self.declare_parameter('deploy_blend_ratio', 1.0)
        self.declare_parameter('deploy_max_blend_ratio_speed', 1.0)
        self.declare_parameter('deploy_pre_goal_wait_sec', 0.25)
        self.declare_parameter('deploy_external_policy', False)
        self.declare_parameter('goal_pose_publish_rate_hz', 30.0)
        self.declare_parameter('sim_control_mode_topic', '/displayport_insertion/control_mode')

        self._use_sim_time = self.get_parameter(
            'use_sim_time').get_parameter_value().bool_value
        self._wait_for_point_topic = self.get_parameter(
            'wait_for_point_topic').get_parameter_value().bool_value
        self._point_topic_name_as_trigger = self.get_parameter(
            'point_topic_name_as_trigger').get_parameter_value().string_value
        self._max_timeout_time_for_action_call = self.get_parameter(
            'max_timeout_time_for_action_call').get_parameter_value().double_value
        self._is_segment_anything_segmentation_enabled = self.get_parameter(
            'is_segment_anything_segmentation_enabled').get_parameter_value().bool_value
        self._socket_mesh_file_path = self.get_parameter(
            'socket_mesh_file_path').get_parameter_value().string_value
        self._cable_mesh_file_path = self.get_parameter(
            'cable_mesh_file_path').get_parameter_value().string_value
        self._manual_grasp_mode = self.get_parameter(
            'manual_grasp_mode').get_parameter_value().bool_value
        self._use_manual_poses = self.get_parameter(
            'use_manual_poses').get_parameter_value().bool_value
        self._manual_plug_pose = list(self.get_parameter(
            'manual_plug_pose').get_parameter_value().double_array_value)
        self._manual_socket_pose = list(self.get_parameter(
            'manual_socket_pose').get_parameter_value().double_array_value)
        self._plug_orientation_correction_values = list(self.get_parameter(
            'plug_orientation_correction').get_parameter_value().double_array_value)
        self._socket_orientation_correction_values = list(self.get_parameter(
            'socket_orientation_correction').get_parameter_value().double_array_value)
        self._socket_mate_offset = list(self.get_parameter(
            'socket_mate_offset').get_parameter_value().double_array_value)
        self._camera_prim_name_in_tf = self.get_parameter(
            'camera_prim_name_in_tf').get_parameter_value().string_value
        self._run_rl_inference = self.get_parameter(
            'run_rl_inference').get_parameter_value().bool_value
        self._use_joint_space_planner = self.get_parameter(
            'use_joint_space_planner').get_parameter_value().bool_value
        self._target_joint_state_for_place_pose_arr_value = self.get_parameter(
            'target_joint_state_for_place_pose').get_parameter_value().double_array_value
        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._offset_for_hover_pose = self.get_parameter(
            'offset_for_hover_pose').get_parameter_value().double_value
        self._hover_pose_position = list(self.get_parameter(
            'hover_pose_position').get_parameter_value().double_array_value)
        self._override_hover_position = self.get_parameter(
            'override_hover_position').get_parameter_value().bool_value
        self._hover_pose_orientation = list(self.get_parameter(
            'hover_pose_orientation').get_parameter_value().double_array_value)
        self._base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self._socket_pose_base_frame = self.get_parameter(
            'socket_pose_base_frame').get_parameter_value().string_value
        self._socket_tf_frame = self.get_parameter(
            'socket_tf_frame').get_parameter_value().string_value
        self._gripper_frame = self.get_parameter(
            'gripper_frame').get_parameter_value().string_value
        self._grasp_frame = self.get_parameter(
            'grasp_frame').get_parameter_value().string_value
        self._gripper_action_name = self.get_parameter(
            'gripper_action_name').get_parameter_value().string_value
        self._gripper_open_position = self.get_parameter(
            'gripper_open_position').get_parameter_value().double_value
        self._gripper_close_position = self.get_parameter(
            'gripper_close_position').get_parameter_value().double_value
        self._gripper_max_effort = self.get_parameter(
            'gripper_max_effort').get_parameter_value().double_value
        self._gripper_settle_time_sec = self.get_parameter(
            'gripper_settle_time_sec').get_parameter_value().double_value
        self._pre_deploy_close_gripper = self.get_parameter(
            'pre_deploy_close_gripper').get_parameter_value().bool_value
        self._pre_deploy_gripper_close_position = self.get_parameter(
            'pre_deploy_gripper_close_position').get_parameter_value().double_value
        self._post_deploy_open_gripper = self.get_parameter(
            'post_deploy_open_gripper').get_parameter_value().bool_value
        self._post_deploy_pre_open_wait_sec = self.get_parameter(
            'post_deploy_pre_open_wait_sec').get_parameter_value().double_value
        self._post_deploy_open_wait_sec = self.get_parameter(
            'post_deploy_open_wait_sec').get_parameter_value().double_value
        self._post_deploy_retract_to_hover = self.get_parameter(
            'post_deploy_retract_to_hover').get_parameter_value().bool_value
        self._post_deploy_retreat_lift_distance = self.get_parameter(
            'post_deploy_retreat_lift_distance').get_parameter_value().double_value
        self._enable_controller_switching = self.get_parameter(
            'enable_controller_switching').get_parameter_value().bool_value
        self._joint_names = list(self.get_parameter(
            'joint_names').get_parameter_value().string_array_value)
        self._joint_limits_file_path = self.get_parameter(
            'joint_limits_file_path').get_parameter_value().string_value
        self._goal_pose_topic = self.get_parameter(
            'goal_pose_topic').get_parameter_value().string_value
        self._deploy_inference_controller_name = self.get_parameter(
            'deploy_inference_controller_name').get_parameter_value().string_value
        self._deploy_safety_controller_name = self.get_parameter(
            'deploy_safety_controller_name').get_parameter_value().string_value
        self._deploy_normal_controller_name = self.get_parameter(
            'deploy_normal_controller_name').get_parameter_value().string_value
        self._deploy_controller_switch_timeout_sec = self.get_parameter(
            'deploy_controller_switch_timeout_sec').get_parameter_value().double_value
        self._deploy_blend_ratio = self.get_parameter(
            'deploy_blend_ratio').get_parameter_value().double_value
        self._deploy_max_blend_ratio_speed = self.get_parameter(
            'deploy_max_blend_ratio_speed').get_parameter_value().double_value
        self._deploy_pre_goal_wait_sec = self.get_parameter(
            'deploy_pre_goal_wait_sec').get_parameter_value().double_value
        self._deploy_external_policy = self.get_parameter(
            'deploy_external_policy').get_parameter_value().bool_value
        self._goal_pose_publish_rate_hz = self.get_parameter(
            'goal_pose_publish_rate_hz').get_parameter_value().double_value
        self._sim_control_mode_topic = self.get_parameter(
            'sim_control_mode_topic').get_parameter_value().string_value
        self._use_task_space_policy = self.get_parameter(
            'use_task_space_policy').get_parameter_value().bool_value
        self._task_space_socket_pose_topic = self.get_parameter(
            'task_space_socket_pose_topic').get_parameter_value().string_value
        self._task_space_enable_topic = self.get_parameter(
            'task_space_enable_topic').get_parameter_value().string_value
        self._task_space_cartesian_controller_name = self.get_parameter(
            'task_space_cartesian_controller_name').get_parameter_value().string_value
        self._task_space_cartesian_impedance_service_name = self.get_parameter(
            'task_space_cartesian_impedance_service_name').get_parameter_value().string_value
        self._task_space_cartesian_stiffness = list(self.get_parameter(
            'task_space_cartesian_stiffness').get_parameter_value().double_array_value)
        self._task_space_cartesian_damping_ratio = list(self.get_parameter(
            'task_space_cartesian_damping_ratio').get_parameter_value().double_array_value)
        self._task_space_release_after_insertion = self.get_parameter(
            'task_space_release_after_insertion').get_parameter_value().bool_value
        self._task_space_return_home_after_insertion = self.get_parameter(
            'task_space_return_home_after_insertion').get_parameter_value().bool_value
        self._task_space_home_pose = list(self.get_parameter(
            'task_space_home_pose').get_parameter_value().double_array_value)
        self._task_space_enable_after_cycles = self.get_parameter(
            'task_space_enable_after_cycles').get_parameter_value().integer_value
        self._time_dilation_factor = self.get_parameter(
            'time_dilation_factor').get_parameter_value().double_value

        if self._use_task_space_policy:
            missing = [
                name for name, value in (
                    ('task_space_socket_pose_topic', self._task_space_socket_pose_topic),
                    ('task_space_enable_topic', self._task_space_enable_topic),
                    ('task_space_cartesian_controller_name',
                     self._task_space_cartesian_controller_name),
                ) if not value
            ]
            if missing:
                raise ValueError(
                    'use_task_space_policy requires: ' + ', '.join(missing))

        if self._deploy_blend_ratio < 0.0 or self._deploy_blend_ratio > 1.0:
            raise ValueError('deploy_blend_ratio must be in the range [0.0, 1.0]')
        if self._deploy_max_blend_ratio_speed <= 0.0 and self._deploy_blend_ratio > 0.0:
            raise ValueError(
                'deploy_max_blend_ratio_speed must be positive when '
                'deploy_blend_ratio is non-zero')
        if self._deploy_pre_goal_wait_sec < 0.0:
            raise ValueError('deploy_pre_goal_wait_sec must be non-negative')
        if self._goal_pose_publish_rate_hz <= 0.0:
            raise ValueError('goal_pose_publish_rate_hz must be positive')
        if self._post_deploy_pre_open_wait_sec < 0.0:
            raise ValueError('post_deploy_pre_open_wait_sec must be non-negative')
        if self._post_deploy_open_wait_sec < 0.0:
            raise ValueError('post_deploy_open_wait_sec must be non-negative')
        if self._post_deploy_retreat_lift_distance < 0.0:
            raise ValueError('post_deploy_retreat_lift_distance must be non-negative')

        if len(self._target_joint_state_for_place_pose_arr_value) != len(self._joint_names):
            raise ValueError(
                f'target_joint_state_for_place_pose has '
                f'{len(self._target_joint_state_for_place_pose_arr_value)} '
                f'values but robot has {len(self._joint_names)} joints.')

        if self._use_manual_poses:
            if len(self._manual_plug_pose) != 7:
                raise ValueError(
                    'manual_plug_pose must contain 7 values [x, y, z, qx, qy, qz, qw]')
            if len(self._manual_socket_pose) != 7:
                raise ValueError(
                    'manual_socket_pose must contain 7 values [x, y, z, qx, qy, qz, qw]')

        self._plug_orientation_correction = self._quaternion_from_array(
            self._plug_orientation_correction_values, 'plug_orientation_correction')
        self._socket_orientation_correction = self._quaternion_from_array(
            self._socket_orientation_correction_values, 'socket_orientation_correction')
        if len(self._socket_mate_offset) != 3:
            raise ValueError('socket_mate_offset must contain 3 values [x, y, z]')
        if not all(math.isfinite(value) for value in self._socket_mate_offset):
            raise ValueError('socket_mate_offset values must be finite numbers')

        self._target_joint_state_for_place_pose = JointState()
        self._target_joint_state_for_place_pose.position = \
            self._target_joint_state_for_place_pose_arr_value
        self._target_joint_state_for_place_pose.name = list(self._joint_names)
        self._target_joint_state_for_place_pose.velocity = \
            [0.0] * len(self._target_joint_state_for_place_pose.position)
        self._target_joint_state_for_place_pose.effort = \
            [0.0] * len(self._target_joint_state_for_place_pose.position)

        self.setUpClients()

    def setUpClients(self) -> None:
        """Set up action/service clients, TF, and the DisplayPortCableInsertion action server."""
        self.failure_count = 0
        self.total_count = 0

        self._detection_client = ActionClient(self, GetObjects, '/get_objects')
        self._get_objects_goal_handle = None
        self._get_objects_result_received = False
        self._get_objects_result_future = None
        self._get_objects_detected_object_id = None

        self._get_object_pose_client = ActionClient(self, GetObjectPose, '/get_object_pose')
        self._get_object_pose_goal_handle = None
        self._get_object_pose_result_received = False
        self._get_object_pose_result_future = None

        self._segmentation_client = ActionClient(self, AddSegmentationMask,
                                                 '/add_segmentation_mask')
        self._segmentation_goal_handle = None
        self._segmentation_result_received = False
        self._segmentation_result_future = None

        self._add_mesh_client = self.create_client(AddMeshToObject, '/add_mesh_to_object')
        self._add_mesh_result_received = False

        self._clear_objects_client = self.create_client(ClearObjects, '/clear_objects')
        self._clear_objects_result_received = False

        self._assign_name_client = self.create_client(AssignNameToObject, '/assign_name_to_object')

        self.latest_pose_gotten = None
        self.object_pose_estimation_result = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._gripper_client = ActionClient(
            self, ParallelGripperCommand, self._gripper_action_name)
        self._gripper_done_event = Event()
        self._gripper_done_result = False

        self._goal_pose_pub = self.create_publisher(PoseStamped, self._goal_pose_topic, 10)
        self._sim_control_mode_pub = None
        if self._sim_control_mode_topic:
            self._sim_control_mode_pub = self.create_publisher(
                String, self._sim_control_mode_topic, 10)

        self._task_space_planner_client = None
        self._task_space_execute_client = None
        self._task_space_impedance_client = None
        if self._use_task_space_policy:
            if self._task_space_return_home_after_insertion:
                if len(self._task_space_home_pose) != 7:
                    raise ValueError(
                        'task_space_home_pose must contain 7 values '
                        '[x, y, z, qx, qy, qz, qw] when '
                        'task_space_return_home_after_insertion is set')
                self._task_space_planner_client = ActionClient(
                    self, MotionPlan, '/cumotion/motion_plan')
                self._task_space_execute_client = ActionClient(
                    self, ExecuteTrajectory, '/execute_trajectory')
            if self._task_space_cartesian_impedance_service_name:
                if SetCartesianImpedance is None:
                    raise RuntimeError(
                        'flexiv_msgs is unavailable, so the Cartesian impedance service '
                        'cannot be called. It ships with the flexiv_ros2 source repository '
                        'rather than as a Debian package; build it into the workspace, or '
                        'clear task_space_cartesian_impedance_service_name to skip this.')
                self._task_space_impedance_client = self.create_client(
                    SetCartesianImpedance,
                    self._task_space_cartesian_impedance_service_name)

        self._task_space_socket_pose_pub = None
        self._task_space_enable_pub = None
        if self._use_task_space_policy:
            # The socket pose arms the graph; the enable topic grants authority.
            self._task_space_socket_pose_pub = self.create_publisher(
                PoseStamped, self._task_space_socket_pose_topic, 10)
            self._task_space_enable_pub = self.create_publisher(
                Bool, self._task_space_enable_topic, 10)

        self._action_server = ActionServer(
            self, DisplayPortCableInsertion, '/displayport_cable_insertion',
            execute_callback=self.execute_callback)

    def _publish_sim_control_mode(self, mode: str, log: bool = True) -> None:
        """Tell the Isaac Sim direct-ROS graph which command source to consume."""
        if self._sim_control_mode_pub is None:
            return
        msg = String()
        msg.data = mode
        self._sim_control_mode_pub.publish(msg)
        if log:
            self.get_logger().info(f'Published sim control mode: {mode}')

    def _publish_sim_control_mode_for_wall_duration(
            self, mode: str, duration_sec: float = 0.5, rate_hz: float = 30.0) -> None:
        """Publish a sim control-mode transition long enough for Isaac Sim to observe it."""
        if self._sim_control_mode_pub is None:
            return
        end_time = time.monotonic() + duration_sec
        period_sec = 1.0 / rate_hz
        should_log = True
        while rclpy.ok() and time.monotonic() < end_time:
            self._publish_sim_control_mode(mode, log=should_log)
            should_log = False
            time.sleep(period_sec)

    def _set_controller_states(self,
                               controllers_to_activate: List[str] = None,
                               controllers_to_deactivate: List[str] = None,
                               timeout_sec: float = 10.0) -> bool:
        """Activate/deactivate ROS2 controllers via the controller manager."""
        if SwitchController is None:
            self.get_logger().error(
                'controller_manager_msgs not available. '
                'Install ros-humble-controller-manager-msgs')
            return False

        if controllers_to_activate is None:
            controllers_to_activate = []
        if controllers_to_deactivate is None:
            controllers_to_deactivate = []

        if not controllers_to_activate and not controllers_to_deactivate:
            self.get_logger().warning('No controllers specified to activate or deactivate')
            return True

        service_name = '/controller_manager/switch_controller'
        client = self.create_client(SwitchController, service_name)

        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Service {service_name} not available '
                                    f'after {timeout_sec} seconds')
            return False

        request = SwitchController.Request()
        request.activate_controllers = controllers_to_activate
        request.deactivate_controllers = controllers_to_deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=timeout_sec).to_msg()

        self.get_logger().info(
            f'Switching controllers - Activating: {controllers_to_activate}, '
            f'Deactivating: {controllers_to_deactivate}')

        try:
            future = client.call_async(request)

            while not future.done():
                time.sleep(0.1)

            if future.result() is None:
                self.get_logger().error('Failed to call controller switch service')
                return False

            result = future.result()
            if result.ok:
                self.get_logger().info('Successfully switched controllers')
                return True
            self.get_logger().error('Controller switch failed')
            return False

        except Exception as e:
            self.get_logger().error(f'Exception during controller switch: {str(e)}')
            return False
        finally:
            self.destroy_client(client)

    def switch_to_streaming_control(self, timeout_sec: float = 10.0) -> bool:
        """Switch from rizon_arm_controller to streaming_position_controller (Flexiv)."""
        return self._set_controller_states(
            controllers_to_activate=['streaming_position_controller'],
            controllers_to_deactivate=['rizon_arm_controller'],
            timeout_sec=timeout_sec
        )

    def switch_to_rizon_arm_control(self, timeout_sec: float = 10.0) -> bool:
        """Switch from streaming_position_controller to rizon_arm_controller (Flexiv)."""
        return self._set_controller_states(
            controllers_to_activate=['rizon_arm_controller'],
            controllers_to_deactivate=['streaming_position_controller'],
            timeout_sec=timeout_sec
        )

    def switch_to_deploy_control(self, timeout_sec: float = None) -> bool:
        """
        Hand arm command authority to the insertion policy.

        Task-space policies command the Flexiv Cartesian motion controller, so
        activating it is what makes the driver borrow RT_CARTESIAN_MOTION_FORCE.
        Joint-space policies activate the ros2_control Deploy controllers.
        """
        if timeout_sec is None:
            timeout_sec = self._deploy_controller_switch_timeout_sec
        if self._use_task_space_policy:
            return self._set_controller_states(
                controllers_to_activate=[
                    self._task_space_cartesian_controller_name],
                controllers_to_deactivate=[self._deploy_normal_controller_name],
                timeout_sec=timeout_sec
            )
        return self._set_controller_states(
            controllers_to_activate=[
                self._deploy_inference_controller_name,
                self._deploy_safety_controller_name,
            ],
            controllers_to_deactivate=[self._deploy_normal_controller_name],
            timeout_sec=timeout_sec
        )

    def switch_from_deploy_control(self, timeout_sec: float = None) -> bool:
        """Restore the nominal joint controller after insertion."""
        if timeout_sec is None:
            timeout_sec = self._deploy_controller_switch_timeout_sec
        if self._use_task_space_policy:
            return self._set_controller_states(
                controllers_to_activate=[self._deploy_normal_controller_name],
                controllers_to_deactivate=[
                    self._task_space_cartesian_controller_name],
                timeout_sec=timeout_sec
            )
        return self._set_controller_states(
            controllers_to_activate=[self._deploy_normal_controller_name],
            controllers_to_deactivate=[
                self._deploy_inference_controller_name,
                self._deploy_safety_controller_name,
            ],
            timeout_sec=timeout_sec
        )

    def _set_deploy_safety_parameters(self, parameters: Dict[str, float]) -> bool:
        """Set runtime parameters on the Deploy SafetyController."""
        if not parameters:
            return True

        controller_name = self._deploy_safety_controller_name.strip('/')
        service_name = f'/{controller_name}/set_parameters'
        param_client = self.create_client(SetParameters, service_name)

        if not param_client.wait_for_service(
                timeout_sec=self._deploy_controller_switch_timeout_sec):
            self.get_logger().error(
                f'Service {service_name} not available after '
                f'{self._deploy_controller_switch_timeout_sec} seconds')
            self.destroy_client(param_client)
            return False

        request = SetParameters.Request()
        for name, value in parameters.items():
            parameter = RCLPYParameter()
            parameter.name = name
            parameter.value = ParameterValue(
                double_value=float(value),
                type=rclpy.Parameter.Type.DOUBLE.value)
            request.parameters.append(parameter)

        try:
            future = param_client.call_async(request)
            response = self.wait_for_service_result(
                future, timeout=self._deploy_controller_switch_timeout_sec)
        finally:
            self.destroy_client(param_client)

        if response is None:
            self.get_logger().error(f'Failed to set Deploy safety parameters: {parameters}')
            return False

        for name, result in zip(parameters.keys(), response.results):
            if not result.successful:
                self.get_logger().error(
                    f'Failed to set Deploy safety parameter {name}: {result.reason}')
                return False

        self.get_logger().info(f'Set Deploy safety parameters: {parameters}')
        return True

    def open_gripper_after_deploy(self) -> bool:
        """Open the gripper after Deploy insertion before retreating."""
        if not self._send_gripper_command(
                self._gripper_open_position,
                'Opening gripper after Deploy',
                'Gripper open'):
            return False

        wait_sec = max(self._gripper_settle_time_sec, self._post_deploy_open_wait_sec)
        if wait_sec > 0.0:
            self.get_logger().info(f'Waiting {wait_sec:.3f}s after gripper open')
            time.sleep(wait_sec)
        return True

    def _blend_ramp_wait_sec(self, blend_ratio_delta: float) -> float:
        """Return the expected SafetyController blend-ramp duration in seconds."""
        if self._deploy_max_blend_ratio_speed <= 0.0:
            return 0.0
        return abs(blend_ratio_delta) / self._deploy_max_blend_ratio_speed

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            self.get_logger().error(f'Failed to get transform from tf: {str(e)}')
            return None

    def trigger_get_objects_goal(self, initial_hint: None | Point = None):
        """Send a goal to the object detection action server."""
        self.get_logger().info('Waiting for action server for object detection...')
        if not self._detection_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'Action server for object detection not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        if initial_hint is not None:
            goal_msg.initial_hint = initial_hint
            goal_msg.use_initial_hint = True

        self.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._detection_client.send_goal_async(
            goal_msg, feedback_callback=self.get_objects_feedback_callback)
        send_goal_future.add_done_callback(self.get_objects_goal_response_callback)

    def get_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_objects_goal_handle = future.result()
        if not self._get_objects_goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by the action server.')
        self._get_objects_result_future = self._get_objects_goal_handle.get_result_async()
        self._get_objects_result_future.add_done_callback(self.get_objects_result_callback)

    def get_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        object_ids = []
        if result.objects:
            self.get_logger().info('Objects detected, the objects id are below:')
            for object_obj in result.objects:
                self.get_logger().info(f'Object: {object_obj.object_id}')
                object_ids.append(object_obj.object_id)

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self._get_objects_detected_object_id = result.objects[0].object_id
            self._get_objects_result_received = True
        else:
            self.get_logger().error(f'Action failed with status: {status}')
            self._get_objects_result_received = False

    def trigger_get_object_pose_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.get_logger().info('Waiting for pose estimation action server...')
        if not self._get_object_pose_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Pose estimation action server not available')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._get_object_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.get_object_pose_feedback_callback)
        send_goal_future.add_done_callback(self.get_object_pose_goal_response_callback)

    def get_object_pose_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_object_pose_goal_handle = future.result()
        if not self._get_object_pose_goal_handle.accepted:
            self.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.get_logger().info('Pose estimation goal accepted by the action server.')
        self._get_object_pose_result_future = self._get_object_pose_goal_handle.get_result_async()
        self._get_object_pose_result_future.add_done_callback(self.get_object_pose_result_callback)

    def get_object_pose_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_object_pose_result_callback(self, future):
        """Handle the result from the action server."""
        status = future.result().status
        result = future.result().result
        self.get_logger().info(f'Pose estimation result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Object pose received for object id')
            self._get_object_pose_result_received = True
            self.latest_pose_gotten = result.object_pose
            self.object_pose_estimation_result = geometry_utils.ros_pose_to_ros_transform(
                result.object_pose)
        else:
            self.get_logger().error(f'Pose estimation action failed with status: {status}')
            self._get_object_pose_result_received = False

    def trigger_get_segmented_object_goal(self, object_id):
        """Send a goal to the object segmentation action server."""
        self.get_logger().info('Waiting for segmentation action server...')
        if not self._segmentation_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Segmentation action server not available after waiting')
            return

        goal_msg = AddSegmentationMask.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to get segmented object for ID: {object_id}')
        send_goal_future = self._segmentation_client.send_goal_async(
            goal_msg, feedback_callback=self.get_segmented_object_feedback_callback)
        send_goal_future.add_done_callback(self.get_segmented_object_goal_response_callback)

    def get_segmented_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._segmentation_goal_handle = future.result()
        if not self._segmentation_goal_handle.accepted:
            self.get_logger().error('Segmentation goal was rejected by the action server.')
            return

        self.get_logger().info('Segmentation goal accepted by the action server.')
        self._segmentation_result_future = self._segmentation_goal_handle.get_result_async()
        self._segmentation_result_future.add_done_callback(
            self.get_segmented_object_result_callback)

    def get_segmented_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Segmentation Feedback: {feedback_msg.feedback}')

    def get_segmented_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Segmented object received: {result}')
            self._segmentation_result_received = True
        else:
            self.get_logger().error(f'Segmentation action failed with status: {status}')
            self._segmentation_result_received = False

    def trigger_add_mesh_to_object_request(self, object_id, mesh_file_path):
        """Send an AddMeshToObject service request."""
        request = AddMeshToObject.Request()
        request.object_ids = [object_id]
        request.mesh_file_paths = [mesh_file_path]
        future = self._add_mesh_client.call_async(request)
        return future

    def send_clear_objects_request(self, object_ids=None):
        """Send a ClearObjects service request."""
        request = ClearObjects.Request()
        if object_ids:
            request.object_ids = object_ids
        future = self._clear_objects_client.call_async(request)
        return future

    def wait_for_result(self, result_var_name: str, timeout: float = 10.0):
        """Poll until the named result flag is set or timeout expires."""
        start_time = self.get_clock().now()
        while rclpy.ok():
            if result_var_name and getattr(self, result_var_name):
                return True

            time_now = self.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.failure_count += 1
                return False
            self.total_count += 1
            time.sleep(0.1)

    def wait_for_service_result(self, future, timeout=10.0):
        """Wait for an async service call to complete."""
        start_time = self.get_clock().now()
        while not future.done():
            time_now = self.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.get_logger().error('Timeout waiting for service result')
                return None
            time.sleep(0.1)
        return future.result()

    def _runtime_elapsed_sec(self, start_ros_time, start_wall_time: float) -> float:
        """Return elapsed runtime seconds, using sim time when configured."""
        if self._use_sim_time:
            elapsed_ros = (self.get_clock().now() - start_ros_time).nanoseconds / 1e9
            if elapsed_ros >= 0.0:
                return elapsed_ros
        return time.monotonic() - start_wall_time

    def _wait_runtime_duration(self, duration_sec: float) -> None:
        """Wait for a duration in sim time when use_sim_time is enabled."""
        if duration_sec <= 0.0:
            return
        start_ros_time = self.get_clock().now()
        start_wall_time = time.monotonic()
        while rclpy.ok():
            remaining_sec = duration_sec - self._runtime_elapsed_sec(
                start_ros_time, start_wall_time)
            if remaining_sec <= 0.0:
                return
            time.sleep(min(0.05, remaining_sec))

    def set_mesh_param(self, mesh_file_path: str) -> bool:
        """Set the mesh_file_path parameter for the foundationpose_node."""
        param_client = self.create_client(
            SetParameters,
            '/foundationpose_node/set_parameters'
        )

        if not param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(
                'Service /foundationpose_node/set_parameters not available'
                ' skipping mesh parameter setting')
            return False

        request = SetParameters.Request()
        parameter = RCLPYParameter()
        parameter.name = 'mesh_file_path'
        parameter.value = ParameterValue(
            string_value=mesh_file_path,
            type=rclpy.Parameter.Type.STRING.value)
        request.parameters = [parameter]

        future = param_client.call_async(request)
        while not future.done():
            time.sleep(0.1)

        if future.result() is None:
            self.get_logger().warning(f'Failed to set mesh_file_path to {mesh_file_path}')
            return False
        self.get_logger().debug(f'Successfully set mesh_file_path to {mesh_file_path}')
        return True

    def assign_name_to_object(self, object_id: int, name: str) -> bool:
        """Assign a TF frame name to a detected object via AssignNameToObject service."""
        if not self._assign_name_client.wait_for_service(
                timeout_sec=self._max_timeout_time_for_action_call):
            self.get_logger().error('AssignNameToObject service not available')
            return False

        request = AssignNameToObject.Request()
        request.object_id = object_id
        request.name = name
        future = self._assign_name_client.call_async(request)
        result = self.wait_for_service_result(
            future, timeout=self._max_timeout_time_for_action_call)
        if result is None or not result.result:
            self.get_logger().error(
                f'Failed to assign name "{name}" to object id {object_id}')
            return False

        self.get_logger().info(f'Assigned name "{name}" to object id {object_id}')
        return True

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """Publish a static transform for the given pose."""
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        if self._use_sim_time:
            child_frame_ids = getattr(self.tf_static_broadcaster, '_child_frame_ids', None)
            if child_frame_ids is not None:
                child_frame_ids.discard(child_frame)

            net_message = getattr(self.tf_static_broadcaster, 'net_message', None)
            if net_message is not None:
                net_message.transforms = [
                    t for t in net_message.transforms if t.child_frame_id != child_frame
                ]

        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.get_logger().info(f'Published static pose frame from {parent_frame} to {child_frame}')

    def publish_pose_on_tf(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """Publish a dynamic transform for the given pose."""
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        self.tf_broadcaster.sendTransform(transform_stamped)
        self.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def wait_for_point_topic_func(self):
        """Wait for a message on the point topic."""
        self.received_messages[self._point_topic_name_as_trigger] = []
        self.get_logger().error('Waiting for point topic input for next step...')
        while True:
            time.sleep(0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.get_logger().info('Point topic received')
                return

    def do_perception_loop(self, initial_hint: Point, mesh_file_path: str,
                           object_frame_name: str) -> bool:
        """Run detection, segmentation, mesh assignment, naming, and pose estimation."""
        self._get_objects_result_received = False
        self._segmentation_result_received = False
        self._get_object_pose_result_received = False

        self.trigger_get_objects_goal(initial_hint)
        self.wait_for_result('_get_objects_result_received',
                             timeout=self._max_timeout_time_for_action_call)
        if not self._get_objects_result_received:
            self.get_logger().error('Object detection action failed')
            self.failure_count += 1
            return False

        self.total_count += 1

        if self._is_segment_anything_segmentation_enabled:
            self._segmentation_result_received = False
            self.trigger_get_segmented_object_goal(self._get_objects_detected_object_id)
            self.wait_for_result('_segmentation_result_received',
                                 timeout=self._max_timeout_time_for_action_call)
            if not self._segmentation_result_received:
                self.get_logger().error('Segmentation action failed')
                self.failure_count += 1
                return False

            future = self.trigger_add_mesh_to_object_request(
                self._get_objects_detected_object_id, mesh_file_path)
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.get_logger().info(f'Add mesh to object result: {result}')
            if result is None:
                self.get_logger().error('Add mesh to object action failed')
                self.failure_count += 1
                return False
            self.total_count += 1

        if not self.assign_name_to_object(self._get_objects_detected_object_id,
                                          object_frame_name):
            self.failure_count += 1
            return False

        self.trigger_get_object_pose_goal(self._get_objects_detected_object_id)
        self.wait_for_result('_get_object_pose_result_received',
                             timeout=self._max_timeout_time_for_action_call)
        if not self._get_object_pose_result_received:
            self.get_logger().error('Object pose estimation action failed')
            self.failure_count += 1
            return False

        self._get_objects_result_received = False
        self._segmentation_result_received = False
        self._get_object_pose_result_received = False
        return True

    def pick_and_place(
        self,
        object_id: int,
        class_id: str = '',
        place_pose: Pose = None,
        gripper_closed_position: float = 0.623,
        use_joint_space_planner: bool = False,
        keep_gripper_closed_after_completion: bool = False,
        only_perform_place: bool = False
    ) -> bool:
        """Pick and place an object using the PickAndPlace action."""
        pick_and_place_client = ActionClient(self, PickAndPlace, '/pick_and_place')

        if not pick_and_place_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('PickAndPlace action server not available')
            return False

        pick_and_place_goal = PickAndPlace.Goal()
        pick_and_place_goal.object_id = object_id
        pick_and_place_goal.class_id = class_id
        pick_and_place_goal.use_joint_space_planner_for_place_pose = use_joint_space_planner
        pick_and_place_goal.target_joint_state_for_place_pose = \
            self._target_joint_state_for_place_pose
        pick_and_place_goal.gripper_closed_position = gripper_closed_position
        pick_and_place_goal.keep_gripper_closed_after_completion = \
            keep_gripper_closed_after_completion
        pick_and_place_goal.place_pose = place_pose
        pick_and_place_goal.only_perform_place = only_perform_place

        self.get_logger().info(f'Sending PickAndPlace goal for object_id: {object_id}')
        goal_future = pick_and_place_client.send_goal_async(pick_and_place_goal)

        while not goal_future.done():
            time.sleep(0.1)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('PickAndPlace goal was rejected')
            return False

        self.get_logger().info('PickAndPlace goal accepted')
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.1)
        result = result_future.result()

        if not result.result.success:
            self.get_logger().error(f'PickAndPlace failed with status: {result.result}')
            return False

        self.get_logger().info(f'PickAndPlace completed successfully: {result.result}')
        return True

    def close_gripper_before_deploy(self) -> bool:
        """Tighten the held plug immediately before starting Deploy insertion."""
        if not self._send_gripper_command(
                self._pre_deploy_gripper_close_position,
                'Closing gripper before Deploy insertion',
                'Pre-Deploy gripper close'):
            return False

        if self._gripper_settle_time_sec > 0.0:
            self.get_logger().info(
                f'Waiting {self._gripper_settle_time_sec:.3f}s after pre-Deploy gripper close')
            self._wait_runtime_duration(self._gripper_settle_time_sec)
        return True

    def _send_gripper_command(self, position: float, description: str, error_prefix: str) -> bool:
        """Send one blocking GripperCommand and report whether it succeeded."""
        self.get_logger().info(
            f'{description}: position={position}, max_effort={self._gripper_max_effort}')
        if not self._gripper_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'Gripper action server {self._gripper_action_name} not available')
            return False

        goal_msg = ParallelGripperCommand.Goal()
        goal_msg.command.position = [float(position)]
        goal_msg.command.effort = [float(self._gripper_max_effort)]
        goal_future = self._gripper_client.send_goal_async(goal_msg)

        while not goal_future.done():
            time.sleep(0.1)
        goal_handle = goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'{error_prefix} goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.1)
        result = result_future.result()

        if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
            status = '<none>' if result is None else result.status
            self.get_logger().error(f'{error_prefix} failed with status: {status}')
            return False
        return True

    @staticmethod
    def _enforce_quaternion_positive(pose: Pose) -> Pose:
        """
        Hemisphere-normalise *pose*'s quaternion to the w >= 0 convention.

        A quaternion and its negation represent the same rotation, but the Deploy
        policy was trained with quaternion_positive (w >= 0) applied. A socket
        pose with w < 0 flips the sign of the orientation keypoint the policy
        observes, which can degrade insertion success. This matches the
        normalization the evaluator applies to its published goal pose.
        """
        if pose.orientation.w < 0.0:
            pose.orientation.x = -pose.orientation.x
            pose.orientation.y = -pose.orientation.y
            pose.orientation.z = -pose.orientation.z
            pose.orientation.w = -pose.orientation.w
        return pose

    def _publish_deploy_goal_pose(
            self, socket_pose: Pose, duration_sec: float,
            after_cycles: int = 0, on_after_cycles=None) -> None:
        """
        Publish the insertion goal pose for the requested duration.

        ``on_after_cycles`` runs once, after ``after_cycles`` poses have been published.
        The task-space path uses it to open the passthrough only after the graph has
        seen this run's socket pose, so a stale command from a previous cycle cannot
        reach the robot when the gate opens.
        """
        period_sec = 1.0 / self._goal_pose_publish_rate_hz
        start_ros_time = self.get_clock().now()
        start_wall_time = time.monotonic()
        next_publish_time = time.monotonic()

        publisher = (
            self._task_space_socket_pose_pub if self._use_task_space_policy
            else self._goal_pose_pub)

        published = 0
        while (
            rclpy.ok()
            and self._runtime_elapsed_sec(start_ros_time, start_wall_time) < duration_sec
        ):
            goal_pose = PoseStamped()
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.header.frame_id = self._socket_pose_base_frame
            goal_pose.pose = socket_pose
            publisher.publish(goal_pose)
            published += 1
            if self._deploy_external_policy:
                self._publish_sim_control_mode('policy', log=False)

            next_publish_time += period_sec
            time.sleep(max(0.0, next_publish_time - time.monotonic()))

            if on_after_cycles is not None and published >= after_cycles:
                on_after_cycles()
                on_after_cycles = None

    def _set_task_space_cartesian_impedance(self) -> bool:
        """
        Set Cartesian impedance now that the Cartesian controller is active.

        Must run after the controller switch: while the driver is still in a
        joint mode the service rejects the call with "Robot is not in Cartesian
        motion-force control mode".
        """
        if self._task_space_impedance_client is None:
            return True
        if not (self._task_space_cartesian_stiffness
                and self._task_space_cartesian_damping_ratio):
            self.get_logger().warning(
                'Cartesian impedance service configured but stiffness/damping are '
                'empty; leaving the driver at its current impedance')
            return True

        service_name = self._task_space_cartesian_impedance_service_name
        if not self._task_space_impedance_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f'Timed out waiting for Cartesian impedance service {service_name!r}')
            return False

        request = SetCartesianImpedance.Request()
        request.stiffness = self._task_space_cartesian_stiffness
        request.damping_ratio = self._task_space_cartesian_damping_ratio
        future = self._task_space_impedance_client.call_async(request)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.get_logger().error(
                f'Timed out calling Cartesian impedance service {service_name!r}')
            return False

        response = future.result()
        if response is None or not response.success:
            message = getattr(response, 'message', 'no response')
            self.get_logger().error(f'Failed to set Cartesian impedance: {message}')
            return False

        self.get_logger().info(
            f'Set Cartesian impedance: stiffness={self._task_space_cartesian_stiffness}, '
            f'damping_ratio={self._task_space_cartesian_damping_ratio}')
        return True

    def _open_gripper(self) -> bool:
        """Open the gripper to release the plug after insertion."""
        if not self._send_gripper_command(
                self._gripper_open_position,
                'Opening gripper to release the plug',
                'Gripper open'):
            return False

        # Give the fingers time to physically clear the plug before retracting.
        if self._gripper_settle_time_sec > 0.0:
            time.sleep(self._gripper_settle_time_sec)
        return True

    def _return_to_home_pose(self) -> bool:
        """Plan and execute a cuMotion motion back to the configured home pose."""
        home_pose = Pose()
        home_pose.position.x = self._task_space_home_pose[0]
        home_pose.position.y = self._task_space_home_pose[1]
        home_pose.position.z = self._task_space_home_pose[2]
        home_pose.orientation.x = self._task_space_home_pose[3]
        home_pose.orientation.y = self._task_space_home_pose[4]
        home_pose.orientation.z = self._task_space_home_pose[5]
        home_pose.orientation.w = self._task_space_home_pose[6]

        self.get_logger().info(f'Planning return to home pose: {self._task_space_home_pose}')
        plan_result = planning_utils.plan_to_pose(
            action_client=self._task_space_planner_client,
            logger=self.get_logger(),
            goal_pose=home_pose,
            link_name=self._base_frame,
            time_dilation_factor=self._time_dilation_factor,
        )
        if plan_result is None or not plan_result.success:
            self.get_logger().error('Failed to plan return to home pose')
            return False

        executed, _ = planning_utils.execute_trajectory(
            self._task_space_execute_client,
            self.get_logger(),
            plan_result.planned_trajectory[0],
        )
        if not executed:
            self.get_logger().error('Failed to execute return to home pose')
            return False

        self.get_logger().info('Returned to home pose')
        return True

    def _retract_after_task_space_insertion(self) -> bool:
        """
        Release the plug and retract once the policy has finished inserting.

        Runs after command authority is back with the joint controller, so the
        gripper opens and the arm retracts under normal trajectory control.
        """
        if self._task_space_release_after_insertion and not self._open_gripper():
            return False
        if self._task_space_return_home_after_insertion and not self._return_to_home_pose():
            return False
        return True

    def _set_task_space_passthrough_enabled(self, enabled: bool) -> None:
        """Open or close the task-space passthrough's command authority gate."""
        self._task_space_enable_pub.publish(Bool(data=enabled))
        self.get_logger().info(
            f'Published {enabled} on {self._task_space_enable_topic}')

    def _task_space_insertion(self, socket_pose: Pose) -> bool:
        """
        Run cable insertion with the task-space policy.

        The policy graph is already up but idle: it holds observations until a
        socket pose arrives and forwards no commands until the enable gate opens.
        Publishing the socket pose starts inference; opening the gate is what
        lets the resulting Cartesian targets reach the robot.
        """
        # The Cartesian controller is active, so the driver accepts the impedance setting.
        if not self._set_task_space_cartesian_impedance():
            return False

        try:
            self.get_logger().info(
                f'Publishing task-space socket pose on '
                f'{self._task_space_socket_pose_topic} at '
                f'{self._goal_pose_publish_rate_hz:.3f} Hz for '
                f'{self._max_timeout_time_for_action_call:.3f}s')
            # Publish with the gate closed first
            self._publish_deploy_goal_pose(
                socket_pose, duration_sec=self._max_timeout_time_for_action_call,
                after_cycles=self._task_space_enable_after_cycles,
                on_after_cycles=lambda: self._set_task_space_passthrough_enabled(True))
        finally:
            # Always drop authority, including on exception; the gate is the immediate stop.
            self._set_task_space_passthrough_enabled(False)

        self.get_logger().info('Finished task-space insertion goal publishing')
        return True

    def deploy_insertion(self, socket_pose: Pose) -> bool:
        """Run cable insertion by publishing goals to Isaac ROS Deploy."""
        socket_pose = self._enforce_quaternion_positive(socket_pose)

        if self._use_task_space_policy:
            return self._task_space_insertion(socket_pose)

        if self._deploy_external_policy:
            self.get_logger().info(
                'External Deploy policy selected; skipping joint-space safety-controller '
                'blend parameter writes')
        elif not self._set_deploy_safety_parameters({
                'max_blend_ratio_speed': self._deploy_max_blend_ratio_speed,
                'blend_ratio': self._deploy_blend_ratio}):
            return False

        ramp_wait_sec = self._deploy_pre_goal_wait_sec
        if not self._deploy_external_policy:
            ramp_wait_sec += self._blend_ramp_wait_sec(self._deploy_blend_ratio)
        if ramp_wait_sec > 0.0:
            self.get_logger().info(
                f'Waiting {ramp_wait_sec:.3f}s for Deploy blend ramp before publishing goal')
            time.sleep(ramp_wait_sec)

        if self._deploy_external_policy:
            self._publish_sim_control_mode('policy')

        self.get_logger().info(
            f'Publishing Deploy insertion goal on {self._goal_pose_topic} at '
            f'{self._goal_pose_publish_rate_hz:.3f} Hz for '
            f'{self._max_timeout_time_for_action_call:.3f}s '
            f'({"sim" if self._use_sim_time else "wall"} time)')
        self._publish_deploy_goal_pose(
            socket_pose, duration_sec=self._max_timeout_time_for_action_call)
        self.get_logger().info('Finished Deploy insertion goal publishing')
        return True

    def create_logging_subscribers(
        self,
        subscription_requests: Iterable[Tuple[str, Any]],
        received_messages: Dict[str, Iterable],
        use_namespace_lookup: bool = True,
        accept_multiple_messages: bool = False,
        add_received_message_timestamps: bool = False,
        qos_profile: QoSProfile = 10
    ) -> Iterable[Subscription]:
        """Create subscribers that log messages into the passed-in dictionary."""
        received_messages.clear()
        if accept_multiple_messages:
            for topic, _ in subscription_requests:
                received_messages[topic] = []

        def make_callback(topic):
            def callback(msg):
                if accept_multiple_messages:
                    if add_received_message_timestamps:
                        received_messages[topic].append((msg, time.time()))
                    else:
                        received_messages[topic].append(msg)
                else:
                    if topic in received_messages:
                        self.get_logger().error(f'Already received a message on topic {topic}!')
                        raise ValueError(f'Already received a message on topic {topic}!')
                    received_messages[topic] = msg

            return callback

        try:
            subscriptions = [
                self.create_subscription(
                    msg_type,
                    self.namespaces[topic] if use_namespace_lookup else topic,
                    make_callback(topic),
                    qos_profile,
                ) for topic, msg_type in subscription_requests
            ]
        except Exception as e:
            print('Failed to create subscriptions:')
            print(e)
            raise

        return subscriptions

    def _pose_from_array(self, pose_values: List[float]) -> Pose:
        """Convert a ``[x, y, z, qx, qy, qz, qw]`` array to ``geometry_msgs/Pose``."""
        pose = Pose()
        pose.position.x = pose_values[0]
        pose.position.y = pose_values[1]
        pose.position.z = pose_values[2]
        pose.orientation.x = pose_values[3]
        pose.orientation.y = pose_values[4]
        pose.orientation.z = pose_values[5]
        pose.orientation.w = pose_values[6]
        return pose

    def _quaternion_from_array(
            self, quaternion_values: List[float], param_name: str) -> Quaternion:
        """Convert a ``[qx, qy, qz, qw]`` array to a normalized ``geometry_msgs/Quaternion``."""
        if len(quaternion_values) != 4:
            raise ValueError(f'{param_name} must contain 4 values [qx, qy, qz, qw]')
        if not all(math.isfinite(value) for value in quaternion_values):
            raise ValueError(f'{param_name} values must be finite numbers')

        quaternion_norm = math.sqrt(sum(value * value for value in quaternion_values))
        if quaternion_norm <= 1e-6:
            raise ValueError(f'{param_name} quaternion must be non-zero')

        return Quaternion(
            x=quaternion_values[0] / quaternion_norm,
            y=quaternion_values[1] / quaternion_norm,
            z=quaternion_values[2] / quaternion_norm,
            w=quaternion_values[3] / quaternion_norm)

    def _correct_plug_orientation(self, pose: Pose) -> Pose:
        """
        Bring the estimated plug frame into the grasp frame convention.

        FoundationPose reports the pose of the plug mesh frame, whose axes do not line up
        with the convention the grasp file and the manual-pose workflow are authored
        against. ``plug_orientation_correction`` is the fixed rotation between the two conventions.
        """
        corrected_pose = Pose()
        corrected_pose.position.x = pose.position.x
        corrected_pose.position.y = pose.position.y
        corrected_pose.position.z = pose.position.z

        estimated_norm = math.sqrt(
            pose.orientation.x ** 2 +
            pose.orientation.y ** 2 +
            pose.orientation.z ** 2 +
            pose.orientation.w ** 2)
        if estimated_norm <= 1e-6:
            self.get_logger().warning(
                'FoundationPose returned an invalid plug orientation; leaving it unchanged')
            corrected_pose.orientation = pose.orientation
            return corrected_pose

        estimated_orientation = Quaternion(
            x=pose.orientation.x / estimated_norm,
            y=pose.orientation.y / estimated_norm,
            z=pose.orientation.z / estimated_norm,
            w=pose.orientation.w / estimated_norm)
        corrected_pose.orientation = geometry_utils.quaternion_multiply(
            estimated_orientation, self._plug_orientation_correction)
        return corrected_pose

    def _correct_socket_pose(self, pose: Pose) -> Pose:
        """
        Bring the estimated socket frame into the mate frame convention.

        FoundationPose reports the pose of the socket mesh frame, which differs from the
        convention the policy and the manual-pose workflow are authored against in two ways:
        its axes are rotated, and its origin sits inside the mesh body rather than on the
        socket rim where the plug mates.

        ``socket_orientation_correction`` is the fixed rotation between the two conventions.
        ``socket_mate_offset`` is the mesh-origin-to-rim translation, expressed in the
        *corrected* socket frame so that a tilted socket still resolves to the rim instead of
        a fixed distance along the base-frame z axis.
        """
        corrected_pose = Pose()

        estimated_norm = math.sqrt(
            pose.orientation.x ** 2 +
            pose.orientation.y ** 2 +
            pose.orientation.z ** 2 +
            pose.orientation.w ** 2)
        if estimated_norm <= 1e-6:
            self.get_logger().warning(
                'FoundationPose returned an invalid socket orientation; leaving it unchanged')
            corrected_pose.position.x = pose.position.x
            corrected_pose.position.y = pose.position.y
            corrected_pose.position.z = pose.position.z
            corrected_pose.orientation = pose.orientation
            return corrected_pose

        estimated_orientation = Quaternion(
            x=pose.orientation.x / estimated_norm,
            y=pose.orientation.y / estimated_norm,
            z=pose.orientation.z / estimated_norm,
            w=pose.orientation.w / estimated_norm)
        corrected_pose.orientation = geometry_utils.quaternion_multiply(
            estimated_orientation, self._socket_orientation_correction)

        mate_offset = geometry_utils.quaternion_to_rotation_matrix(
            corrected_pose.orientation) @ np.array(self._socket_mate_offset)
        corrected_pose.position.x = pose.position.x + float(mate_offset[0])
        corrected_pose.position.y = pose.position.y + float(mate_offset[1])
        corrected_pose.position.z = pose.position.z + float(mate_offset[2])
        return corrected_pose

    def _transform_camera_pose_to_base(self, pose: Pose) -> Pose | None:
        """Transform a camera-frame pose into the robot base frame."""
        base_pose_camera = self._get_transform_from_tf(
            self._camera_prim_name_in_tf, self._base_frame)
        if base_pose_camera is None:
            return None

        base_transform_object = (
            geometry_utils.transform_to_matrix(base_pose_camera) @
            geometry_utils.pose_to_matrix(pose)
        )
        translation, orientation = geometry_utils.matrix_to_ros_translation_quaternion(
            base_transform_object)
        base_pose_object = Pose()
        base_pose_object.position.x = translation.x
        base_pose_object.position.y = translation.y
        base_pose_object.position.z = translation.z
        base_pose_object.orientation = orientation
        return base_pose_object

    def _apply_hover_orientation(self, hover_pose: Pose, socket_pose: Pose) -> None:
        """Set the hover orientation from config, falling back to the socket orientation."""
        quat = self._hover_pose_orientation
        quat_norm = 0.0
        if len(quat) == 4:
            quat_norm = math.sqrt(sum(value * value for value in quat))
        if quat_norm > 1e-6:
            hover_pose.orientation.x = quat[0] / quat_norm
            hover_pose.orientation.y = quat[1] / quat_norm
            hover_pose.orientation.z = quat[2] / quat_norm
            hover_pose.orientation.w = quat[3] / quat_norm
        else:
            hover_pose.orientation = socket_pose.orientation

    def _compute_hover_pose_from_socket(self, socket_pose: Pose) -> Pose:
        """
        Compute the hover pose.

        Uses the configured absolute hover position when ``override_hover_position``
        is set, otherwise derives it from the socket (socket x/y, socket z + offset).
        The socket pose is left untouched so it can still be used independently (e.g.
        as the Deploy insertion goal).
        """
        hover_pose = Pose()
        if self._override_hover_position and len(self._hover_pose_position) == 3:
            hover_pose.position.x = self._hover_pose_position[0]
            hover_pose.position.y = self._hover_pose_position[1]
            hover_pose.position.z = self._hover_pose_position[2]
        else:
            hover_pose.position.x = socket_pose.position.x
            hover_pose.position.y = socket_pose.position.y
            hover_pose.position.z = socket_pose.position.z + self._offset_for_hover_pose
        self._apply_hover_orientation(hover_pose, socket_pose)
        return hover_pose

    def _wait_for_tf_publication(self) -> None:
        """Give freshly published static transforms a moment to reach the TF buffer."""
        time.sleep(1.0)

    def _get_socket_insertion_pose(self):
        """Read the socket pose used to anchor the insertion policy."""
        self._wait_for_tf_publication()
        socket_pose_transform = self._get_transform_from_tf(
            self._socket_tf_frame, self._socket_pose_base_frame)
        if socket_pose_transform is None:
            self.get_logger().error('Failed to get socket pose for Deploy insertion')
            return None

        socket_pose_for_insertion = geometry_utils.get_pose_from_transform(socket_pose_transform)
        self.publish_pose_on_tf_static(
            socket_pose_for_insertion,
            parent_frame=self._socket_pose_base_frame,
            child_frame='deploy_insertion_pose_frame')
        self.get_logger().info(f'Socket pose for Deploy insertion: {socket_pose_for_insertion}')
        return socket_pose_for_insertion

    def _restore_deploy_control(
        self,
        deploy_control_active: bool,
        deploy_blend_active: bool,
    ) -> bool:
        is_controller_restored = True
        # No blend ratio on the task-space path; the gate already dropped authority.
        if ((deploy_control_active or deploy_blend_active)
                and not self._use_task_space_policy):
            if self._set_deploy_safety_parameters({'blend_ratio': 0.0}):
                ramp_down_wait_sec = self._blend_ramp_wait_sec(self._deploy_blend_ratio)
                if ramp_down_wait_sec > 0.0:
                    self.get_logger().info(
                        f'Waiting {ramp_down_wait_sec:.3f}s for Deploy blend ramp down')
                    time.sleep(ramp_down_wait_sec)
            else:
                self.get_logger().error('Failed to ramp down Deploy blend ratio')

        if deploy_control_active:
            is_controller_restored = self.switch_from_deploy_control()
            if not is_controller_restored:
                self.get_logger().error('Failed to restore nominal controller')

        if self._deploy_external_policy:
            self._publish_sim_control_mode_for_wall_duration('joint')

        return is_controller_restored

    def _run_deploy_policy(self, socket_pose_for_insertion: Pose) -> bool:
        if self._pre_deploy_close_gripper:
            if not self.close_gripper_before_deploy():
                return False

        if not self._run_rl_inference:
            return True

        deploy_control_active = False
        deploy_blend_active = False
        is_deploy_insertion_success = False
        is_controller_restored = True

        try:
            if self._enable_controller_switching:
                if not self.switch_to_deploy_control():
                    self.get_logger().error('Failed to switch to Deploy control')
                    return False
                deploy_control_active = True

            if self._wait_for_point_topic:
                self.get_logger().info(
                    'Waiting for confirmation to start Deploy policy. '
                    f'Click any point on image. Will insert relative to '
                    f'{self._socket_tf_frame}')
                self.wait_for_point_topic_func()

            is_deploy_insertion_success = self.deploy_insertion(
                socket_pose=socket_pose_for_insertion)
            deploy_blend_active = (
                is_deploy_insertion_success and not self._deploy_external_policy)
        finally:
            is_controller_restored = self._restore_deploy_control(
                deploy_control_active, deploy_blend_active)

        if not is_deploy_insertion_success:
            self.get_logger().error('Deploy insertion failed')
            return False
        if not is_controller_restored:
            return False

        self.get_logger().info('Deploy insertion completed successfully')
        return True

    def _release_after_deploy(self, socket_pose_for_insertion: Pose) -> bool:
        if self._post_deploy_open_gripper:
            if self._post_deploy_pre_open_wait_sec > 0.0:
                self.get_logger().info(
                    'Waiting '
                    f'{self._post_deploy_pre_open_wait_sec:.3f}s after Deploy '
                    'insertion before opening gripper '
                    f'({"sim" if self._use_sim_time else "wall"} time)')
                self._wait_runtime_duration(self._post_deploy_pre_open_wait_sec)
            if not self.open_gripper_after_deploy():
                return False

        if self._post_deploy_retract_to_hover:
            hover_pose = self._compute_hover_pose_from_socket(socket_pose_for_insertion)
            if self._post_deploy_retreat_lift_distance > 0.0:
                hover_pose.position.z += self._post_deploy_retreat_lift_distance
            self.publish_pose_on_tf(
                hover_pose,
                parent_frame=self._socket_pose_base_frame,
                child_frame='post_deploy_hover_pose_frame')
            self.get_logger().info(
                'Retreating after Deploy insertion to hover pose with '
                f'{self._post_deploy_retreat_lift_distance:.3f}m')
            if not self.pick_and_place(
                    object_id=0,
                    place_pose=hover_pose,
                    gripper_closed_position=self._gripper_close_position,
                    use_joint_space_planner=False,
                    keep_gripper_closed_after_completion=True,
                    only_perform_place=True):
                self.get_logger().error('Post-Deploy retreat to hover failed')
                return False

        return True

    def _complete_insertion_after_hover(self, is_pick_success: bool) -> bool:
        """Run optional Deploy insertion after the pick-and-hover step."""
        if not is_pick_success:
            self.get_logger().error('Pick and place action failed')
            self.failure_count += 1
            return False

        socket_pose_for_insertion = self._get_socket_insertion_pose()
        if socket_pose_for_insertion is None:
            return False

        if not self._run_deploy_policy(socket_pose_for_insertion):
            return False

        if self._run_rl_inference and not self._release_after_deploy(socket_pose_for_insertion):
            return False

        # Only once the joint controller has authority again.
        if self._run_rl_inference and self._use_task_space_policy:
            if not self._retract_after_task_space_insertion():
                self.get_logger().error('Post-insertion retract failed')
                return False

        self.get_logger().info('DisplayPort cable insertion workflow completed successfully')
        return True

    def displayport_insertion_with_manual_poses(self) -> bool:
        """Run DisplayPort insertion using YAML-provided plug and socket poses."""
        plug_pose = self._pose_from_array(self._manual_plug_pose)
        socket_pose = self._pose_from_array(self._manual_socket_pose)

        self.get_logger().info('Starting DisplayPort cable insertion with manual poses')

        self.publish_pose_on_tf_static(
            socket_pose, parent_frame=self._base_frame, child_frame=self._socket_tf_frame)
        self.publish_pose_on_tf_static(
            plug_pose, parent_frame=self._base_frame, child_frame='detected_object1')
        self.publish_pose_on_tf_static(
            plug_pose, parent_frame=self._base_frame, child_frame='displayport_cable_frame')

        self._wait_for_tf_publication()

        hover_pose = self._compute_hover_pose_from_socket(socket_pose)
        self.get_logger().info(
            f'Computed hover pose with z offset: {self._offset_for_hover_pose}')
        self.publish_pose_on_tf(
            hover_pose,
            parent_frame=self._base_frame,
            child_frame='hover_pose_frame')

        is_pick_success = self.pick_and_place(
            object_id=0,
            place_pose=hover_pose,
            gripper_closed_position=self._gripper_close_position,
            use_joint_space_planner=self._use_joint_space_planner,
            keep_gripper_closed_after_completion=True
        )
        return self._complete_insertion_after_hover(is_pick_success)

    def displayport_insertion(self) -> bool:
        """Run the DisplayPort cable insertion workflow."""
        if self._use_manual_poses:
            return self.displayport_insertion_with_manual_poses()

        if not self._wait_for_point_topic:
            raise RuntimeError(
                'Automatic DisplayPort insertion has no point-trigger source: set '
                'use_manual_poses:=true to use YAML poses, or wait_for_point_topic:=true '
                'to drive perception from clicked points.')

        if self._wait_for_point_topic:
            self.received_messages = {}
            self._subs = self.create_logging_subscribers(
                [(self._point_topic_name_as_trigger, Point)],
                self.received_messages,
                use_namespace_lookup=False,
                accept_multiple_messages=True,
                qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info('Starting DisplayPort cable insertion workflow')

        self.get_logger().info('Waiting for point topic input for socket perception')
        self.wait_for_point_topic_func()
        socket_hint = self.received_messages[self._point_topic_name_as_trigger][-1]
        self.received_messages[self._point_topic_name_as_trigger] = []

        self.get_logger().info(
            f'Setting mesh file path for socket estimation: {self._socket_mesh_file_path}')
        if not self.set_mesh_param(self._socket_mesh_file_path):
            self.get_logger().error(
                'Failed to set FoundationPose mesh for socket estimation; aborting workflow')
            return False

        raw_socket_frame = 'displayport_socket_estimate_frame'
        is_socket_success = self.do_perception_loop(
            socket_hint, self._socket_mesh_file_path, raw_socket_frame)
        while not is_socket_success:
            self.get_logger().error('Socket perception failed, retrying...')
            is_socket_success = self.do_perception_loop(
                socket_hint, self._socket_mesh_file_path, raw_socket_frame)
            time.sleep(2)

        socket_estimate = self.latest_pose_gotten
        if not self._use_manual_poses:
            socket_estimate = self._correct_socket_pose(socket_estimate)
        self.socket_pose = self._transform_camera_pose_to_base(socket_estimate)
        if self.socket_pose is None:
            self.get_logger().error(
                'Failed to transform the socket pose into the robot base frame')
            return False
        self.publish_pose_on_tf_static(
            self.socket_pose,
            parent_frame=self._base_frame,
            child_frame=self._socket_tf_frame)

        cable_object_id = 0
        if not self._manual_grasp_mode:
            self.get_logger().info('Waiting for point topic input for cable perception')
            self.wait_for_point_topic_func()
            cable_hint = self.received_messages[self._point_topic_name_as_trigger][-1]
            self.received_messages[self._point_topic_name_as_trigger] = []

            self.get_logger().info(
                f'Setting mesh file path for cable estimation: {self._cable_mesh_file_path}')
            if not self.set_mesh_param(self._cable_mesh_file_path):
                self.get_logger().error(
                    'Failed to set FoundationPose mesh for cable estimation; aborting workflow')
                return False

            raw_cable_frame = 'displayport_cable_estimate_frame'
            is_cable_success = self.do_perception_loop(
                cable_hint, self._cable_mesh_file_path, raw_cable_frame)
            while not is_cable_success:
                self.get_logger().error('Cable perception failed, retrying...')
                is_cable_success = self.do_perception_loop(
                    cable_hint, self._cable_mesh_file_path, raw_cable_frame)
                time.sleep(2)

            self.cable_pose = self._correct_plug_orientation(self.latest_pose_gotten)
            self.cable_pose = self._transform_camera_pose_to_base(self.cable_pose)
            if self.cable_pose is None:
                self.get_logger().error(
                    'Failed to transform the corrected plug pose into the robot base frame')
                return False
            cable_object_id = self._get_objects_detected_object_id
            self.publish_pose_on_tf_static(
                self.cable_pose,
                parent_frame=self._base_frame,
                child_frame='detected_object1')
            self.publish_pose_on_tf_static(
                self.cable_pose,
                parent_frame=self._base_frame,
                child_frame='displayport_cable_frame')

        self._wait_for_tf_publication()

        socket_transform = self._get_transform_from_tf(
            self._socket_tf_frame, self._base_frame)
        if socket_transform is None:
            self.get_logger().error('Failed to get socket transform for hover pose')
            return False

        socket_pose = geometry_utils.get_pose_from_transform(socket_transform)
        hover_pose = self._compute_hover_pose_from_socket(socket_pose)

        self.get_logger().info(
            f'Computed hover pose with z offset: {self._offset_for_hover_pose}')
        self.publish_pose_on_tf(
            hover_pose,
            parent_frame=self._base_frame,
            child_frame='hover_pose_frame')

        if self._manual_grasp_mode:
            is_pick_success = self.pick_and_place(
                object_id=0,
                place_pose=hover_pose,
                gripper_closed_position=self._gripper_close_position,
                use_joint_space_planner=self._use_joint_space_planner,
                only_perform_place=True
            )
        else:
            is_pick_success = self.pick_and_place(
                object_id=cable_object_id,
                place_pose=hover_pose,
                gripper_closed_position=self._gripper_close_position,
                use_joint_space_planner=self._use_joint_space_planner,
                keep_gripper_closed_after_completion=True
            )

        if not is_pick_success:
            self.get_logger().error('Pick and place action failed')
            self.failure_count += 1
            return False

        return self._complete_insertion_after_hover(True)

    def execute_callback(self, goal_handle) -> DisplayPortCableInsertion.Result:
        """Execute the DisplayPortCableInsertion action."""
        self.get_logger().info('Executing DisplayPort cable insertion goal...')
        result = DisplayPortCableInsertion.Result()

        success = self.displayport_insertion()

        if success:
            result.success = True
            goal_handle.succeed()
        else:
            result.success = False
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)

    displayport_insertion_orchestrator = IsaacRosManipulationDisplayPortInsertionOrchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(displayport_insertion_orchestrator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        displayport_insertion_orchestrator.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    finally:
        executor.shutdown()
        displayport_insertion_orchestrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
