#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from copy import deepcopy
import json
import math
import os
from threading import Event
import time
from typing import Any, List

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Pose, PoseArray, TransformStamped, Vector3
from isaac_ros_cumotion_interfaces.action import AttachObject, IKSolution, MotionPlan
from isaac_ros_manipulation_interfaces.action import GetObjectPose, PickAndPlace
from isaac_ros_manipulation_orchestration.utils.end_effector_marker import EndEffectorMarker
from isaac_ros_manipulation_ros_python_utils.grasp_reader import GraspReader, Transformation
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    ObjectAttachmentShape
)
import isaac_ros_manipulation_ros_python_utils.planning_utils as planning_utils
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import GetCartesianPath

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker
import yaml

ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')


class PickAndPlaceOrchestrator(Node):

    def __init__(self):
        super().__init__('pick_and_place_orchestrator')
        # Declare parameters
        self.declare_parameter('gripper_collision_links', [''])
        self.declare_parameter('gripper_action_name', '/robotiq_gripper_controller/gripper_cmd')
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('gripper_close_position', 0.65)
        self.declare_parameter('gripper_settle_time_sec', 0.0)
        self.declare_parameter('gripper_max_effort', 10.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'gripper_frame')
        self.declare_parameter('grasp_frame', 'grasp_frame')
        self.declare_parameter('joint_names', [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ])
        self.declare_parameter('time_dilation_factor', 0.2)
        self.declare_parameter('use_ground_truth_pose_from_sim', False)
        self.declare_parameter('publish_grasp_frame', False)
        self.declare_parameter('sleep_time_before_planner_tries_sec', 0.5)
        self.declare_parameter('object_frame_name', 'detected_object1')
        self.declare_parameter('num_planner_tries', 5)
        self.declare_parameter('grasp_file_path', '')
        self.declare_parameter('grasp_approach_offset_distance', [0.0, 0.0, -0.15])
        self.declare_parameter('retract_offset_distance', [0.0, 0.0, 0.10])
        self.declare_parameter('attach_object_shape', str(ObjectAttachmentShape.CUBOID.value))
        self.declare_parameter('attach_object_mesh_file_path', '')
        self.declare_parameter('attach_object_scale', [0.05, 0.05, 0.1])
        self.declare_parameter('save_intermediate_outputs', True)
        self.declare_parameter('save_intermediate_outputs_dir', ISAAC_ROS_WS)
        self.declare_parameter('joint_limits_file_path', '')

        # Add parameters for home pose
        # The format is [x, y, z, qx, qy, qz, qw]
        self.declare_parameter('move_to_home_pose_after_place', False)
        self.declare_parameter('home_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # This needs to be an array of size == dof of the robot.
        self.declare_parameter(
            'seed_state_for_ik_solver_for_joint_space_planner',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('use_exact_target_joint_state_for_place_pose', False)

        # Whether the grasp approach and retract are in the world frame or the goal frame
        self.declare_parameter('grasp_approach_in_world_frame', False)
        self.declare_parameter('retract_in_world_frame', True)
        self.declare_parameter('use_cartesian_pickup_retract', False)
        self.declare_parameter('cartesian_retract_group_name', 'rizon_arm')
        self.declare_parameter('disable_pick_planning_scene', False)
        self.declare_parameter('disable_place_planning_scene', False)

        # When this parameter is set to True, the place pose of the end effector is taken from the
        # RViz via the interactive marker and place pose in the action request is ignored.
        self.declare_parameter('use_pose_from_rviz', False)
        # The mesh resource URI for the end effector; empty draws no end-effector mesh
        self.declare_parameter('end_effector_mesh_resource_uri', '')

        # Extract the parameters
        self._save_intermediate_outputs = self.get_parameter(
            'save_intermediate_outputs').get_parameter_value().bool_value
        self._save_intermediate_outputs_dir = self.get_parameter(
            'save_intermediate_outputs_dir').get_parameter_value().string_value

        self._is_running_isaac_sim = \
            self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self._use_ground_truth_pose_from_sim = self.get_parameter(
            'use_ground_truth_pose_from_sim').get_parameter_value().bool_value
        self._publish_grasp_frame = self.get_parameter(
            'publish_grasp_frame').get_parameter_value().bool_value
        self._sleep_time_before_planner_tries_sec = self.get_parameter(
            'sleep_time_before_planner_tries_sec').get_parameter_value().double_value
        self._num_planner_tries_ = self.get_parameter(
            'num_planner_tries').get_parameter_value().integer_value
        self._time_dilation_factor = self.get_parameter(
            'time_dilation_factor').get_parameter_value().double_value
        self._object_frame_name = \
            self.get_parameter('object_frame_name').get_parameter_value().string_value
        self._gripper_collision_links = (
            self.get_parameter('gripper_collision_links').get_parameter_value().string_array_value
        )
        gripper_action_name = (
            self.get_parameter('gripper_action_name').get_parameter_value().string_value
        )
        self._gripper_open_position = (
            self.get_parameter('gripper_open_position').get_parameter_value().double_value
        )
        self._gripper_close_position = (
            self.get_parameter('gripper_close_position').get_parameter_value().double_value
        )
        self._gripper_settle_time_sec = (
            self.get_parameter('gripper_settle_time_sec').get_parameter_value().double_value
        )
        self._gripper_max_effort = (
            self.get_parameter('gripper_max_effort').get_parameter_value().double_value
        )
        self._grasp_file_path = self.get_parameter(
            'grasp_file_path').get_parameter_value().string_value
        self._grasp_approach_offset_distance = self.get_parameter(
            'grasp_approach_offset_distance').get_parameter_value().double_array_value
        self._retract_offset_distance = self.get_parameter(
            'retract_offset_distance').get_parameter_value().double_array_value
        self._grasp_approach_in_world_frame = self.get_parameter(
            'grasp_approach_in_world_frame').get_parameter_value().bool_value
        self._retract_in_world_frame = self.get_parameter(
            'retract_in_world_frame').get_parameter_value().bool_value
        self._use_cartesian_pickup_retract = self.get_parameter(
            'use_cartesian_pickup_retract').get_parameter_value().bool_value
        self._cartesian_retract_group_name = self.get_parameter(
            'cartesian_retract_group_name').get_parameter_value().string_value
        self._disable_pick_planning_scene = self.get_parameter(
            'disable_pick_planning_scene').get_parameter_value().bool_value
        self._disable_place_planning_scene = self.get_parameter(
            'disable_place_planning_scene').get_parameter_value().bool_value

        self._use_pose_from_rviz = self.get_parameter(
            'use_pose_from_rviz').get_parameter_value().bool_value
        self._end_effector_mesh_resource_uri = self.get_parameter(
            'end_effector_mesh_resource_uri').get_parameter_value().string_value

        self._attach_object_shape = ObjectAttachmentShape(self.get_parameter(
            'attach_object_shape').get_parameter_value().string_value)
        self._attach_object_mesh_file_path = self.get_parameter(
            'attach_object_mesh_file_path').get_parameter_value().string_value
        attach_object_scale_list = self.get_parameter(
            'attach_object_scale').get_parameter_value().double_array_value

        self._move_to_home_pose_after_place = self.get_parameter(
            'move_to_home_pose_after_place').get_parameter_value().bool_value
        self._home_pose = self.get_parameter(
            'home_pose').get_parameter_value().double_array_value
        if len(attach_object_scale_list) != 3:
            self.get_logger().error('Received object scale length other than 3!')
            raise ValueError('Excepted object scale to be length 3!')

        self._base_frame = self.get_parameter(
            'base_frame').get_parameter_value().string_value
        self._gripper_frame = self.get_parameter(
            'gripper_frame').get_parameter_value().string_value
        self._grasp_frame = self.get_parameter(
            'grasp_frame').get_parameter_value().string_value
        self._joint_names = list(self.get_parameter(
            'joint_names').get_parameter_value().string_array_value)

        self._seed_state_for_ik_solver_for_joint_space_planner = self.get_parameter(
            'seed_state_for_ik_solver_for_joint_space_planner'
        ).get_parameter_value().double_array_value
        if len(self._seed_state_for_ik_solver_for_joint_space_planner) != \
                len(self._joint_names):
            raise ValueError(
                f'seed_state_for_ik_solver_for_joint_space_planner has '
                f'{len(self._seed_state_for_ik_solver_for_joint_space_planner)}'
                f' values but robot has {len(self._joint_names)} joints.')
        self._use_exact_target_joint_state_for_place_pose = self.get_parameter(
            'use_exact_target_joint_state_for_place_pose'
        ).get_parameter_value().bool_value

        self._attach_object_scale = Vector3()
        self._attach_object_scale.x = attach_object_scale_list[0]
        self._attach_object_scale.y = attach_object_scale_list[1]
        self._attach_object_scale.z = attach_object_scale_list[2]

        self._action_server = ActionServer(
            self, PickAndPlace, '/pick_and_place',
            execute_callback=self.execute_callback, cancel_callback=self.cancel_callback,)
        self._get_object_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self._get_object_pose_client = ActionClient(
            self, GetObjectPose, '/get_object_pose', callback_group=self._get_object_pose_cb_group)
        self._get_pose_done_event = Event()
        self._get_pose_done_result = None

        self._planner_cb_group = MutuallyExclusiveCallbackGroup()
        self.planner_action_client = ActionClient(
            self, MotionPlan, '/cumotion/motion_plan',
            callback_group=self._planner_cb_group)
        self.plan_result = None

        self._execute_plan_cb_group = MutuallyExclusiveCallbackGroup()
        self._execute_plan_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self._execute_plan_cb_group)

        self._cartesian_path_cb_group = MutuallyExclusiveCallbackGroup()
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self._cartesian_path_cb_group)

        self._object_attach_cb_group = MutuallyExclusiveCallbackGroup()
        self._object_attach_client = ActionClient(
            self, AttachObject, 'attach_object', callback_group=self._object_attach_cb_group)
        self._object_attach_done_event = Event()
        self._object_attach_done_result = False

        self._gripper_cb_group = MutuallyExclusiveCallbackGroup()
        self._gripper_client = ActionClient(
            self, ParallelGripperCommand, gripper_action_name,
            callback_group=self._gripper_cb_group)
        self._gripper_done_event = Event()
        self._gripper_done_result = False

        self._ik_solution_cb_group = MutuallyExclusiveCallbackGroup()
        self._ik_solution_client = ActionClient(
            self, IKSolution, '/cumotion/ik', callback_group=self._ik_solution_cb_group)
        self._ik_solution_done_event = Event()
        self._ik_solution_done_result = None

        self._grasp_reader = GraspReader(self._grasp_file_path)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._planning_scene_world = None
        self._planning_scene_sub = self.create_subscription(
            PlanningScene, '/planning_scene',
            self._planning_scene_callback, 10)

        self._client_goal_handles = []

        if self._use_pose_from_rviz:
            future = self._tf_buffer.wait_for_transform_async(
                self._base_frame, self._gripper_frame, rclpy.time.Time())
            future.add_done_callback(self.initialize_marker)

        self._joint_limits = {}
        self._joint_limits_file_path = self.get_parameter(
            'joint_limits_file_path').get_parameter_value().string_value
        with open(self._joint_limits_file_path, 'r') as file:
            raw_limits = yaml.safe_load(file)

        # Map joint limit keys to the actual prefixed joint names used by this
        # robot so that limits like "joint1" match "Rizon4s-062839_joint1".
        raw_joint_limits = raw_limits.get('joint_limits', {})
        resolved_limits = {}
        for jname in self._joint_names:
            for limit_key, limit_val in raw_joint_limits.items():
                if jname == limit_key or jname.endswith(f'_{limit_key}'):
                    resolved_limits[jname] = limit_val
                    break
        self._joint_limits = {'joint_limits': resolved_limits}

        self.get_logger().info('Pick and Place Orchestrator has been started.')

    def _planning_scene_callback(self, msg):
        self._planning_scene_world = msg.world
        self.get_logger().info(
            'Updated planning scene with %d collision objects' %
            len(msg.world.collision_objects), once=True)

    def initialize_marker(self, future):
        self._end_effector_marker = EndEffectorMarker(
            node=self, marker_namespace='end_effector_marker',
            mesh_resource_uri=self._end_effector_mesh_resource_uri)

        if (future.done()):
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._gripper_frame, rclpy.time.Time())
            initial_pose = Pose()
            initial_pose.position.x = transform.transform.translation.x
            initial_pose.position.y = transform.transform.translation.y
            initial_pose.position.z = transform.transform.translation.z
            initial_pose.orientation.w = transform.transform.rotation.w
            initial_pose.orientation.x = transform.transform.rotation.x
            initial_pose.orientation.y = transform.transform.rotation.y
            initial_pose.orientation.z = transform.transform.rotation.z
            self._end_effector_marker.set_pose(initial_pose)

        self.get_logger().info('End effector marker initialized.')

    def publish_grasp_transform(self, grasp_pose: Transformation | Pose,
                                frame_name: str = 'grasp_frame'):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Transformation | Pose): The grasp pose to publish.
            frame_name (str, optional): Frame name. Defaults to 'grasp_frame'.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = 'world'
        transform_stamped.child_frame_id = frame_name

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z

        if not isinstance(grasp_pose, Pose):
            transform_stamped.transform.rotation.x = grasp_pose.quaternion.x
            transform_stamped.transform.rotation.y = grasp_pose.quaternion.y
            transform_stamped.transform.rotation.z = grasp_pose.quaternion.z
            transform_stamped.transform.rotation.w = grasp_pose.quaternion.w
        else:
            transform_stamped.transform.rotation.x = grasp_pose.orientation.x
            transform_stamped.transform.rotation.y = grasp_pose.orientation.y
            transform_stamped.transform.rotation.z = grasp_pose.orientation.z
            transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self._tf_broadcaster.sendTransform(transform_stamped)
        self.get_logger().debug(f'Published grasp frame: {transform_stamped.child_frame_id}')

    def wait_for_server(self, action_client: ActionClient, timeout_sec: float = 5.0) -> bool:
        """
        Wait for the action server to be available.

        Args
        ----
            action_client (ActionClient): Action client
            timeout_sec (float, optional): timeout for client call. Defaults to 5.0.

        Returns
        -------
            bool: Returns if action server is ready for requests

        """
        self.get_logger().info(f'Waiting for {action_client._action_name} action server...')
        action_server_available = action_client.wait_for_server(timeout_sec)
        if not action_server_available:
            self.get_logger().error(
                f'{action_client._action_name} action server is not available. Aborting goal.')
            return False
        return True

    def read_grasp_poses(self) -> PoseArray:
        """
        Read the grasp poses from a yaml file.

        Returns
        -------
            PoseArray: List of grasps to send to the planner for goal set planning.

        """
        poses_arr = PoseArray()
        try:
            # Get the grasp pose
            grasp_poses = self._grasp_reader.get_pose_for_pick_task(
                world_frame=self._base_frame,
                object_frame_name=self._object_frame_name,
                tf_buffer=self._tf_buffer,
            )

            for i, grasp_pose in enumerate(grasp_poses):
                if self._publish_grasp_frame:
                    self.publish_grasp_transform(grasp_pose, f'grasp_frame_{i}')
                pose = Pose()
                poses_arr.header.frame_id = self._base_frame
                pose.position.x = grasp_pose.position.x
                pose.position.y = grasp_pose.position.y
                pose.position.z = grasp_pose.position.z
                pose.orientation.w = grasp_pose.quaternion.w
                pose.orientation.x = grasp_pose.quaternion.x
                pose.orientation.y = grasp_pose.quaternion.y
                pose.orientation.z = grasp_pose.quaternion.z
                poses_arr.poses.append(pose)

        except Exception as e:
            self.get_logger().error(f'Failed to compute grasp pose: {e}')

        self.get_logger().info(f'Grasp poses: {poses_arr.poses}')

        return poses_arr

    def get_plan_grasp(self, goal_pose_array: PoseArray) -> bool:
        """
        Plan for the list of grasps.

        Args
        ----
            goal_pose_array (PoseArray): List of grasps w.r.t base link

        Returns
        -------
            bool: Returns if a plan for grasp was successful

        """
        # Constuct the contraints axis based on the linear offset distance
        # Index 0-2 is for rotational constraints in x, y, and z respectively
        # Index 3-5 is for linear constraints in x, y, and z respectively
        grasp_approach_path_constraints = [0.1] * 6
        for i, offset in enumerate(self._grasp_approach_offset_distance):
            # If offset is greater than 0.0 do not constraint that axis
            if not math.isclose(offset, 0.0, abs_tol=1e-5):
                grasp_approach_path_constraints[3 + i] = 0.0

        retract_path_constraints = [0.1] * 6
        for i, offset in enumerate(self._retract_offset_distance):
            # If offset is greater than 0.0 do not constraint that axis
            if not math.isclose(offset, 0.0, abs_tol=1e-5):
                retract_path_constraints[3 + i] = 0.0
        start_time = time.time()
        self.plan_result = planning_utils.plan_grasp(
            action_client=self.planner_action_client,
            logger=self.get_logger(),
            goal_poses=goal_pose_array,
            link_name=self._base_frame,
            grasp_approach_offset_distance=list(self._grasp_approach_offset_distance),
            grasp_approach_path_constraint=grasp_approach_path_constraints,
            retract_offset_distance=list(self._retract_offset_distance),
            retract_path_constraint=retract_path_constraints,
            grasp_approach_constraint_in_goal_frame=not self._grasp_approach_in_world_frame,
            retract_constraint_in_goal_frame=not self._retract_in_world_frame,
            time_dilation_factor=self._time_dilation_factor,
            disable_collision_links=self._gripper_collision_links,
            update_planning_scene=not self._disable_pick_planning_scene,
            planning_scene_world=(
                None if self._disable_pick_planning_scene else self._planning_scene_world
            ),
            plan_grasp_to_retract=not self._use_cartesian_pickup_retract,
            grasp_translation_path_deviation_limit=-1.0,
        )
        end_time = time.time()
        self.get_logger().info(f'Planning time for pick phase: {end_time - start_time} seconds')
        if self.plan_result is None or not self.plan_result.success:
            if self.plan_result is not None:
                self.get_logger().error(f'Planning failed. {self.plan_result.message}')
                self.get_logger().error(f'Error Code: {self.plan_result.error_code}')
            else:
                self.get_logger().error('Planning failed. No result received.')
            return False
        return True

    def _get_joint_state(self, joint_positions: List[float]) -> JointState:
        """Get the joint state from the joint positions."""
        joint_state = JointState()
        joint_state.position = joint_positions
        joint_state.velocity = [0.0] * len(joint_positions)
        joint_state.effort = [0.0] * len(joint_positions)
        joint_state.name = list(self._joint_names)
        return joint_state

    def get_ik_solutions(self, pose: Pose):
        """Get the IK solutions for a given pose."""
        goal = IKSolution.Goal()
        goal.mesh_resource = self._attach_object_mesh_file_path
        goal.object_shape = self._attach_object_shape.value
        goal.object_scale = self._attach_object_scale
        goal.enable_aabb_clearing = True
        goal.world_frame = self._base_frame
        goal.object_frame = self._grasp_frame
        goal.object_esdf_clearing_padding = [0.05, 0.05, 0.05]
        goal.num_solutions_to_return = 16
        goal.goal_pose = pose
        goal.seed_state = self._get_joint_state(
            self._seed_state_for_ik_solver_for_joint_space_planner
        )
        goal_future = self._ik_solution_client.send_goal_async(goal)
        goal_future.add_done_callback(self.ik_solution_goal_response_cb)

    def ik_solution_goal_response_cb(self, future):
        """Get the response from the IK solution goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('IK solution goal rejected.')
            self._ik_solution_done_result = None
            self._ik_solution_done_event.set()
            return
        # Cache the goal handle for canceling
        ik_solution_result_future = goal_handle.get_result_async()
        ik_solution_result_future.add_done_callback(self.ik_solution_goal_result_cb)

    def ik_solution_goal_result_cb(self, future):
        """Get the result from the IK solution goal."""
        result = future.result().result

        if result.success:
            self.get_logger().info('IK solution goal succeeded.')
            self._ik_solution_done_result = result

            # Save this IK result in file.
            # file in output folder.
            if self._save_intermediate_outputs:
                output_file_path = os.path.join(
                    self._save_intermediate_outputs_dir, 'ik_result.json')

                all_joints = []
                for i, joint_state in enumerate(result.joint_states):
                    # make this a dict object,
                    joint_state_dict = {
                        'position': list(joint_state.position),
                        'velocity': list(joint_state.velocity),
                        'effort': list(joint_state.effort),
                        'name': list(joint_state.name),
                        'index': i,
                        'success': result.success[i],
                    }

                    all_joints.append(joint_state_dict)

                with open(output_file_path, 'w') as f:
                    json.dump(all_joints, f, indent=4)

        else:
            self.get_logger().error('IK solution goal failed {result}')
            self._ik_solution_done_result = None
        self._ik_solution_done_event.set()

    def _filter_target_joint_state_to_planned_joints(
            self, target_joint_state: JointState) -> JointState:
        """Filter a target joint state down to joints known to the planner limits."""
        joint_limits_dict = {}
        for joint_name, joint_limits in self._joint_limits['joint_limits'].items():
            joint_limits_dict[joint_name] = (joint_limits['min_position'],
                                             joint_limits['max_position'])

        arm_indices = [
            i for i, name in enumerate(target_joint_state.name)
            if name in joint_limits_dict
        ]
        if len(arm_indices) == len(target_joint_state.name):
            return target_joint_state

        filtered = JointState()
        filtered.header = target_joint_state.header
        filtered.name = [target_joint_state.name[i] for i in arm_indices]
        filtered.position = [target_joint_state.position[i] for i in arm_indices]
        filtered.velocity = [
            target_joint_state.velocity[i] if i < len(target_joint_state.velocity) else 0.0
            for i in arm_indices
        ]
        filtered.effort = [
            target_joint_state.effort[i] if i < len(target_joint_state.effort) else 0.0
            for i in arm_indices
        ]
        return filtered

    def _plan_exact_target_joint_state(self, target_joint_state: JointState) -> bool:
        """Plan directly to the requested place joint state."""
        if target_joint_state is None or not target_joint_state.position:
            self.get_logger().error(
                'Exact target joint-state place planning requested without a target.')
            return False

        target_joint_state = self._filter_target_joint_state_to_planned_joints(
            target_joint_state)
        if not target_joint_state.position:
            self.get_logger().error(
                'Exact target joint-state place planning has no joints after filtering.')
            return False

        self.get_logger().info(
            f'Planning place phase directly to target joint state: {target_joint_state}; '
            f'time_dilation_factor={self._time_dilation_factor}')
        self.plan_result = planning_utils.plan_to_joint_state(
            action_client=self.planner_action_client,
            logger=self.get_logger(),
            goal_state=target_joint_state,
            time_dilation_factor=self._time_dilation_factor,
            update_planning_scene=not self._disable_place_planning_scene,
            planning_scene_world=self._planning_scene_world,
        )
        if self.plan_result is None or not self.plan_result.success:
            self.get_logger().error('Planning exact target joint state failed.')
            return False
        return True

    def _plan_joint_space_pose(self, pose: Pose, target_joint_state: JointState = None) -> bool:
        """Plan joint space pose."""
        self.get_ik_solutions(pose)
        self._ik_solution_done_event.wait()
        self._ik_solution_done_event.clear()

        if self._ik_solution_done_result is None:
            self.get_logger().error('IK solution call failed.')
            return False

        received_joint_states = self._ik_solution_done_result.joint_states
        success_list = self._ik_solution_done_result.success

        # filter out the joint states that are not successful
        possible_joint_states = []
        for joint_state, success in zip(received_joint_states, success_list):
            if success:
                possible_joint_states.append(joint_state)
        if not possible_joint_states:
            self.get_logger().error(
                'IK solution call returned no successful joint-state candidates.')
            return False

        joint_limits_dict = {}
        for joint_name, joint_limits in self._joint_limits['joint_limits'].items():
            joint_limits_dict[joint_name] = (joint_limits['min_position'],
                                             joint_limits['max_position'])

        if target_joint_state is not None:
            target_joint_state = self._filter_target_joint_state_to_planned_joints(
                target_joint_state)

        # Save target joint space in json file.
        if self._save_intermediate_outputs:
            target_joint_state_dict = {
                'position': list(target_joint_state.position),
                'velocity': list(target_joint_state.velocity),
                'effort': list(target_joint_state.effort),
                'name': list(target_joint_state.name),
            }
            with open(
                os.path.join(self._save_intermediate_outputs_dir, 'target_joint_state.json'), 'w'
            ) as f:
                json.dump(target_joint_state_dict, f, indent=4)

        # Get sorted indexes in order (closest to farthest) from the target joint state
        sorted_indexes = planning_utils.get_sorted_indexes_of_closest_joint_states(
            possible_joint_states, target_joint_state, joint_limits_dict)

        # Save the indexes as well.
        self.get_logger().info(f'Sorted indexes: {sorted_indexes}')
        if self._save_intermediate_outputs:
            with open(
                os.path.join(self._save_intermediate_outputs_dir, 'sorted_indexes.json'), 'w'
            ) as f:
                sorted_indexes_dict = {
                    'sorted_indexes': sorted_indexes,
                }
                json.dump(sorted_indexes_dict, f, indent=4)

        closest_index = sorted_indexes[0]
        selected_joint_state = possible_joint_states[closest_index]
        self.get_logger().info(f'Selected joint state: {selected_joint_state}')

        # TODO(kchahal): Add an extension to OOD to log policy error if robot is not in the
        # correct joint state.

        self.plan_result = planning_utils.plan_to_joint_state(
            action_client=self.planner_action_client,
            logger=self.get_logger(),
            goal_state=selected_joint_state,
            time_dilation_factor=self._time_dilation_factor,
            update_planning_scene=not self._disable_place_planning_scene,
            planning_scene_world=self._planning_scene_world,
        )

        if self.plan_result is None or not self.plan_result.success:
            self.get_logger().error('Planning failed.')
            return False
        return True

    def get_plan_pose(self, pose: Pose, plan_joint_trajectory: bool = False,
                      target_joint_state: JointState = None) -> bool:
        """
        Get planning pose from cuMotion.

        Args
        ----
            pose (Pose): Pose in ROS format

        Returns
        -------
            bool: Whether planning for pose was successful or not.

        """
        if plan_joint_trajectory:
            if self._use_exact_target_joint_state_for_place_pose:
                self.get_logger().info('Planning place phase to exact target joint state')
                return self._plan_exact_target_joint_state(target_joint_state)
            self.get_logger().info('Planning place pose in joint space pose')
            return self._plan_joint_space_pose(pose, target_joint_state)

        self.get_logger().info('Planning place pose in cartesian space')
        self.plan_result = planning_utils.plan_to_pose(
            action_client=self.planner_action_client,
            logger=self.get_logger(),
            goal_pose=pose,
            link_name=self._base_frame,
            time_dilation_factor=self._time_dilation_factor,
            update_planning_scene=not self._disable_place_planning_scene,
            planning_scene_world=self._planning_scene_world,
        )
        if self.plan_result is None or not self.plan_result.success:
            self.get_logger().error('Planning failed.')
            return False
        return True

    def move_straight_up_with_cartesian_path(self) -> bool:
        """Execute a MoveIt Cartesian +Z retract from the current gripper pose."""
        if not self._cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/compute_cartesian_path service is not available')
            return False

        solution = self._compute_straight_up_cartesian_path(self._gripper_frame)
        if solution is not None:
            retract_success, _ = planning_utils.execute_trajectory(
                self._execute_plan_client, self.get_logger(), solution)
            if retract_success:
                return True

            self.get_logger().error(
                f'Cartesian retract execution failed for {self._gripper_frame}')

        self.get_logger().error('All Cartesian retract path attempts failed')
        return False

    def _compute_straight_up_cartesian_path(self, link_name: str):
        """Compute a straight +Z Cartesian path for a specific link."""
        try:
            current_transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                link_name,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=2.0),
            )
        except Exception as exc:
            self.get_logger().error(
                f'Failed to look up {link_name} transform for Cartesian retract: {exc}')
            return None

        current_pose = Pose()
        current_pose.position.x = current_transform.transform.translation.x
        current_pose.position.y = current_transform.transform.translation.y
        current_pose.position.z = current_transform.transform.translation.z
        current_pose.orientation = current_transform.transform.rotation

        offset = [float(value) for value in self._retract_offset_distance]
        distance = math.sqrt(sum(value * value for value in offset))
        waypoint_count = max(1, int(math.ceil(distance / 0.005)))
        waypoints = []
        for waypoint_index in range(1, waypoint_count + 1):
            scale = float(waypoint_index) / float(waypoint_count)
            waypoint = deepcopy(current_pose)
            waypoint.position.x += offset[0] * scale
            waypoint.position.y += offset[1] * scale
            waypoint.position.z += offset[2] * scale
            waypoints.append(waypoint)

        request = GetCartesianPath.Request()
        request.header.frame_id = self._base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.is_diff = True
        request.group_name = self._cartesian_retract_group_name
        request.link_name = link_name
        request.waypoints = waypoints
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.prismatic_jump_threshold = 0.0
        request.revolute_jump_threshold = 0.0
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = max(0.01, min(1.0, self._time_dilation_factor))
        request.max_acceleration_scaling_factor = max(0.01, min(1.0, self._time_dilation_factor))
        request.cartesian_speed_limited_link = link_name
        request.max_cartesian_speed = 0.04

        self.get_logger().info(
            f'Requesting Cartesian retract for {link_name} with '
            f'avoid_collisions=True: z={current_pose.position.z:.4f} '
            f'to z={waypoints[-1].position.z:.4f} in {self._base_frame} '
            f'using {len(waypoints)} waypoints')
        future = self._cartesian_path_client.call_async(request)
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 30.0:
                self.get_logger().error(
                    f'Timed out waiting for Cartesian retract path for {link_name}')
                return None
            time.sleep(0.1)

        if future.exception() is not None:
            self.get_logger().error(
                f'Cartesian retract service call failed: {future.exception()}')
            return None

        response = future.result()
        self.get_logger().info(
            f'Cartesian retract path fraction for {link_name} '
            f'(avoid_collisions=True): {response.fraction:.3f}; '
            f'error_code: {response.error_code.val}; '
            f'points: {len(response.solution.joint_trajectory.points)}')
        if response.fraction < 0.99:
            self.get_logger().error(
                f'Cartesian retract for {link_name} did not produce a complete '
                'straight-line path')
            return None
        if not response.solution.joint_trajectory.points:
            self.get_logger().error('Cartesian retract service returned an empty trajectory')
            return None
        return response.solution

    def trigger_get_object_pose(self, goal_handle):
        """
        Trigger the initial action call for gettting object pose.

        Args
        ----
            goal_handle (_type_): Goal handle

        Returns
        -------
            None

        """
        self._get_pose_done_event.clear()
        get_object_pose_goal = GetObjectPose.Goal()
        get_object_pose_goal.object_id = goal_handle.request.object_id
        get_object_pose_goal_future = self._get_object_pose_client.send_goal_async(
            get_object_pose_goal)
        get_object_pose_goal_future.add_done_callback(self.get_object_pose_goal_response_cb)

    def get_object_pose_goal_response_cb(self, future: Any):
        """
        Get the response from initial acceptance of object pose estimation server.

        Args
        ----
            future (Any): Future for action call

        Returns
        -------
            None

        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for GetObjectPose')
            self._get_pose_done_result = None
            self._get_pose_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for GetObjectPose')
        get_object_pose_result_future = goal_handle.get_result_async()
        get_object_pose_result_future.add_done_callback(self.get_object_pose_goal_result_cb)

    def get_object_pose_goal_result_cb(self, future: Any):
        """
        Get the result from the object pose estimation action server.

        Args
        ----
            future (Any): Future describing the action call status

        Returns
        -------
            None

        """
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._get_pose_done_result = result
            self.get_logger().info('GetObjectPose action server succeeded.')
        elif status == GoalStatus.STATUS_ABORTED:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server aborted.')
        elif status == GoalStatus.STATUS_CANCELED:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server canceled.')
        else:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server failed.')
        self._get_pose_done_event.set()

    def trigger_object_attach(self, do_attach: bool):
        """
        Trigger object attachment or deattachment based on argument variable.

        Args
        ----
            do_attach (bool): Boolean indicating object attachment else deattachment

        Returns
        -------
            None

        """
        self._object_attach_done_event.clear()
        object_attach_goal = AttachObject.Goal()
        object_attach_goal.attach_object = do_attach
        object_attach_goal.object_config = self._create_object_config()

        def feedback(feedback_msg):
            return self.get_logger().info(f'Object Attachment Feedback = \
                                          {feedback_msg.feedback.status}')

        object_attach_goal_future = self._object_attach_client.send_goal_async(
            object_attach_goal, feedback_callback=feedback)
        object_attach_goal_future.add_done_callback(self.object_attach_goal_response_cb)

    def _create_object_config(self) -> Marker:
        """
        Create the object attachment config according to the user selection for CUBE or SPHERE.

        Returns
        -------
            Marker: Marker object that is used in the client request to object attachment

        """
        marker = Marker()
        gripper_pose_object = self._grasp_reader.get_grasp_pose_object(
            index=self.plan_result.goal_index,
            gripper_frame=self._gripper_frame,
            grasp_frame=self._grasp_frame,
            tf_buffer=self._tf_buffer)

        pose = Pose()
        pose.position.x = gripper_pose_object.position.x
        pose.position.y = gripper_pose_object.position.y
        pose.position.z = gripper_pose_object.position.z
        pose.orientation.x = gripper_pose_object.quaternion.x
        pose.orientation.y = gripper_pose_object.quaternion.y
        pose.orientation.z = gripper_pose_object.quaternion.z
        pose.orientation.w = gripper_pose_object.quaternion.w
        marker.pose = pose
        marker.scale = self._attach_object_scale

        if self._attach_object_shape == ObjectAttachmentShape.SPHERE:
            marker.type = Marker.SPHERE
        elif self._attach_object_shape == ObjectAttachmentShape.CUBOID:
            marker.type = Marker.CUBE
        elif self._attach_object_shape == ObjectAttachmentShape.CUSTOM_MESH:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = self._attach_object_mesh_file_path
        else:
            self.get_logger().error('Received unknown object type!')

        marker.header.frame_id = self._grasp_frame
        marker.frame_locked = True
        marker.color.r = 1.0
        marker.color.a = 1.0

        return marker

    def object_attach_goal_response_cb(self, future):
        """
        Get the object attachment response.

        Args
        ----
            future (_type_): Future for object attachment

        Returns
        -------
            None

        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for AttachObject')
            self._object_attach_done_result = False
            self._object_attach_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for AttachObject')
        object_attach_result_future = goal_handle.get_result_async()
        object_attach_result_future.add_done_callback(self.object_attach_goal_result_cb)

    def object_attach_goal_result_cb(self, future):
        """
        Get the object attachment goal result.

        Args
        ----
            future (_type_): Future for object attachment

        Returns
        -------
            None

        """
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'{result.outcome}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('AttachObject action server succeeded.')
            self._object_attach_done_result = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('AttachObject action server aborted.')
            self._object_attach_done_result = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('AttachObject action server canceled.')
            self._object_attach_done_result = False
        else:
            self.get_logger().error('AttachObject action server failed.')
            self._object_attach_done_result = False

        self._object_attach_done_event.set()

    def trigger_gripper(self, position: float, max_effort: float):
        """
        Trigger gripper to either open or close.

        Args
        ----
            position (float): Position
            max_effort (float): Max effort

        Returns
        -------
            None

        """
        self._gripper_done_event.clear()
        gripper_goal = ParallelGripperCommand.Goal()
        gripper_goal.command.position = [float(position)]
        gripper_goal.command.effort = [float(max_effort)]

        gripper_goal_future = self._gripper_client.send_goal_async(gripper_goal)
        gripper_goal_future.add_done_callback(self.gripper_goal_response_cb)

    def gripper_goal_response_cb(self, future):
        """
        Get goal response for Gripper.

        Args
        ----
            future (_type_): Future for goal response

        Returns
        -------
            None

        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for ParallelGripperCommand')
            self._gripper_done_result = False
            self._gripper_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for ParallelGripperCommand')
        gripper_result_future = goal_handle.get_result_async()
        gripper_result_future.add_done_callback(self.gripper_goal_result_cb)

    def gripper_goal_result_cb(self, future: Any):
        """
        Get the final result from gripper trigger action call.

        Args
        ----
            future (Any): Future for goal result

        Returns
        -------
            None

        """
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('ParallelGripperCommand action server succeeded.')
            self._gripper_done_result = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('ParallelGripperCommand action server aborted.')
            self._gripper_done_result = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('ParallelGripperCommand action server canceled.')
            self._gripper_done_result = False
        else:
            self.get_logger().error('ParallelGripperCommand action server failed.')
            self._gripper_done_result = False

        self._gripper_done_event.set()

    def close_gripper(self, position: float = None, max_effort: float = 10.0) -> bool:
        """
        Close gripper by triggering an action call.

        Args
        ----
            position (float, optional): Position. Defaults to gripper_close_position param.
            max_effort (float, optional): Max effort. Defaults to 10.0.

        Returns
        -------
            bool: bool: Status of whether gripper closed successfully or not

        """
        if position is None:
            position = self._gripper_close_position
        self.get_logger().info('Closing gripper')
        self.trigger_gripper(position, max_effort)

        self._gripper_done_event.wait()
        if not self._gripper_done_result:
            self.get_logger().error('Failed to close the gripper.')
            return False

        if self._gripper_settle_time_sec > 0:
            self.get_logger().info(
                f'Waiting {self._gripper_settle_time_sec}s for gripper to settle')
            time.sleep(self._gripper_settle_time_sec)

        return True

    def open_gripper(self, position: float = None, max_effort: float = 10.0) -> bool:
        """
        Open gripper by triggering an action call.

        Args
        ----
            position (float, optional): Position. Defaults to gripper_open_position param.
            max_effort (float, optional): Max effort. Defaults to 10.0.

        Returns
        -------
            bool: bool: Status of whether gripper opened successfully or not

        """
        if position is None:
            position = self._gripper_open_position
        self.get_logger().info('Opening gripper')
        self.trigger_gripper(position, max_effort)

        self._gripper_done_event.wait()
        if not self._gripper_done_result:
            self.get_logger().error('Failed to open the gripper.')
            return False

        if self._gripper_settle_time_sec > 0:
            self.get_logger().info(
                f'Waiting {self._gripper_settle_time_sec}s for gripper to settle')
            time.sleep(self._gripper_settle_time_sec)

        return True

    def client_cancel_done(self, future):
        """
        Cancel the client call.

        Args
        ----
            future (_type_): Future for cancel action call

        Returns
        -------
            None

        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().error('Goal failed to cancel')

    def cancel_callback(self, goal_handle) -> CancelResponse:
        """
        Trigger cancel callback if user has tried to cancel the action call.

        Args
        ----
            goal_handle (_type_): Goal handle

        Returns
        -------
            CancelResponse: Cancel response

        """
        self.get_logger().info('Received cancel request')

        # Cancel all the client goal handles that are executing or canceling
        filtered_client_goal_handles = [
            client_goal_handle for client_goal_handle in self._client_goal_handles
            if client_goal_handle.status == GoalStatus.STATUS_EXECUTING or
            client_goal_handle.status == GoalStatus.STATUS_CANCELING
        ]
        for client_goal_handle in filtered_client_goal_handles:
            client_goal_handle.cancel_goal_async().add_done_callback(self.client_cancel_done)

        # Clear the client goal handles
        self._client_goal_handles.clear()
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle) -> PickAndPlace.Result:
        """
        Execute the action call functionality.

        Args
        ----
            goal_handle (_type_): Goal handle

        Returns
        -------
            PickAndPlace.Result: Pick and place result

        """
        self.get_logger().info('Executing goal...')
        result = PickAndPlace.Result()

        only_perform_place = goal_handle.request.only_perform_place

        # Wait for the get_object_pose action server to be available, do not wait if we use ground
        # truth pose in sim
        if not only_perform_place and not self._use_ground_truth_pose_from_sim:
            if not self.wait_for_server(self._get_object_pose_client):
                result.success = False
                goal_handle.abort()
                return result

        # Wait for the object attachment/detachment action server to be available
        if not only_perform_place and not self.wait_for_server(self._object_attach_client):
            result.success = False
            goal_handle.abort()
            return result

        # Wait for the gripper action server to be available
        if not only_perform_place and not self.wait_for_server(self._gripper_client):
            result.success = False
            goal_handle.abort()
            return result

        # Wait for the IK solution action server to be available
        if not self.wait_for_server(self._ik_solution_client):
            result.success = False
            goal_handle.abort()
            return result

        # Trigger the get object pose action if sim ground truth is not enabled
        if not only_perform_place and not self._use_ground_truth_pose_from_sim:
            self.trigger_get_object_pose(goal_handle)
            # Wait for action to be done
            self._get_pose_done_event.wait()
            if self._get_pose_done_result is None:
                self.get_logger().error('Failed to get object pose.')
                result.success = False
                goal_handle.abort()
                return result

        # Trigger the planning for pick phase
        self.get_logger().info('Starting orchestrator for pick and place')
        pick_success = False
        plan_only_retraction = False
        retry_cartesian_retract_only = False

        # Open gripper only for pick flows; place-only starts with an already-held object.
        if not only_perform_place:
            if not self.open_gripper():
                result.success = False
                goal_handle.abort()
                return result

        gripper_closed_position = goal_handle.request.gripper_closed_position

        for i in range(self._num_planner_tries_):
            if only_perform_place:
                pick_success = True
                break
            if goal_handle.status == GoalStatus.STATUS_CANCELING or \
               goal_handle.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warning('Received request to cancel goal...')
                result.success = False
                goal_handle.abort()
                return result
            if retry_cartesian_retract_only:
                self.get_logger().info('Re-executing Cartesian pickup retract')
                if not self.move_straight_up_with_cartesian_path():
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                pick_success = True
                break
            self.get_logger().info(
                f'Executing pick pose {i+1} / {self._num_planner_tries_}')
            if self.get_plan_grasp(self.read_grasp_poses()):
                # Executing grasp trajectory
                self.get_logger().info(
                    f'Found plan grasp trajectories, number of trajectories'
                    f': {len(self.plan_result.planned_trajectory)}')

                if plan_only_retraction:
                    self.get_logger().info('Re-executing retraction plan')
                    pick_success, _ = planning_utils.execute_trajectory(
                        self._execute_plan_client, self.get_logger(),
                        self.plan_result.planned_trajectory[2])
                    if not pick_success:
                        plan_only_retraction = True
                        time.sleep(self._sleep_time_before_planner_tries_sec)
                        continue
                    pick_success = True
                    break
                # Executing pre-grasp trajectory
                pick_success, _ = planning_utils.execute_trajectory(
                    self._execute_plan_client, self.get_logger(),
                    self.plan_result.planned_trajectory[0])
                if not pick_success:
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                # Executing approach trajectory
                pick_success, _ = planning_utils.execute_trajectory(
                    self._execute_plan_client, self.get_logger(),
                    self.plan_result.planned_trajectory[1])
                if not pick_success:
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                # align for both real and sim.
                if not self.close_gripper(gripper_closed_position, self._gripper_max_effort):
                    result.success = False
                    goal_handle.abort()
                    return result
                if self._use_cartesian_pickup_retract:
                    self.get_logger().info(
                        'Executing single Cartesian pickup retract from closed-gripper pose')
                    pick_success = self.move_straight_up_with_cartesian_path()
                else:
                    # Executing retract trajectory
                    self.get_logger().info('Executing plan')
                    pick_success, _ = planning_utils.execute_trajectory(
                        self._execute_plan_client, self.get_logger(),
                        self.plan_result.planned_trajectory[2])
                if not pick_success:
                    if self._use_cartesian_pickup_retract:
                        retry_cartesian_retract_only = True
                    else:
                        plan_only_retraction = True
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                pick_success = True
                break
            else:
                self.get_logger().error('Planning for pick phase failed, trying again')
            self.get_logger().info(
                f'Waiting for {self._sleep_time_before_planner_tries_sec} seconds')
            time.sleep(self._sleep_time_before_planner_tries_sec)

        if not pick_success:
            self.get_logger().error('Planning for pick phase failed.')
            result.success = False
            goal_handle.abort()
            return result

        # Attach object
        if not only_perform_place:
            self.get_logger().info('Triggering object attachment')
            self.trigger_object_attach(do_attach=True)

            # Wait for action to be done
            self._object_attach_done_event.wait()
            if not self._object_attach_done_result:
                self.get_logger().error('Failed to attach object.')
                result.success = False
                goal_handle.abort()
                return result

        # Clearing the result for the next action (detach object)
        self._object_attach_done_result = False

        # Trigger the planning for drop phase
        self.get_logger().info('Getting place pose')
        place_pose = None
        if self._use_pose_from_rviz:
            place_pose = self._end_effector_marker.get_pose()
        else:
            place_pose = goal_handle.request.place_pose

        use_joint_space_planner = goal_handle.request.use_joint_space_planner_for_place_pose
        target_joint_state = goal_handle.request.target_joint_state_for_place_pose
        keep_gripper_closed_after_completion = \
            goal_handle.request.keep_gripper_closed_after_completion

        self.get_logger().info(f'Place pose: {place_pose}')

        # Publish on Rviz for debugging where the robot will place the object
        self.publish_grasp_transform(place_pose, 'place_pose')
        place_success = False
        for i in range(self._num_planner_tries_):
            if goal_handle.status == GoalStatus.STATUS_CANCELING or \
               goal_handle.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warning('Received request to cancel goal...')
                result.success = False
                goal_handle.abort()
                return result
            self.get_logger().info(
                f'Executing place pose {i+1} / {self._num_planner_tries_} after '
                f'{self._sleep_time_before_planner_tries_sec} second pause')
            start_time = time.time()
            if self.get_plan_pose(place_pose, plan_joint_trajectory=use_joint_space_planner,
                                  target_joint_state=target_joint_state):
                # Executing grasp trajectory
                place_success, _ = planning_utils.execute_trajectory(
                    self._execute_plan_client, self.get_logger(),
                    self.plan_result.planned_trajectory[0])
                end_time = time.time()
                self.get_logger().info(
                    f'Planning time for place phase: {end_time - start_time} seconds')
                if not place_success:
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                if keep_gripper_closed_after_completion:
                    place_success = True
                    break
                if not self.open_gripper():
                    result.success = False
                    goal_handle.abort()
                    return result
                place_success = True
                break
            else:
                self.get_logger().error('Planning for drop phase failed, trying again')
            time.sleep(self._sleep_time_before_planner_tries_sec)

        if not place_success:
            self.get_logger().error('Planning for drop phase failed.')
            result.success = False
            goal_handle.abort()
            return result
        # Detach object
        if not only_perform_place:
            self.get_logger().info('Triggering object detachment')
            self.trigger_object_attach(do_attach=False)
            # Wait for action to be done
            self._object_attach_done_event.wait()
            if not self._object_attach_done_result:
                self.get_logger().error('Failed to detach object.')
                result.success = False
                goal_handle.abort()
                return result

        if use_joint_space_planner or only_perform_place:
            result.success = True
            goal_handle.succeed()
            return result

        if self._move_to_home_pose_after_place:
            plan_home_pose = False
            home_pose_obj = Pose()
            home_pose_obj.position.x = self._home_pose[0]
            home_pose_obj.position.y = self._home_pose[1]
            home_pose_obj.position.z = self._home_pose[2]
            home_pose_obj.orientation.x = self._home_pose[3]
            home_pose_obj.orientation.y = self._home_pose[4]
            home_pose_obj.orientation.z = self._home_pose[5]
            home_pose_obj.orientation.w = self._home_pose[6]

            self.get_logger().info('Planning for home pose')
            start_time = time.time()
            if self.get_plan_pose(home_pose_obj):
                self.get_logger().info('Executing home pose')
                end_time = time.time()
                self.get_logger().info(
                    f'Planning time for home pose: {end_time - start_time} seconds')
                plan_home_pose, _ = planning_utils.execute_trajectory(
                    self._execute_plan_client, self.get_logger(),
                    self.plan_result.planned_trajectory[0])
            else:
                plan_home_pose = False
            self.get_logger().info('Home pose reached')
            if not plan_home_pose:
                self.get_logger().error('Failed to plan home pose')
                result.success = False
                goal_handle.abort()
                return result

        goal_handle.succeed()
        result.success = True
        return result


def main(args=None):
    rclpy.init(args=args)

    pick_and_place_orchestrator = PickAndPlaceOrchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(pick_and_place_orchestrator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pick_and_place_orchestrator.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    finally:
        executor.shutdown()
        pick_and_place_orchestrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
