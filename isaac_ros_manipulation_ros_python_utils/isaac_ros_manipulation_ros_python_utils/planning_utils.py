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
Utility functions for motion planning with cuMotion.

This module provides standalone functions for sending motion plan goals
to the cuMotion action server and executing planned trajectories.
Functions follow the send-and-wait paradigm used in perception_utils.py,
avoiding the need for callback functions in the calling class.
"""

from logging import Logger
import time
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import Pose, PoseArray, Vector3
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import PlanningSceneWorld, RobotTrajectory
import numpy as np
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def wait_for_result(future, logger: Logger, timeout: float = 10.0) -> Any:
    """
    Wait for the result of the future.

    This is a helper function which sends a goal, makes sure its accepted and
    then waits for result. It is a common paradigm used by ActionClients in ROS2.
    """
    start_time = time.time()
    while not future.done():
        if time.time() - start_time > timeout:
            logger.error('Timeout waiting for result')
            return False
        time.sleep(0.1)
    goal_handle = future.result()

    if not goal_handle.accepted:
        logger.error('Goal was rejected')
        raise RuntimeError('Goal was rejected')

    logger.info('Goal accepted')

    # Wait for the result
    result_future = goal_handle.get_result_async()
    while not result_future.done():
        if time.time() - start_time > timeout:
            logger.error('Timeout waiting for result')
            return False
        time.sleep(0.1)
    return result_future.result().result


def send_motion_plan_goal(action_client: ActionClient,
                          logger: Logger,
                          goal_msg: MotionPlan.Goal,
                          timeout: float = 120.0) -> Any:
    """
    Send a MotionPlan goal to the action server and wait for the result.

    This is a generic helper that handles server availability, goal sending,
    and result waiting using the async send-and-wait paradigm from
    perception_utils.wait_for_result.

    Args
    ----
        action_client (ActionClient): The action client for cuMotion motion
            planning (MotionPlan action type)
        logger (Logger): The logger to use
        goal_msg (MotionPlan.Goal): The fully constructed goal message
        timeout (float): Maximum time to wait for the result in seconds.
            Defaults to 120.0.

    Returns
    -------
        Any: The MotionPlan.Result if successful, None on failure.
            The result contains:
            - success (bool): Whether planning succeeded
            - message (str): Status/error message
            - goal_index (int): Index of the selected goal (for goal set planning)
            - error_code (MoveItErrorCodes): Error code
            - planned_trajectory (list[RobotTrajectory]): Planned trajectories
            - planning_time (float): Time spent planning

    """
    logger.info('Waiting for motion plan action server...')
    if not action_client.wait_for_server(timeout_sec=10.0):
        logger.error(
            'Motion plan action server not available after waiting')
        return None

    logger.info('Sending motion plan goal...')
    future = action_client.send_goal_async(goal_msg)

    try:
        result = wait_for_result(future, logger, timeout=timeout)
    except RuntimeError as e:
        logger.error(f'Motion plan goal failed: {e}')
        return None

    if result is not None and result is not False:
        logger.info('Received motion plan result')
        return result

    logger.error('Failed to get motion plan result')
    return None


def plan_grasp(action_client: ActionClient,
               logger: Logger,
               goal_poses: PoseArray,
               link_name: str,
               grasp_approach_offset_distance: List[float],
               grasp_approach_path_constraint: List[float],
               retract_offset_distance: List[float],
               retract_path_constraint: List[float],
               grasp_approach_constraint_in_goal_frame: bool,
               retract_constraint_in_goal_frame: bool,
               time_dilation_factor: float,
               disable_collision_links: List[str],
               object_frame: str = 'detected_object1',
               world_frame: str = 'base_link',
               mesh_resource: str = '',
               object_shape: str = '',
               object_scale: Optional[List[float]] = None,
               enable_aabb_clearing: bool = False,
               object_esdf_clearing_padding: Optional[List[float]] = None,
               update_planning_scene: bool = False,
               planning_scene_world: Optional[PlanningSceneWorld] = None,
               plan_approach_to_grasp: bool = True,
               plan_grasp_to_retract: bool = True,
               grasp_translation_path_deviation_limit: float = 0.001,
               grasp_translation_terminal_deviation_limit: float = -1.0,
               grasp_enable_orientation_path_axis_constraint: bool = True,
               grasp_orientation_path_axis_deviation_limit: float = -1.0,
               grasp_orientation_terminal_deviation_limit: float = -1.0,
               retract_translation_path_deviation_limit: float = 0.001,
               retract_translation_terminal_deviation_limit: float = -1.0,
               retract_enable_orientation_path_axis_constraint: bool = True,
               retract_orientation_path_axis_deviation_limit: float = -1.0,
               retract_orientation_terminal_deviation_limit: float = -1.0,
               timeout: float = 120.0) -> Any:
    """
    Plan a grasp motion including approach, grasp, and retract trajectories.

    Constructs a MotionPlan goal for grasp planning and sends it to the
    cuMotion action server. The resulting plan includes separate trajectories
    for the approach-to-grasp and grasp-to-retract phases.

    Args
    ----
        action_client (ActionClient): The action client for cuMotion motion
            planning (MotionPlan action type)
        logger (Logger): The logger to use
        goal_poses (PoseArray): Array of candidate grasp poses for goal set
            planning. The planner selects the best reachable pose.
        link_name (str): Reference frame for the goal poses (e.g., 'base_link')
        grasp_approach_offset_distance (list[float]): 3D offset distance for
            grasp approach motion [x, y, z]
        grasp_approach_path_constraint (list[float]): 6D path constraints for
            grasp approach [rx, ry, rz, x, y, z]. Values near 0.0 indicate
            unconstrained axes, values > 0 indicate constrained axes.
        retract_offset_distance (list[float]): 3D offset distance for retract
            motion after grasping [x, y, z]
        retract_path_constraint (list[float]): 6D path constraints for retract
            motion [rx, ry, rz, x, y, z]
        grasp_approach_constraint_in_goal_frame (bool): Whether grasp approach
            constraints are specified in the goal frame (True) or world
            frame (False)
        retract_constraint_in_goal_frame (bool): Whether retract constraints
            are specified in the goal frame (True) or world frame (False)
        time_dilation_factor (float): Factor to dilate trajectory execution
            time. Lower values result in slower execution.
        disable_collision_links (list[str]): Links to disable collision
            checking for (e.g., gripper finger links)
        object_frame (str): TF frame of the object being grasped.
            Defaults to 'detected_object1'.
        world_frame (str): World reference frame name.
            Defaults to 'base_link'.
        mesh_resource (str): Path to the object mesh file for collision
            clearing. Defaults to ''.
        object_shape (str): Shape for AABB clearing ('SPHERE', 'CUBOID',
            or 'CUSTOM_MESH'). Defaults to ''.
        object_scale (list[float]): Scale factors for AABB clearing shape
            [x, y, z]. Defaults to [0.1, 0.1, 0.1].
        enable_aabb_clearing (bool): Whether to enable AABB clearing during
            planning. Defaults to False.
        object_esdf_clearing_padding (list[float]): Padding for ESDF clearing
            [x, y, z]. Defaults to [0.05, 0.05, 0.05].
        update_planning_scene (bool): Whether to include the MoveIt planning
            scene for collision checking. Defaults to False.
        planning_scene_world (PlanningSceneWorld): The MoveIt planning scene
            world data. Required if update_planning_scene is True.
        timeout (float): Maximum time to wait for the result in seconds.
            Defaults to 120.0.

    Returns
    -------
        Any: The MotionPlan.Result if successful, None on failure.
            On success, result.planned_trajectory contains two trajectories:
            [0] approach-to-grasp, [1] grasp-to-retract.
            result.goal_index indicates which grasp pose was selected.

    """
    if object_scale is None:
        object_scale = [0.1, 0.1, 0.1]
    if object_esdf_clearing_padding is None:
        object_esdf_clearing_padding = [0.05, 0.05, 0.05]

    goal_msg = MotionPlan.Goal()
    goal_msg.goal_pose = goal_poses
    goal_msg.goal_pose.header.frame_id = link_name
    goal_msg.use_current_state = True
    goal_msg.plan_cspace = False
    goal_msg.plan_pose = False
    goal_msg.plan_grasp = True
    goal_msg.plan_approach_to_grasp = plan_approach_to_grasp
    goal_msg.plan_grasp_to_retract = plan_grasp_to_retract
    goal_msg.time_dilation_factor = time_dilation_factor
    goal_msg.hold_partial_pose = False

    # Grasp approach offset (orientation kept as identity)
    grasp_offset_pose = Pose()
    grasp_offset_pose.position.x = grasp_approach_offset_distance[0]
    grasp_offset_pose.position.y = grasp_approach_offset_distance[1]
    grasp_offset_pose.position.z = grasp_approach_offset_distance[2]
    grasp_offset_pose.orientation.w = 1.0
    goal_msg.grasp_offset_pose = grasp_offset_pose
    goal_msg.grasp_translation_path_deviation_limit = grasp_translation_path_deviation_limit
    goal_msg.grasp_translation_terminal_deviation_limit = \
        grasp_translation_terminal_deviation_limit
    goal_msg.grasp_enable_orientation_path_axis_constraint = \
        grasp_enable_orientation_path_axis_constraint
    goal_msg.grasp_orientation_path_axis_deviation_limit = \
        grasp_orientation_path_axis_deviation_limit
    goal_msg.grasp_orientation_terminal_deviation_limit = \
        grasp_orientation_terminal_deviation_limit

    # Retract offset (orientation kept as identity)
    retract_offset_pose = Pose()
    retract_offset_pose.position.x = retract_offset_distance[0]
    retract_offset_pose.position.y = retract_offset_distance[1]
    retract_offset_pose.position.z = retract_offset_distance[2]
    retract_offset_pose.orientation.w = 1.0
    goal_msg.retract_offset_pose = retract_offset_pose
    # Constrain the grasp-to-retract phase to a straight Cartesian line AND hold
    # the tool orientation fixed along that line, so the plug lifts straight up
    # (world +Z) instead of reorienting the wrist and dipping. Mirrors the tuned
    # behavior-tree pick config. A negative value means "use the optimizer
    # default" for that particular limit.
    goal_msg.retract_translation_path_deviation_limit = retract_translation_path_deviation_limit
    goal_msg.retract_translation_terminal_deviation_limit = \
        retract_translation_terminal_deviation_limit
    goal_msg.retract_enable_orientation_path_axis_constraint = \
        retract_enable_orientation_path_axis_constraint
    goal_msg.retract_orientation_path_axis_deviation_limit = \
        retract_orientation_path_axis_deviation_limit
    goal_msg.retract_orientation_terminal_deviation_limit = \
        retract_orientation_terminal_deviation_limit

    goal_msg.grasp_approach_constraint_in_goal_frame = \
        grasp_approach_constraint_in_goal_frame
    goal_msg.retract_constraint_in_goal_frame = retract_constraint_in_goal_frame
    goal_msg.object_frame = object_frame
    goal_msg.world_frame = world_frame
    goal_msg.mesh_resource = mesh_resource
    goal_msg.update_esdf = True
    goal_msg.enable_aabb_clearing = enable_aabb_clearing
    goal_msg.clear_esdf = enable_aabb_clearing
    goal_msg.object_esdf_clearing_padding = object_esdf_clearing_padding

    # Object shape and scale for AABB clearing
    goal_msg.object_shape = object_shape
    scale = Vector3()
    scale.x = object_scale[0]
    scale.y = object_scale[1]
    scale.z = object_scale[2]
    goal_msg.object_scale = scale

    # Planning scene integration
    goal_msg.use_planning_scene = update_planning_scene
    if update_planning_scene and planning_scene_world is not None:
        goal_msg.world = planning_scene_world

    return send_motion_plan_goal(action_client, logger, goal_msg, timeout)


def plan_to_pose(action_client: ActionClient,
                 logger: Logger,
                 goal_pose: Pose,
                 link_name: str,
                 time_dilation_factor: float,
                 disable_collision_links: Optional[List[str]] = None,
                 mesh_resource: str = '',
                 object_shape: str = '',
                 object_scale: Optional[List[float]] = None,
                 enable_aabb_clearing: bool = False,
                 object_esdf_clearing_padding: Optional[List[float]] = None,
                 update_planning_scene: bool = False,
                 planning_scene_world: Optional[PlanningSceneWorld] = None,
                 timeout: float = 120.0) -> Any:
    """
    Plan a motion to a target Cartesian pose.

    Constructs a MotionPlan goal for Cartesian pose planning and sends it
    to the cuMotion action server. The goal pose is wrapped in a PoseArray
    for compatibility with the MotionPlan action interface.

    Args
    ----
        action_client (ActionClient): The action client for cuMotion motion
            planning (MotionPlan action type)
        logger (Logger): The logger to use
        goal_pose (Pose): Target end-effector pose in the specified link frame
        link_name (str): Reference frame for the goal pose (e.g., 'base_link')
        time_dilation_factor (float): Factor to dilate trajectory execution
            time. Lower values result in slower execution.
        disable_collision_links (list[str]): Links to disable collision
            checking for. Defaults to empty list.
        mesh_resource (str): Path to the object mesh file for collision
            clearing. Defaults to ''.
        object_shape (str): Shape for AABB clearing ('SPHERE', 'CUBOID',
            etc.). Defaults to ''.
        object_scale (list[float]): Scale factors for AABB clearing shape
            [x, y, z]. Defaults to [0.1, 0.1, 0.1].
        enable_aabb_clearing (bool): Whether to enable AABB clearing during
            planning. Defaults to False.
        object_esdf_clearing_padding (list[float]): Padding for ESDF clearing
            [x, y, z]. Defaults to [0.05, 0.05, 0.05].
        update_planning_scene (bool): Whether to include the MoveIt planning
            scene for collision checking. Defaults to False.
        planning_scene_world (PlanningSceneWorld): The MoveIt planning scene
            world data. Required if update_planning_scene is True.
        timeout (float): Maximum time to wait for the result in seconds.
            Defaults to 120.0.

    Returns
    -------
        Any: The MotionPlan.Result if successful, None on failure.
            On success, result.planned_trajectory[0] contains the trajectory.

    """
    if disable_collision_links is None:
        disable_collision_links = []
    if object_scale is None:
        object_scale = [0.1, 0.1, 0.1]
    if object_esdf_clearing_padding is None:
        object_esdf_clearing_padding = [0.05, 0.05, 0.05]

    goal_msg = MotionPlan.Goal()
    goal_msg.goal_pose = PoseArray()
    goal_msg.goal_pose.header.frame_id = link_name
    goal_msg.goal_pose.poses.append(goal_pose)
    goal_msg.plan_pose = True
    goal_msg.plan_cspace = False
    goal_msg.plan_grasp = False
    goal_msg.use_current_state = True
    goal_msg.time_dilation_factor = time_dilation_factor
    goal_msg.hold_partial_pose = False
    goal_msg.mesh_resource = mesh_resource
    goal_msg.update_esdf = True
    goal_msg.enable_aabb_clearing = enable_aabb_clearing
    goal_msg.clear_esdf = enable_aabb_clearing
    goal_msg.object_esdf_clearing_padding = object_esdf_clearing_padding

    # Object shape and scale for AABB clearing
    goal_msg.object_shape = object_shape
    scale = Vector3()
    scale.x = object_scale[0]
    scale.y = object_scale[1]
    scale.z = object_scale[2]
    goal_msg.object_scale = scale

    # Planning scene integration
    goal_msg.use_planning_scene = update_planning_scene
    if update_planning_scene and planning_scene_world is not None:
        goal_msg.world = planning_scene_world

    return send_motion_plan_goal(action_client, logger, goal_msg, timeout)


def plan_to_joint_state(action_client: ActionClient,
                        logger: Logger,
                        goal_state: JointState,
                        time_dilation_factor: float,
                        disable_collision_links: Optional[List[str]] = None,
                        mesh_resource: str = '',
                        object_shape: str = '',
                        object_scale: Optional[List[float]] = None,
                        enable_aabb_clearing: bool = False,
                        object_esdf_clearing_padding: Optional[List[float]] = None,
                        update_planning_scene: bool = False,
                        planning_scene_world: Optional[PlanningSceneWorld] = None,
                        timeout: float = 120.0) -> Any:
    """
    Plan a motion to a target joint state (configuration space planning).

    Constructs a MotionPlan goal for joint-space planning and sends it
    to the cuMotion action server. This is used when a specific joint
    configuration is desired rather than a Cartesian pose.

    Args
    ----
        action_client (ActionClient): The action client for cuMotion motion
            planning (MotionPlan action type)
        logger (Logger): The logger to use
        goal_state (JointState): Target joint state with position and joint
            names fields populated
        time_dilation_factor (float): Factor to dilate trajectory execution
            time. Lower values result in slower execution.
        disable_collision_links (list[str]): Links to disable collision
            checking for. Defaults to empty list.
        mesh_resource (str): Path to the object mesh file for collision
            clearing. Defaults to ''.
        object_shape (str): Shape for AABB clearing ('SPHERE', 'CUBOID',
            etc.). Defaults to ''.
        object_scale (list[float]): Scale factors for AABB clearing shape
            [x, y, z]. Defaults to [0.1, 0.1, 0.1].
        enable_aabb_clearing (bool): Whether to enable AABB clearing during
            planning. Defaults to False.
        object_esdf_clearing_padding (list[float]): Padding for ESDF clearing
            [x, y, z]. Defaults to [0.05, 0.05, 0.05].
        update_planning_scene (bool): Whether to include the MoveIt planning
            scene for collision checking. Defaults to False.
        planning_scene_world (PlanningSceneWorld): The MoveIt planning scene
            world data. Required if update_planning_scene is True.
        timeout (float): Maximum time to wait for the result in seconds.
            Defaults to 120.0.

    Returns
    -------
        Any: The MotionPlan.Result if successful, None on failure.
            On success, result.planned_trajectory[0] contains the trajectory.

    """
    if disable_collision_links is None:
        disable_collision_links = []
    if object_scale is None:
        object_scale = [0.1, 0.1, 0.1]
    if object_esdf_clearing_padding is None:
        object_esdf_clearing_padding = [0.05, 0.05, 0.05]

    goal_msg = MotionPlan.Goal()
    goal_msg.goal_state = goal_state
    goal_msg.plan_cspace = True
    goal_msg.plan_pose = False
    goal_msg.plan_grasp = False
    goal_msg.use_current_state = True
    goal_msg.time_dilation_factor = time_dilation_factor
    goal_msg.hold_partial_pose = False
    goal_msg.mesh_resource = mesh_resource
    goal_msg.update_esdf = True
    goal_msg.enable_aabb_clearing = enable_aabb_clearing
    goal_msg.clear_esdf = enable_aabb_clearing
    goal_msg.object_esdf_clearing_padding = object_esdf_clearing_padding

    # Object shape and scale
    goal_msg.object_shape = object_shape
    scale = Vector3()
    scale.x = object_scale[0]
    scale.y = object_scale[1]
    scale.z = object_scale[2]
    goal_msg.object_scale = scale

    # Planning scene integration
    goal_msg.use_planning_scene = update_planning_scene
    if update_planning_scene and planning_scene_world is not None:
        goal_msg.world = planning_scene_world

    return send_motion_plan_goal(action_client, logger, goal_msg, timeout)


def execute_trajectory(action_client: ActionClient,
                       logger: Logger,
                       robot_trajectory: RobotTrajectory,
                       timeout: float = 120.0) -> Tuple[bool, Any]:
    """
    Execute a planned trajectory via the /execute_trajectory action server.

    Sends a planned trajectory to the MoveIt ExecuteTrajectory action server
    and waits for execution to complete. Uses the same send-and-wait paradigm
    as the planning functions.

    Note: This function does not perform joint state validation before
    execution (unlike CumotionGoalSetClient.execute_plan). The caller is
    responsible for ensuring the robot's current state is compatible with
    the trajectory start state.

    Args
    ----
        action_client (ActionClient): The action client for trajectory
            execution (ExecuteTrajectory action type)
        logger (Logger): The logger to use
        robot_trajectory (RobotTrajectory): The trajectory to execute,
            typically from MotionPlan.Result.planned_trajectory
        timeout (float): Maximum time to wait for execution in seconds.
            Defaults to 120.0.

    Returns
    -------
        tuple[bool, Any]: A tuple of (success, result).
            success is True if trajectory was executed, False otherwise.
            result is the ExecuteTrajectory.Result or None on failure.

    """
    logger.info('Waiting for execute trajectory action server...')
    if not action_client.wait_for_server(timeout_sec=10.0):
        logger.error(
            'Execute trajectory action server not available after waiting')
        return False, None

    goal_msg = ExecuteTrajectory.Goal()
    goal_msg.trajectory = robot_trajectory
    goal_msg.trajectory.joint_trajectory.header = Header()
    goal_msg.trajectory.multi_dof_joint_trajectory.header = Header()

    logger.info('Sending trajectory for execution...')
    future = action_client.send_goal_async(goal_msg)

    try:
        result = wait_for_result(future, logger, timeout=timeout)
    except RuntimeError as e:
        logger.error(f'Execute trajectory goal failed: {e}')
        return False, None

    if result is False or result is None:
        logger.error('Failed to execute trajectory')
        return False, None

    logger.info('Trajectory execution completed')
    return True, result


def find_closest_joint_state(target_joint_position: np.ndarray,
                             candidate_joint_state: np.ndarray) -> float:
    """
    Find the closest joint state to target positions.

    Args
    ----
        target_joint_position [dof]: Target joint position
        candidate_joint_state [dof]: Candidate joint state

    Returns
    -------
        Distance between the target and candidate joint states

    """
    assert len(target_joint_position) == len(candidate_joint_state), \
        f'Target joint position: {target_joint_position} and candidate joint' \
        f' state: {candidate_joint_state}have different lengths'

    target_position = np.array(target_joint_position)
    dist = np.linalg.norm(target_position - candidate_joint_state)

    return dist


def get_sorted_indexes_of_closest_joint_states(ik_possible_joint_states: List[JointState],
                                               target_joint_state: JointState,
                                               joint_limits: Dict[str, Tuple[float, float]]
                                               ) -> List[int]:
    """
    Get sorted indexes in order (closest to furthest) from the target joint state.

    Args
    ----
        ik_possible_joint_states: List of possible joint states
        target_joint_state: Target joint state
        joint_limits: Joint limits, the dict is key and the value is a tuple of (min, max)

    Returns
    -------
        List of indexes of the closest joint states

    """
    distances_and_indexes = []
    target_positions = np.array(target_joint_state.position)

    # Make sure joint limits have all the same keys as names and have a tuple of floats.
    assert set(joint_limits.keys()) == set(target_joint_state.name), \
        f'Joint limits: {joint_limits} and target joint state: {target_joint_state}' \
        f'have different keys: {set(joint_limits.keys()) - set(target_joint_state.name)}'

    for joint_name, joint_limit in joint_limits.items():
        assert isinstance(joint_limit, tuple) and len(joint_limit) == 2, \
            f'Joint limits for joint name: {joint_name} are not a tuple of length 2'

    target_name_list = list(target_joint_state.name)

    for i, joint_state in enumerate(ik_possible_joint_states):
        current_positions = np.array(joint_state.position)
        assert len(current_positions) == len(target_joint_state.position), \
            f'Current positions: {current_positions} and target positions' \
            f': {target_joint_state.position}' \
            f'have different lengths'

        if list(joint_state.name) != target_name_list:
            name_to_pos = dict(zip(joint_state.name, joint_state.position))
            current_positions = np.array(
                [name_to_pos[n] for n in target_name_list])

        distance = find_closest_joint_state(
            target_positions, current_positions)

        distances_and_indexes.append((distance, i))

    sorted_indexes = [index for _, index in sorted(distances_and_indexes)]
    return sorted_indexes
