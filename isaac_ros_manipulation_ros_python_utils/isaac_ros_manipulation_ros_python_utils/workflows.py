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

from typing import List

from isaac_ros_manipulation_ros_python_utils.config import (
    DisplayPortInsertionConfig, WorkflowType
)

from launch.actions import Shutdown
from launch_ros.actions import Node


def get_pick_and_place_orchestrator(
    workflow_config: DisplayPortInsertionConfig
) -> List[Node]:
    """
    Get the pick and place orchestrator.

    Args
    ----
        workflow_config (WorkflowConfig) : WorkflowConfig that dictates Isaac Sim workflows

    Returns
    -------
        List[Node] : ROS node for Pick and place

    """
    if workflow_config.workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
        object_frame_name = 'displayport_cable_frame'
    elif workflow_config.use_ground_truth_pose_in_sim:
        object_frame_name = workflow_config.sim_gt_asset_frame_id
    else:
        object_frame_name = 'detected_object1'
    # Robot- and gripper-specific values live on ``driver_config``; workflow-level
    # knobs (retries, scales, attachments, etc.) stay on ``workflow_config``.
    driver_config = workflow_config.driver_config
    grasp_approach_offset = [0.0, 0.0, -0.15]
    pick_retract_offset = getattr(workflow_config, 'pick_retract_offset_distance', [])
    if pick_retract_offset:
        retract_offset = pick_retract_offset
    else:
        retract_offset = [0.0, 0.0, 0.15]
    pick_and_place_orchestrator_node = Node(
        package='isaac_ros_manipulation_contact_rich_insertion',
        executable='pick_and_place_orchestrator.py',
        name='pick_and_place_orchestrator',
        parameters=[{
            'grasp_file_path': workflow_config.grasps_file_path,
            'isaac_sim': workflow_config.use_sim_time,
            'use_sim_time': workflow_config.use_sim_time,
            'time_dilation_factor': float(workflow_config.time_dilation_factor),
            'retract_offset_distance': retract_offset,
            'grasp_approach_offset_distance': grasp_approach_offset,
            'object_frame_name': object_frame_name,
            'use_ground_truth_pose_from_sim': workflow_config.use_ground_truth_pose_in_sim,
            'sleep_time_before_planner_tries_sec': workflow_config.pick_and_place_retry_wait_time,
            'num_planner_tries': workflow_config.pick_and_place_planner_retries,
            'publish_grasp_frame': True,
            'use_cartesian_pickup_retract':
                getattr(workflow_config, 'use_cartesian_pickup_retract', False),
            # Vendor drivers name the arm planning group differently; Flexiv v3.0
            # uses '<robot_sn>_arm'. Fall back to the driver's derived name before
            # the historical UR-era default.
            'cartesian_retract_group_name':
                getattr(workflow_config, 'cartesian_retract_group_name', None)
                or getattr(driver_config, 'planning_group_name', 'rizon_arm'),
            'disable_pick_planning_scene':
                getattr(workflow_config, 'disable_pick_planning_scene', False),
            'disable_place_planning_scene':
                getattr(workflow_config, 'disable_place_planning_scene', False),
            'gripper_collision_links': driver_config.gripper_collision_links,
            'attach_object_shape': workflow_config.object_attachment_type.value,
            'attach_object_scale': workflow_config.object_attachment_scale,
            'attach_object_mesh_file_path': workflow_config.attach_object_mesh_file_path,
            'grasp_approach_in_world_frame': False,
            'retract_in_world_frame': True,
            'use_pose_from_rviz': workflow_config.use_pose_from_rviz,
            'end_effector_mesh_resource_uri': workflow_config.end_effector_mesh_resource_uri,
            'move_to_home_pose_after_place': workflow_config.move_to_home_pose_after_place,
            'home_pose': workflow_config.home_pose,
            'seed_state_for_ik_solver_for_joint_space_planner':
                workflow_config.seed_state_for_ik_solver_for_joint_space_planner,
            'use_exact_target_joint_state_for_place_pose':
                getattr(workflow_config, 'use_exact_joint_state_place_pose', False),
            'base_frame': driver_config.base_frame,
            'gripper_frame': driver_config.gripper_frame,
            'grasp_frame': driver_config.grasp_frame,
            'gripper_action_name': driver_config.gripper_action_name,
            'gripper_open_position': driver_config.gripper_open_position,
            'gripper_close_position': driver_config.gripper_close_position,
            'gripper_settle_time_sec': driver_config.gripper_settle_time_sec,
            'gripper_max_effort': driver_config.gripper_max_effort,
            'joint_names': driver_config.joint_names,
            'joint_limits_file_path': driver_config.joint_limits_file_path,
        }],
        output='screen',
        on_exit=Shutdown(),
    )
    return [pick_and_place_orchestrator_node]
