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

import collections

from isaac_ros_manipulation_orchestration.utils.params_loader import (
    BlackboardConfig, RetryConfig, ServerTimeoutConfig
)
from isaac_ros_manipulation_ros_python_utils.grasp_reader import GraspReaderManager
import py_trees


class BlackboardInitializer:
    """Initialize blackboard variables from configuration files."""

    def __init__(
        self,
        blackboard_params_file: str,
        package_name: str
    ):
        """
        Initialize the blackboard configuration initializer.

        Args:
        ----
        blackboard_params_file (str): Path to the blackboard parameters YAML file.
            Expected to contain all necessary configuration values for initialization.
        package_name (str): Name of the package to load parameters from.

        """
        self.config = BlackboardConfig(blackboard_params_file, package_name)

    def initialize_blackboard(self) -> py_trees.blackboard.Client:
        """
        Create and populate a blackboard client with all configuration values.

        Returns
        -------
            py_trees.blackboard.Client: Configured blackboard client instance with all
                parameters registered and set from the configuration file.

        """
        blackboard = py_trees.blackboard.Client(name='ConfigClient')

        blackboard.register_key(
            key='next_object_id',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.next_object_id = self.config.next_object_id

        blackboard.register_key(
            key='selected_object_id',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.selected_object_id = self.config.selected_object_id

        blackboard.register_key(
            key='object_info_cache',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.object_info_cache = self.config.object_info_cache

        blackboard.register_key(
            key='max_num_next_object',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_num_next_object = self.config.max_num_next_object

        blackboard.register_key(
            key='use_drop_pose_from_rviz',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.use_drop_pose_from_rviz = self.config.use_drop_pose_from_rviz

        blackboard.register_key(
            key='rviz_drop_pose',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.rviz_drop_pose = self.config.rviz_drop_pose

        blackboard.register_key(
            key='abort_motion',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.abort_motion = self.config.abort_motion

        blackboard.register_key(
            key='active_obj_id',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.active_obj_id = self.config.active_obj_id

        blackboard.register_key(
            key='goal_drop_pose',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.goal_drop_pose = self.config.goal_drop_pose

        blackboard.register_key(
            key='home_pose',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.home_pose = self.config.home_pose

        blackboard.register_key(
            key='grasp_reader_manager',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.grasp_reader_manager = GraspReaderManager(
            self.config.supported_objects_config
        )

        blackboard.register_key(
            key='mesh_file_paths',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.mesh_file_paths = {
            obj.class_id: obj.mesh_file_path
            for obj in self.config.supported_objects_config.supported_objects.values()
        }

        blackboard.register_key(
            key='supported_class_ids',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.supported_class_ids = {
            obj.class_id for obj in self.config.supported_objects_config.supported_objects.values()
        }

        blackboard.register_key(
            key='target_poses',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.target_poses = self.config.target_poses

        blackboard.register_key(
            key='class_ids',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.class_ids = self.config.class_ids

        blackboard.register_key(
            key='mode',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.mode = self.config.mode

        # Status reporting variables
        blackboard.register_key(
            key='workflow_status',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.workflow_status = self.config.workflow_status

        blackboard.register_key(
            key='workflow_summary',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.workflow_summary = self.config.workflow_summary

        blackboard.register_key(
            key='workflow_feedback_queue',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.workflow_feedback_queue = collections.deque()

        # The action server sets this after preparing a goal and clears it
        # before releasing that goal. Behavior-tree guards use it to keep
        # goal-dependent work disabled while no goal is active.
        blackboard.register_key(
            key='action_goal_active',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.action_goal_active = False

        return blackboard

    def setup_retry_configuration(self, retry_config: RetryConfig):
        """
        Store retry parameters on the blackboard.

        Args:
        ----
        retry_config (RetryConfig): Retry configuration containing all retry parameters.

        """
        blackboard = py_trees.blackboard.Client(name='RetryConfigClient')

        blackboard.register_key(
            key='max_planning_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_planning_retries = retry_config.max_planning_retries

        blackboard.register_key(
            key='max_controller_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_controller_retries = retry_config.max_controller_retries

        blackboard.register_key(
            key='max_detection_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_detection_retries = retry_config.max_detection_retries

        blackboard.register_key(
            key='max_pose_estimation_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_pose_estimation_retries = retry_config.max_pose_estimation_retries

        blackboard.register_key(
            key='max_gripper_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_gripper_retries = retry_config.max_gripper_retries

        blackboard.register_key(
            key='max_attachment_retries',
            access=py_trees.common.Access.WRITE,
        )
        blackboard.max_attachment_retries = retry_config.max_attachment_retries

    def setup_server_timeout_configuration(self, server_timeout_config: ServerTimeoutConfig):
        """
        Store server timeout parameters on the blackboard.

        Args:
        ----
        server_timeout_config (ServerTimeoutConfig): Server timeout configuration containing
            all timeout parameters.

        Raises
        ------
        ValueError: If server_timeout_config is None or contains invalid values.

        """
        if server_timeout_config is None:
            raise ValueError('server_timeout_config cannot be None')

        blackboard = py_trees.blackboard.Client(name='ServerTimeoutConfigClient')

        blackboard.register_key(
            key='server_timeout_config',
            access=py_trees.common.Access.WRITE,
        )

        # Convert ServerTimeoutConfig object to dictionary format as expected by base classes
        # This format is validated against BaseActionBehavior and BaseServiceBehavior usage
        timeout_dict = {
            'startup_server_timeout_sec': server_timeout_config.startup_server_timeout_sec,
            'runtime_retry_timeout_sec': server_timeout_config.runtime_retry_timeout_sec,
            'server_check_interval_sec': server_timeout_config.server_check_interval_sec
        }

        # Validate required timeout values are present and valid
        if (timeout_dict['runtime_retry_timeout_sec'] is None or
                timeout_dict['runtime_retry_timeout_sec'] <= 0):
            raise ValueError('runtime_retry_timeout_sec must be a positive number')
        if (timeout_dict['server_check_interval_sec'] is None or
                timeout_dict['server_check_interval_sec'] <= 0):
            raise ValueError('server_check_interval_sec must be a positive number')

        blackboard.server_timeout_config = timeout_dict
