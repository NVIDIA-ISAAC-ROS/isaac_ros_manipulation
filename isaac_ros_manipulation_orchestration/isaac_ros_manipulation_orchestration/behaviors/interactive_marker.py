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

import copy

from geometry_msgs.msg import Pose, PoseStamped
from isaac_ros_manipulation_orchestration.utils.end_effector_marker import EndEffectorMarker
from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees
import rclpy


class InteractiveMarker(py_trees.behaviour.Behaviour):
    """
    Create an interactive marker and update the blackboard with the end effector marker pose.

    This interactive marker in RViz allows users to manually specify target poses
    for the robot end effector. This behavior creates an EndEffectorMarker and publishes it
    to RViz, requiring 'tf_buffer' to be available on the blackboard for coordinate frame
    transformations. The marker pose is continuously updated on the blackboard as
    'rviz_drop_pose' for use by motion planning behaviors.

    The behavior includes a user confirmation period after marker initialization, allowing
    users time to adjust the drop pose before the workflow proceeds. During this period,
    warning messages are logged to inform users about the current drop pose location.
    The rviz_drop_pose is only set on the blackboard after the confirmation period expires
    or when the user actively modifies the marker pose.

    Parameters
    ----------
    name : str
        Name of the behavior
    mesh_resource_uri : str
        URI of the mesh resource to display for the interactive marker
    reference_frame : str
        Name of the reference frame with respect to which the pose is being provided
    end_effector_frame : str
        Name of the end effector frame to transform to
    user_confirmation_timeout : float, optional
        Time in seconds to wait for user confirmation before proceeding (default: 10.0)

    """

    # Class constant for warning interval - implementation detail, not user-configurable
    _WARNING_INTERVAL_SECONDS = 2.0

    def __init__(self, name: str, mesh_resource_uri: str, reference_frame: str = 'base_link',
                 end_effector_frame: str = 'gripper_frame',
                 user_confirmation_timeout: float = 10.0):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self._end_effector_marker = None
        self._mesh_resource_uri = mesh_resource_uri
        self._reference_frame = reference_frame
        self._end_effector_frame = end_effector_frame
        self._user_confirmation_timeout = user_confirmation_timeout

        # Tolerance for pose change detection (1mm for position, ~0.001 rad for orientation)
        self._change_tolerance = 1e-3
        # Track previous pose for change detection
        self._previous_pose = None

        # Timer and warning tracking for user confirmation period
        self._confirmation_timer = None
        self._last_warning_time = None
        self._initial_pose = None
        self._pose_confirmed = False  # Track if pose has been confirmed/set

        self.blackboard.register_key(
            key='tf_buffer', access=py_trees.common.Access.READ)
        self.blackboard.register_key(key='rviz_drop_pose', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key='is_drop_pose_updating',
                                     access=py_trees.common.Access.WRITE)
        self._state = BehaviorStatus.IDLE

    def setup(self, **kwargs):
        """
        Set up the behavior by creating the interactive marker.

        This is called once when the behavior tree is constructed.

        Returns
        -------
            bool: True if setup was successful

        Raises
        ------
            KeyError: If 'node' is not in kwargs

        """
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e

        self._end_effector_marker = EndEffectorMarker(
            self.node, 'end_effector_marker', self._mesh_resource_uri)

        return True

    def update(self):
        """
        Update blackboard goal drop pose with end effector marker pose as PoseStamped.

        Returns
        -------
            py_trees.common.Status:
            RUNNING if the behavior is in progress,
            SUCCESS if the behavior succeeded,
            FAILURE if the behavior failed

        """
        if self._state == BehaviorStatus.FAILED:
            self.node.get_logger().error(f'[{self.name}] Failed to initialize interactive marker')
            return py_trees.common.Status.FAILURE

        if self._state == BehaviorStatus.IDLE:
            return self._trigger_lookup()

        if self._state == BehaviorStatus.IN_PROGRESS:
            return py_trees.common.Status.RUNNING

        if self._state == BehaviorStatus.SUCCEEDED:
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(f'[{self.name}] Unexpected state: {self._state}')
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Clean up when this behavior finishes running."""
        self._state = BehaviorStatus.IDLE
        # Clean up timer if it exists
        if self._confirmation_timer is not None:
            self._confirmation_timer = None
        self._last_warning_time = None
        self._initial_pose = None
        self._pose_confirmed = False

    def _trigger_lookup(self):
        """Trigger the lookup of the end effector marker pose."""
        self._state = BehaviorStatus.IN_PROGRESS
        tf_buffer = self.blackboard.tf_buffer
        if tf_buffer is None:
            self.node.get_logger().error(
                'TF buffer is not initialized, unable to initialize the marker.')
            self._state = BehaviorStatus.FAILED
            return py_trees.common.Status.FAILURE

        future = tf_buffer.wait_for_transform_async(
            self._reference_frame, self._end_effector_frame, rclpy.time.Time())
        future.add_done_callback(self._initialize_marker)
        return py_trees.common.Status.RUNNING

    def _initialize_marker(self, future):
        """
        Initialize the interactive marker and start user confirmation period.

        This is called when the interactive marker is initialized.
        """
        if future.done():
            try:
                result = future.result()

                # Check if the reference frame matches our expected frame
                if result.header.frame_id != self._reference_frame:
                    self.node.get_logger().error(
                        f'[{self.name}] Frame mismatch: Expecting poses in '
                        f"'{self._reference_frame}' frame but RViz interactive marker "
                        f"returned pose in '{result.header.frame_id}' frame. ")
                    self._state = BehaviorStatus.FAILED
                    return

                # Extract pose from TransformStamped
                pose = Pose()
                pose.position.x = result.transform.translation.x
                pose.position.y = result.transform.translation.y
                pose.position.z = result.transform.translation.z
                pose.orientation.x = result.transform.rotation.x
                pose.orientation.y = result.transform.rotation.y
                pose.orientation.z = result.transform.rotation.z
                pose.orientation.w = result.transform.rotation.w

                self._end_effector_marker.set_pose(pose)

                # Store initial pose for confirmation period
                self._initial_pose = copy.deepcopy(pose)

                # Start user confirmation timer and warning system
                self._confirmation_timer = self.node.get_clock().now()
                self._last_warning_time = self.node.get_clock().now()

                # Log initial warning about drop pose assignment
                self.node.get_logger().warning(
                    f'[{self.name}] ATTENTION: Drop pose initialized to current '
                    f'{self._end_effector_frame} position: x={pose.position.x:.3f}, '
                    f'y={pose.position.y:.3f}, z={pose.position.z:.3f}. '
                    f'You have {self._user_confirmation_timeout} seconds to adjust the drop pose '
                    f'using the interactive marker in RViz before the workflow proceeds.')

                self._state = BehaviorStatus.SUCCEEDED
            except Exception as e:
                self.node.get_logger().error(f'Error getting transform result: {e}')
                self._state = BehaviorStatus.FAILED

        else:
            self.node.get_logger().error('Failed to initialize the marker.')
            self._state = BehaviorStatus.FAILED

    def _process_result(self):
        """
        Process the interactive marker result and update blackboard with PoseStamped.

        Only sets rviz_drop_pose after confirmation period expires or user modifies pose.

        Returns
        -------
        py_trees.common.Status
            RUNNING to continuously update the pose from the interactive marker

        """
        current_pose = self._end_effector_marker.get_pose()

        # Handle user confirmation period if still active
        if self._confirmation_timer is not None:
            self._handle_confirmation_period(current_pose)

        # Only set rviz_drop_pose on blackboard after confirmation period or user modification
        if self._pose_confirmed:
            self._update_blackboard_with_pose(current_pose)

        return py_trees.common.Status.RUNNING

    def _handle_confirmation_period(self, current_pose: Pose):
        """
        Handle the user confirmation period logic including timeout and warnings.

        Parameters
        ----------
        current_pose : Pose
            The current pose from the interactive marker

        """
        current_time = self.node.get_clock().now()
        elapsed_time = (current_time - self._confirmation_timer).nanoseconds / 1e9
        user_modified_pose = not self._poses_equal(
            current_pose, self._initial_pose)

        # User actively modified the pose - confirm immediately
        if user_modified_pose and not self._pose_confirmed:
            self.node.get_logger().info(
                f'[{self.name}] Drop pose modified by user. Confirming new drop location.')
            self._confirm_pose()
            return

        # Check if confirmation period has expired
        if elapsed_time >= self._user_confirmation_timeout:
            self._handle_confirmation_timeout()
            return

        # Still in confirmation period - issue periodic warnings
        self._issue_periodic_warnings(elapsed_time)

    def _confirm_pose(self):
        """Mark the pose as confirmed and end the confirmation period."""
        self._pose_confirmed = True
        self._confirmation_timer = None
        self._last_warning_time = None
        self._initial_pose = None

    def _handle_confirmation_timeout(self):
        """
        Handle the end of the confirmation timeout period.

        Since user modifications are handled earlier in the flow, we only reach
        this method when the timeout expires without user interaction.
        """
        if not self._pose_confirmed:
            self.node.get_logger().warning(
                f'[{self.name}] User confirmation period completed. '
                f'Drop pose was NOT modified - using current {self._end_effector_frame} '
                f'pose as drop location.')

        self._confirm_pose()

    def _issue_periodic_warnings(self, elapsed_time: float):
        """
        Issue periodic warning messages during the confirmation period.

        This method is only called when the user has NOT modified the pose,
        so we always issue warnings about the unchanged drop pose location.

        Parameters
        ----------
        elapsed_time : float
            Time elapsed since confirmation period started

        """
        current_time = self.node.get_clock().now()
        time_since_last_warning = (current_time - self._last_warning_time).nanoseconds / 1e9

        if time_since_last_warning >= self._WARNING_INTERVAL_SECONDS:
            remaining_time = self._user_confirmation_timeout - elapsed_time

            self.node.get_logger().warning(
                f'[{self.name}] WARNING: Drop pose is still at {self._end_effector_frame} '
                f'pose. Adjust the interactive marker in RViz to set the desired drop location. '
                f'Time remaining: {remaining_time:.1f} seconds.')

            self._last_warning_time = current_time

    def _update_blackboard_with_pose(self, current_pose: Pose):
        """
        Update the blackboard with the confirmed pose.

        Parameters
        ----------
        current_pose : Pose
            The current pose to set on the blackboard

        """
        # Convert Pose to PoseStamped before setting on blackboard
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self._reference_frame
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose = current_pose

        # Check if pose has changed since previous tick (indicates active dragging)
        pose_changed = not self._poses_equal(current_pose, self._previous_pose)

        # Update the blackboard
        self.blackboard.rviz_drop_pose = pose_stamped

        # Set is_drop_pose_updating based on whether user is actively changing the pose
        self.blackboard.is_drop_pose_updating = pose_changed

        # Log only when pose is actively being changed
        if pose_changed:
            self.node.get_logger().debug(
                f'{self.name}: Drop pose updated - '
                f'pos=({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, '
                f'{current_pose.position.z:.3f}) '
                f'ori=({current_pose.orientation.x:.3f}, {current_pose.orientation.y:.3f}, '
                f'{current_pose.orientation.z:.3f}, {current_pose.orientation.w:.3f})')

        # Update tracking
        self._previous_pose = copy.deepcopy(current_pose)

    def _poses_equal(self, pose1: Pose, pose2: Pose) -> bool:
        """
        Check if two poses are equal within tolerance.

        Parameters
        ----------
        pose1 : Pose
            First pose to compare
        pose2 : Pose
            Second pose to compare

        Returns
        -------
        bool
            True if poses are equal within tolerance, False otherwise

        """
        if pose1 is None or pose2 is None:
            return False

        # Direct position comparison (more reliable than using pose_difference)
        pos_diff_x = abs(pose1.position.x - pose2.position.x)
        pos_diff_y = abs(pose1.position.y - pose2.position.y)
        pos_diff_z = abs(pose1.position.z - pose2.position.z)

        position_changed = (pos_diff_x > self._change_tolerance or
                            pos_diff_y > self._change_tolerance or
                            pos_diff_z > self._change_tolerance)

        # Direct quaternion comparison (avoid pose_difference issues)
        ori_diff_x = abs(pose1.orientation.x - pose2.orientation.x)
        ori_diff_y = abs(pose1.orientation.y - pose2.orientation.y)
        ori_diff_z = abs(pose1.orientation.z - pose2.orientation.z)
        ori_diff_w = abs(pose1.orientation.w - pose2.orientation.w)

        orientation_changed = (ori_diff_x > self._change_tolerance or
                               ori_diff_y > self._change_tolerance or
                               ori_diff_z > self._change_tolerance or
                               ori_diff_w > self._change_tolerance)

        return not (position_changed or orientation_changed)
