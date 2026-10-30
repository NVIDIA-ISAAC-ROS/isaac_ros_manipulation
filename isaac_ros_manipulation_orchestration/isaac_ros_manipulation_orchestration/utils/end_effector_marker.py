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

from dataclasses import dataclass
import threading

from geometry_msgs.msg import Pose
from interactive_markers import InteractiveMarkerServer
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import InteractiveMarkerFeedback
from visualization_msgs.msg import Marker


@dataclass
class MarkerProperties:
    """Config to track marker properties."""

    color: ColorRGBA
    frame_id: str
    mesh_resource_uri: str
    name: str
    pose: Pose
    scale: float


class EndEffectorMarker:
    """
    Class for creating interactive marker.

    This class creates an interactive marker to vizualize the end effector.
    This helps in visualizing the end effector pose.
    """

    def __init__(self, node: Node, marker_namespace: str, mesh_resource_uri: str):
        self._node = node  # Store node reference for logging
        self._lock = threading.Lock()
        self._marker_prop = MarkerProperties(
            color=ColorRGBA(r=0.1, g=0.7, b=0.2, a=1.0),
            frame_id='base_link',
            mesh_resource_uri=mesh_resource_uri,
            name='End effector marker',
            pose=Pose(),
            scale=0.2
        )

        # RViz 15.2.6 dereferences a null resource-retriever owner when a
        # MESH_RESOURCE is embedded in an InteractiveMarkerControl.
        # Using CUBE type until upstream RViz bug is fixed.
        self.marker_type = Marker.CUBE
        if mesh_resource_uri:
            node.get_logger().warning(
                'RViz cannot safely render mesh resources in interactive markers; '
                'using a cube marker')

        self._interactive_marker_server = InteractiveMarkerServer(self._node, marker_namespace)
        self.make_gripper_marker()
        self._interactive_marker_server.applyChanges()

    def make_gripper_marker(self):
        """Create interactive marker."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._marker_prop.frame_id
        int_marker.scale = self._marker_prop.scale
        int_marker.name = self._marker_prop.name

        # gripper
        marker = Marker()
        marker.type = self.marker_type
        marker.color = self._marker_prop.color
        marker.scale.x = self._marker_prop.scale * 0.5
        marker.scale.y = self._marker_prop.scale * 0.5
        marker.scale.z = self._marker_prop.scale * 0.5
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers = [marker]
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        int_marker.controls.append(control)

        # add axes
        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        control.name = 'rotate_x'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 0.0
        control.name = 'move_x'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 0.707
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 0.707
        control.name = 'rotate_y'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 0.707
        control.orientation.x = 0.0
        control.orientation.y = 0.0
        control.orientation.z = 0.707
        control.name = 'move_y'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 0.707
        control.orientation.x = 0.0
        control.orientation.y = 0.707
        control.orientation.z = 0.0
        control.name = 'rotate_z'
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.orientation.w = 0.707
        control.orientation.x = 0.0
        control.orientation.y = 0.707
        control.orientation.z = 0.0
        control.name = 'move_z'
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)

        self._interactive_marker_server.insert(int_marker, feedback_callback=self.process_feedback)

    def process_feedback(self, feedback):
        """Extract pose from interactive marker."""
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            with self._lock:
                self._marker_prop.pose = feedback.pose

    def set_pose(self, pose: Pose):
        """Set pose of the marker."""
        try:
            with self._lock:
                self._marker_prop.pose = pose
            self._interactive_marker_server.setPose(name=self._marker_prop.name, pose=pose)
            self._interactive_marker_server.applyChanges()
        except Exception as e:
            self._node.get_logger().error(
                f'[EndEffectorMarker] Error in setting pose of interactive marker: {e}')

    def get_pose(self):
        """Get pose of the marker."""
        with self._lock:
            return self._marker_prop.pose
