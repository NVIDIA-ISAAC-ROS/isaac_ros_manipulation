#!/usr/bin/env python3
# flake8: noqa: E402
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
from tf2_ros import Buffer
from isaac_ros_manipulation_ros_python_utils.grasp_reader import GraspReader, Transformation
from scipy.spatial.transform import Rotation as R
import pytest
import numpy as np
import launch_testing
from launch_ros.actions import Node
import launch
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from unittest.mock import MagicMock, mock_open, patch
import unittest
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_description():
    """Generate a test description for launch_test."""
    return launch.LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'world', '--child-frame-id', 'base_link',
            ]
        ),
        launch_testing.actions.ReadyToTest()
    ])


class TestGraspReader(unittest.TestCase):
    """Test class for GraspReader functionality."""

    def setUp(self):
        """Create a GraspReader fixture with mocked grasp data."""
        self.mock_yaml_data = {
            'grasps': {
                'grasp1': {
                    'position': [0.1, 0.2, 0.3],
                    'orientation': {
                        'xyz': [0.0, 0.0, 0.0],
                        'w': 1.0
                    }
                },
                'grasp2': {
                    'position': [0.4, 0.5, 0.6],
                    'orientation': {
                        'xyz': [0.0, 0.7071, 0.0],
                        'w': 0.7071
                    }
                }
            }
        }

        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', return_value=self.mock_yaml_data):
                self.grasp_reader = GraspReader('dummy_path.yaml')

    def test_get_transformation_matrix(self):
        """Test get_transformation_matrix with identity quaternion."""
        quaternion = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        translation = [1.0, 2.0, 3.0]

        matrix = self.grasp_reader.get_transformation_matrix(quaternion, translation)

        expected = np.eye(4)
        expected[:3, 3] = translation

        np.testing.assert_array_almost_equal(matrix, expected)

    def test_get_transformation_matrix_with_rotation(self):
        """Test get_transformation_matrix with a 90-degree rotation around Y axis."""
        # 90-degree rotation around Y axis
        quaternion = {'x': 0.0, 'y': 0.7071068, 'z': 0.0, 'w': 0.7071068}
        translation = [1.0, 2.0, 3.0]

        matrix = self.grasp_reader.get_transformation_matrix(quaternion, translation)

        # Expected rotation matrix for 90-degree rotation around Y
        rot_y_90 = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [-1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        np.testing.assert_array_almost_equal(matrix, rot_y_90)

    def test_get_transformation_matrix_from_transformation(self):
        """Test get_transformation_matrix_from_transformation function."""
        transform_obj = Transformation(
            quaternion={'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            position=[1.0, 2.0, 3.0]
        )

        matrix = self.grasp_reader.get_transformation_matrix_from_transformation(transform_obj)

        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]

        np.testing.assert_array_almost_equal(matrix, expected)

    def test_matrix_to_ros_translation_quaternion(self):
        """Test matrix_to_ros_translation_quaternion function."""
        # Create a test matrix (90-degree rotation around X with translation)
        r = R.from_euler('x', 90, degrees=True)
        rot_matrix = r.as_matrix()

        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = [1.0, 2.0, 3.0]

        translation, quaternion = self.grasp_reader.matrix_to_ros_translation_quaternion(matrix)

        # Check translation
        assert translation.x == pytest.approx(1.0)
        assert translation.y == pytest.approx(2.0)
        assert translation.z == pytest.approx(3.0)

        # Check quaternion (representing 90-degree rotation around X)
        assert quaternion.x == pytest.approx(0.7071068, abs=1e-6)
        assert quaternion.y == pytest.approx(0.0, abs=1e-6)
        assert quaternion.z == pytest.approx(0.0, abs=1e-6)
        assert quaternion.w == pytest.approx(0.7071068, abs=1e-6)

    def test_get_transformation_matrix_from_ros(self):
        """Test get_transformation_matrix_from_ros_ function."""
        # Create a ROS transform
        transform = Transform()
        transform.translation = Vector3(x=1.0, y=2.0, z=3.0)
        transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        transform_stamped = TransformStamped()
        transform_stamped.transform = transform

        matrix = self.grasp_reader.get_transformation_matrix_from_ros_(transform_stamped.transform)

        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]

        np.testing.assert_array_almost_equal(matrix, expected)

    def test_round_trip_conversion(self):
        """Test round trip conversion from matrix to ROS and back."""
        # Start with a known matrix
        original_matrix = np.eye(4)
        # 45-degree rotation around Z
        rotation = R.from_euler('z', 45, degrees=True).as_matrix()
        original_matrix[:3, :3] = rotation
        original_matrix[:3, 3] = [1.5, 2.5, 3.5]

        # Convert to ROS objects
        translation, quaternion = self.grasp_reader.matrix_to_ros_translation_quaternion(
            original_matrix)

        # Create a Transform for the return conversion
        transform = Transform()
        transform.translation = translation
        transform.rotation = quaternion

        # Convert back to matrix
        reconstructed_matrix = self.grasp_reader.get_transformation_matrix_from_ros_(transform)

        # Check if the matrices are the same
        np.testing.assert_array_almost_equal(original_matrix, reconstructed_matrix)

    @patch('isaac_ros_manipulation_ros_python_utils.grasp_reader.lookup_transform_from_tf')
    def test_get_grasp_pose_object(self, mock_lookup_transform):
        """Test get_grasp_pose_object with mocked TF buffer."""
        # Mock TF lookup to return an identity transform
        transform = TransformStamped()
        transform.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        transform.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        mock_lookup_transform.return_value = transform

        # Call the function with index 0 (first grasp in our mock data)
        mock_tf_buffer = MagicMock(spec=Buffer)
        result = self.grasp_reader.get_grasp_pose_object(
            index=0,
            gripper_frame='gripper',
            grasp_frame='grasp',
            tf_buffer=mock_tf_buffer
        )

        # We check the negative of the expected values becuase the function returns the
        # grasp_pose_object which is the inverse of the actual grasp pose
        assert result.quaternion.x == pytest.approx(0.0)
        assert result.quaternion.y == pytest.approx(0.0)
        assert result.quaternion.z == pytest.approx(0.0)
        assert result.quaternion.w == pytest.approx(1.0)
        assert result.position.x == pytest.approx(-0.1)
        assert result.position.y == pytest.approx(-0.2)
        assert result.position.z == pytest.approx(-0.3)

    def test_get_pose_for_pick_task_from_cached_pose_with_rotation(self):
        """
        Test get_pose_for_pick_task_from_cached_pose with rotated object pose.

        This test validates both translation and rotation transformations,
        ensuring proper matrix math for grasp pose computation.
        """
        # Define a cached object pose with 90-degree rotation around Z-axis
        cached_pose = {
            'position': [1.0, 2.0, 3.0],
            'orientation': [0.0, 0.0, 0.7071068, 0.7071068]  # 90-degree rotation around Z
        }

        # Convert dict pose to 4x4 matrix
        quat_dict = {
            'x': cached_pose['orientation'][0],
            'y': cached_pose['orientation'][1],
            'z': cached_pose['orientation'][2],
            'w': cached_pose['orientation'][3]
        }
        world_pose_object = self.grasp_reader.get_transformation_matrix(
            quat_dict, cached_pose['position'])

        # Call the updated method with matrix
        result = self.grasp_reader.get_pose_for_pick_task_from_cached_pose(world_pose_object)

        # Verify we get the expected number of grasp poses
        assert len(result) == 2  # Two grasps in our mock data

        # For a 90-degree rotation around Z, the first grasp offset [0.1, 0.2, 0.3]
        # becomes [-0.2, 0.1, 0.3] in the rotated frame
        first_grasp = result[0]
        assert first_grasp.position.x == pytest.approx(0.8, abs=1e-6)  # 1.0 - 0.2
        assert first_grasp.position.y == pytest.approx(2.1, abs=1e-6)  # 2.0 + 0.1
        assert first_grasp.position.z == pytest.approx(3.3, abs=1e-6)  # 3.0 + 0.3
