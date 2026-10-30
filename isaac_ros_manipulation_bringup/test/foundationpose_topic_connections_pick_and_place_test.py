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
"""Tests the graph structure of RT-DETR and Foundation Pose in Pick and Placeworkflow."""

import os
from typing import Dict

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils import (
    FoundationPoseTopicConnectionsGraphTest,
    get_params_from_config_file_set_in_env
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'isaac_sim'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    isaac_ros_manipulation_test_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'test', 'include')

    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Set workflow type for this specific test
    params['workflow_type'] = 'PICK_AND_PLACE'
    params['manual_mode'] = 'true'
    params['headless'] = 'true'

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_test_include_dir,
                 '/perception_foundationpose_connections_test.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
    else:
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'world', '--child-frame-id', 'base_link',
            ]
        ))
    return FoundationPosePickAndPlaceGraphTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay
    )


class FoundationPosePickAndPlaceGraphTest(FoundationPoseTopicConnectionsGraphTest):
    """Test for graph structure of Foundation Pose and RT-DETR integration."""

    def _get_expected_graph(self) -> Dict:
        """
        Define the expected graph structure.

        Returns
        -------
            Dict: A dictionary representation of the expected node connections

        """
        # Create the expected graph with all nodes and connections
        expected_graph = {}

        # Initialize all nodes
        nodes_to_include = [
            '/detection2_d_to_mask',
            '/foundation_pose_server',
            '/foundationpose_node',
            '/manipulator_container',
            '/object_detection_server',
            '/object_info_server',
            '/resize_mask_node',
            '/rtdetr_decoder',
            '/rtdetr_image_to_tensor_node',
            '/rtdetr_interleaved_to_planar_node',
            '/rtdetr_pad_node',
            '/rtdetr_preprocessor',
            '/rtdetr_reshape_node',
            '/rtdetr_resize_node',
            '/rtdetr_tensor_rt',
            '/detection_scaler_up',
            '/detection_scaler_down'
        ]

        for node_name in nodes_to_include:
            expected_graph[node_name] = {
                'publishes': [],
                'subscribes': []
            }

        # Add connections for detection2_d_to_mask
        expected_graph['/detection2_d_to_mask']['publishes'].append({
            'topic': '/rt_detr_segmentation',
            'to': ['/resize_mask_node']
        })

        expected_graph['/detection_scaler_down']['subscribes'].append({
            'topic': '/foundation_pose_server/bbox',
            'from': ['/foundation_pose_server']
        })
        expected_graph['/detection_scaler_down']['publishes'].append({
            'topic': '/filtered_detection2_d',
            'to': ['/detection2_d_to_mask']
        })

        expected_graph['/detection2_d_to_mask']['subscribes'].append({
            'topic': '/filtered_detection2_d',
            'from': ['/detection_scaler_down']
        })

        # Add connections for foundation_pose_server
        expected_graph['/foundation_pose_server']['publishes'].extend([
            {'topic': '/foundation_pose_server/bbox', 'to': ['/detection_scaler_down']},
            {'topic': '/foundation_pose_server/depth',
             'to': ['/foundationpose_node']},
            {'topic': '/foundation_pose_server/image',
             'to': ['/foundationpose_node']},
            {'topic': '/foundation_pose_server/pose_estimation/output', 'to': []},
            {'topic': '/foundation_pose_server/camera_info', 'to': ['/foundationpose_node']}
        ])
        expected_graph['/foundation_pose_server']['subscribes'].extend([
            {'topic': '/front_stereo_camera/depth/ground_truth', 'from': []},
            {'topic': '/front_stereo_camera/left/image_raw', 'from': []},
            {'topic': '/pose_estimation/output', 'from': ['/foundationpose_node']},
            {'topic': '/front_stereo_camera/left/camera_info', 'from': []}
        ])

        # Add connections for foundationpose_node
        expected_graph['/foundationpose_node']['publishes'].extend([
            {'topic': '/pose_estimation/output', 'to': ['/foundation_pose_server']},
            {'topic': '/pose_estimation/pose_matrix_output', 'to': []},
        ])
        expected_graph['/foundationpose_node']['subscribes'].extend([
            {'topic': '/foundation_pose_server/camera_info', 'from': ['/foundation_pose_server']},
            {'topic': '/foundation_pose_server/depth',
             'from': ['/foundation_pose_server']},
            {'topic': '/foundation_pose_server/image',
             'from': ['/foundation_pose_server']},
            {'topic': '/segmentation', 'from': ['/resize_mask_node']},
        ])

        # Add connections for manipulator_container
        expected_graph['/manipulator_container']['subscribes'].append({
            'topic': '/clock',
            'from': []
        })

        # Add connections for object_detection_server
        expected_graph['/object_detection_server']['publishes'].extend([
            {'topic': '/object_detection_server/detections_output', 'to': []},
            {'topic': '/object_detection_server/image_rect', 'to': []}
        ])
        expected_graph['/object_detection_server']['subscribes'].extend([
            {'topic': '/detections', 'from': ['/detection_scaler_up']},
            {'topic': '/front_stereo_camera/left/image_raw', 'from': []}
        ])

        # Add connections for object_info_server (empty)
        # Already initialized with empty lists

        # Add connections for resize_mask_node
        expected_graph['/resize_mask_node']['publishes'].extend([
            {'topic': '/segmentation', 'to': ['/foundationpose_node']},
        ])
        expected_graph['/resize_mask_node']['subscribes'].extend([
            {'topic': '/rt_detr_segmentation', 'from': ['/detection2_d_to_mask']},
        ])

        # Add connections for rtdetr_decoder
        expected_graph['/rtdetr_decoder']['publishes'].append({
            'topic': '/unscaled_detections_rtdetr',
            'to': ['/detection_scaler_up']
        })

        expected_graph['/detection_scaler_up']['publishes'].append({
            'topic': '/detections',
            'to': ['/object_detection_server']
        })

        expected_graph['/rtdetr_decoder']['subscribes'].extend([
            {'topic': '/tensor_sub', 'from': ['/rtdetr_tensor_rt']},
        ])

        # Add connections for rtdetr_image_to_tensor_node
        expected_graph['/rtdetr_image_to_tensor_node']['publishes'].extend([
            {'topic': '/normalized_tensor', 'to': ['/rtdetr_interleaved_to_planar_node']},
        ])
        expected_graph['/rtdetr_image_to_tensor_node']['subscribes'].extend([
            {'topic': '/padded_image', 'from': ['/rtdetr_pad_node']},
        ])

        # Add connections for rtdetr_interleaved_to_planar_node
        expected_graph['/rtdetr_interleaved_to_planar_node']['publishes'].extend([
            {'topic': '/planar_tensor', 'to': ['/rtdetr_reshape_node']},
        ])
        expected_graph['/rtdetr_interleaved_to_planar_node']['subscribes'].extend([
            {'topic': '/normalized_tensor', 'from': ['/rtdetr_image_to_tensor_node']},
        ])

        # Add connections for rtdetr_pad_node
        expected_graph['/rtdetr_pad_node']['publishes'].extend([
            {'topic': '/padded_image', 'to': ['/rtdetr_image_to_tensor_node']},
        ])
        expected_graph['/rtdetr_pad_node']['subscribes'].extend([
            {'topic': '/resize/image', 'from': ['/rtdetr_resize_node']},
        ])

        # Add connections for rtdetr_preprocessor
        expected_graph['/rtdetr_preprocessor']['publishes'].extend([
            {'topic': '/tensor_pub', 'to': ['/rtdetr_tensor_rt']},
        ])
        expected_graph['/rtdetr_preprocessor']['subscribes'].extend([
            {'topic': '/reshaped_tensor', 'from': ['/rtdetr_reshape_node']},
        ])

        # Add connections for rtdetr_reshape_node
        expected_graph['/rtdetr_reshape_node']['publishes'].extend([
            {'topic': '/reshaped_tensor', 'to': ['/rtdetr_preprocessor']},
        ])
        expected_graph['/rtdetr_reshape_node']['subscribes'].extend([
            {'topic': '/planar_tensor', 'from': ['/rtdetr_interleaved_to_planar_node']},
        ])

        # Add connections for rtdetr_resize_node
        expected_graph['/rtdetr_resize_node']['publishes'].extend([
            {'topic': '/resize/image', 'to': ['/rtdetr_pad_node']},
        ])
        expected_graph['/rtdetr_resize_node']['subscribes'].extend([])

        # Add connections for rtdetr_tensor_rt
        expected_graph['/rtdetr_tensor_rt']['publishes'].extend([
            {'topic': '/tensor_sub', 'to': ['/rtdetr_decoder']},
        ])
        expected_graph['/rtdetr_tensor_rt']['subscribes'].extend([
            {'topic': '/tensor_pub', 'from': ['/rtdetr_preprocessor']},
        ])

        # Log the expected graph nodes
        self.node.get_logger().info('Expected graph nodes:')
        for node_name in expected_graph.keys():
            self.node.get_logger().info(f'  - {node_name}')

        return expected_graph
