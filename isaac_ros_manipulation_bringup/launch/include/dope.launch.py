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

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_launch_utils.all_types import (
    ComposableNode, DeclareLaunchArgument, GroupAction,
    IncludeLaunchDescription, LaunchConfiguration, LaunchDescription,
    LoadComposableNodes, PythonLaunchDescriptionSource
)
import isaac_ros_manipulation_ros_python_utils.constants as constants


def generate_launch_description():
    """Generate launch description for DOPE encoder->TensorRT->DOPE decoder."""
    launch_args = [
        DeclareLaunchArgument(
            'dope_model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'dope_engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'dope_max_workspace_size',
            default_value='536870912',
            description='The TensorRT builder workspace size in bytes'),
        DeclareLaunchArgument(
            'object_name',
            default_value='soup',
            description='The object class that the DOPE network is detecting'),
        DeclareLaunchArgument(
            'dope_map_peak_threshold',
            default_value='0.1',
            description='The minimum value of a peak in a DOPE belief map'),
        DeclareLaunchArgument(
            'stability_num_samples',
            default_value='2',
            description='Number of samples to ensure pose stability'),
        DeclareLaunchArgument(
            'stability_distance_threshold',
            default_value='0.1',
            description='Maximum distance in meters between poses for stability'),
        DeclareLaunchArgument(
            'stability_angle_threshold',
            default_value='5.0',
            description='Maximum angle in degrees between poses for stability'),
        DeclareLaunchArgument(
            'dope_enable_tf_publishing',
            default_value='false',
            description='Enable DOPE publishing pose to TF'),
        DeclareLaunchArgument(
            'rotation_y_axis',
            default_value='0.0',
            description='Enable DOPE pose estimation to be rotated by X degrees along y axis'),
        DeclareLaunchArgument(
            'rotation_x_axis',
            default_value='0.0',
            description='Enable DOPE pose estimation to be rotated by X degrees along x axis'),
        DeclareLaunchArgument(
            'rotation_z_axis',
            default_value='0.0',
            description='Enable DOPE pose estimation to be rotated by X degrees along z axis'),
        DeclareLaunchArgument(
            'input_fps',
            default_value='30',
            description='FPS for input message to the drop node'),
        DeclareLaunchArgument(
            'dropped_fps',
            default_value='28',
            description='FPS that are dropped by the drop node'),
        DeclareLaunchArgument(
            'dope_input_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for DOPE input'),
    ]

    image_input_topic = LaunchConfiguration('image_input_topic')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width', default='1920')
    input_image_height = LaunchConfiguration('input_image_height', default='1080')
    dope_network_image_width = LaunchConfiguration('dope_network_image_width', default='640')
    dope_network_image_height = LaunchConfiguration('dope_network_image_height', default='480')
    encoder_image_mean = LaunchConfiguration('encoder_image_mean', default='[0.485, 0.456, 0.406]')
    encoder_image_stddev = LaunchConfiguration(
        'encoder_image_stddev', default='[0.229, 0.224, 0.225]')

    # Tensor RT parameters
    dope_model_file_path = LaunchConfiguration('dope_model_file_path')
    dope_engine_file_path = LaunchConfiguration('dope_engine_file_path')
    dope_max_workspace_size = LaunchConfiguration('dope_max_workspace_size')
    input_tensor_names = LaunchConfiguration('input_tensor_names', default='["input_tensor"]')
    input_binding_names = LaunchConfiguration('input_binding_names', default='["input"]')
    output_tensor_names = LaunchConfiguration('output_tensor_names', default='["output"]')
    output_binding_names = LaunchConfiguration('output_binding_names', default='["output"]')
    tensorrt_verbose = LaunchConfiguration('tensorrt_verbose', default='False')
    force_engine_update = LaunchConfiguration('force_engine_update', default='False')

    # DOPE Decoder parameters
    object_name = LaunchConfiguration('object_name')
    dope_map_peak_threshold = LaunchConfiguration('dope_map_peak_threshold')
    dope_enable_tf_publishing = LaunchConfiguration('dope_enable_tf_publishing')
    rotation_y_axis = LaunchConfiguration('rotation_y_axis')
    rotation_x_axis = LaunchConfiguration('rotation_x_axis')
    rotation_z_axis = LaunchConfiguration('rotation_z_axis')

    # DOPE pose filtering parameters
    stability_num_samples = LaunchConfiguration('stability_num_samples')
    stability_distance_threshold = LaunchConfiguration('stability_distance_threshold')
    stability_angle_threshold = LaunchConfiguration('stability_angle_threshold')

    # Drop node parameters
    input_fps = LaunchConfiguration('input_fps')
    dropped_fps = LaunchConfiguration('dropped_fps')
    dope_input_qos = LaunchConfiguration('dope_input_qos')

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    dope_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': dope_network_image_width,
            'network_image_height': dope_network_image_height,
            'input_qos': dope_input_qos,
            'output_qos': dope_input_qos,
            'image_mean': encoder_image_mean,
            'image_stddev': encoder_image_stddev,
            'attach_to_shared_component_container': 'True',
            'component_container_name': constants.MANIPULATOR_CONTAINER_NAME,
            'dnn_image_encoder_namespace': 'dope_encoder',
            'image_input_topic': '/dope_encoder/image_dropped',
            'camera_info_input_topic': '/dope_encoder/camera_info_dropped',
            'tensor_output_topic': '/tensor_pub',
            'tensor_name': 'input_tensor',
            'keep_aspect_ratio': 'False'
        }.items(),
    )

    drop_node = ComposableNode(
        name='dope_drop_node',
        package='isaac_ros_topic_tools',
        plugin='nvidia::isaac_ros::topic_tools::CameraDropNode',
        parameters=[{
            'X': dropped_fps,
            'Y': input_fps,
            'mode': 'mono',
            'input_qos': dope_input_qos,
            'output_qos': dope_input_qos,
            'input_queue_size': 1,
            'output_queue_size': 1,
            'sync_queue_size': 5,
            'max_latency_threshold': 0.1,
            'enforce_max_latency': True,
        }],
        remappings=[
            ('image_1', image_input_topic),
            ('camera_info_1', camera_info_input_topic),
            ('image_1_drop', '/dope_encoder/image_dropped'),
            ('camera_info_1_drop', '/dope_encoder/camera_info_dropped'),
        ])

    dope_inference_node = ComposableNode(
        name='dope_inference',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': dope_model_file_path,
            'engine_file_path': dope_engine_file_path,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'output_tensor_names': output_tensor_names,
            'output_binding_names': output_binding_names,
            'verbose': tensorrt_verbose,
            'input_qos': dope_input_qos,
            'output_qos': dope_input_qos,
            'force_engine_update': force_engine_update,
            'max_workspace_size': dope_max_workspace_size,
        }])

    dope_decoder_node = ComposableNode(
        name='dope_decoder',
        package='isaac_ros_dope',
        plugin='nvidia::isaac_ros::dope::DopeDecoderNode',
        parameters=[{
            'object_name': object_name,
            'enable_tf_publishing': dope_enable_tf_publishing,
            'map_peak_threshold': dope_map_peak_threshold,
            'tf_frame_name': 'detected_object',
            'rotation_y_axis': rotation_y_axis,
            'rotation_x_axis': rotation_x_axis,
            'rotation_z_axis': rotation_z_axis,
        }],
        remappings=[('belief_map_array', 'tensor_sub'),
                    ('dope/detections', 'detections'),
                    ('camera_info', '/dope_encoder/crop/camera_info')]
    )

    detection3_d_array_to_pose_node = ComposableNode(
        name='detection3_d_array_to_pose_node',
        package='isaac_ros_pose_proc',
        plugin='nvidia::isaac_ros::pose_proc::Detection3DArrayToPoseNode',
        parameters=[{
            'desired_class_id': object_name,
        }],
        remappings=[('detection3_d_array_input', 'detections'),
                    ('pose_output', 'selected_pose')]
    )

    stability_filter_node = ComposableNode(
        name='stability_filter',
        package='isaac_ros_pose_proc',
        plugin='nvidia::isaac_ros::pose_proc::StabilityFilterNode',
        parameters=[{
            'enable_tf_publishing': True,
            'child_frame_id': 'detected_object1',
            'num_samples': stability_num_samples,
            'distance_threshold': stability_distance_threshold,
            'quat_prod_threshold': stability_angle_threshold
        }],
        remappings=[('pose_input', 'selected_pose'),
                    ('pose_output', 'stable_pose')]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[
            dope_decoder_node,
            dope_inference_node,
            drop_node,
            detection3_d_array_to_pose_node,
            stability_filter_node,
        ])

    final_launch = GroupAction(
        actions=[
            load_composable_nodes,
        ],)

    return LaunchDescription(launch_args + [final_launch, dope_encoder_launch])
