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

from isaac_ros_launch_utils.all_types import (
    ComposableNode, DeclareLaunchArgument, GroupAction, LaunchConfiguration,
    LaunchDescription, LoadComposableNodes, OpaqueFunction
)
import isaac_ros_manipulation_ros_python_utils.constants as constants

from launch_ros.actions import Node

RT_DETR_MODEL_INPUT_WIDTH = 640
RT_DETR_MODEL_INPUT_HEIGHT = 640


def launch_setup(context, *args, **kwargs):
    engine_file_path = LaunchConfiguration('rtdetr_engine_file_path')
    input_qos = LaunchConfiguration('input_qos')
    output_qos = LaunchConfiguration('output_qos')
    detections_2d_array_output_topic = LaunchConfiguration(
        'detections_2d_array_output_topic', default='detections_output')

    image_input_topic = LaunchConfiguration('image_input_topic')
    depth_topic_name = LaunchConfiguration('depth_topic_name')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')

    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    depth_image_width = LaunchConfiguration('depth_image_width')
    depth_image_height = LaunchConfiguration('depth_image_height')

    input_fps = LaunchConfiguration('input_fps')
    dropped_fps = LaunchConfiguration('dropped_fps')

    rt_detr_confidence_threshold = LaunchConfiguration('rt_detr_confidence_threshold')
    foundationpose_server_input_camera_info_topic = LaunchConfiguration(
        'foundationpose_server_input_camera_info_topic')

    rtdetr_is_object_following = str(context.perform_substitution(LaunchConfiguration(
        'rtdetr_is_object_following', default='False')))

    object_class_id = LaunchConfiguration('object_class_id')
    # This one is used by the foundation pose server to send in detections to generate
    # a mask and route to foundation pose node.
    detection2_d_topic = LaunchConfiguration('detection2_d_topic')

    drop_node_nodes = []

    dropped_image_topic_name = image_input_topic
    dropped_camera_info_topic_name = camera_info_input_topic
    dropped_depth_topic_name = depth_topic_name

    unscaled_detections_rtdetr_topic = 'unscaled_detections_rtdetr'
    filtered_detection2_d_topic = 'filtered_detection2_d'

    enforce_max_latency = True
    if context.perform_substitution(LaunchConfiguration('use_sim_time')) == 'true':
        enforce_max_latency = False

    if rtdetr_is_object_following == 'True':
        dropped_image_topic_name = '/rtdetr/image_dropped'
        dropped_camera_info_topic_name = '/rtdetr/camera_info_dropped'
        dropped_depth_topic_name = '/rtdetr/depth_dropped'
        drop_node_nodes.append(ComposableNode(
            name='rtdetr_drop_node',
            package='isaac_ros_topic_tools',
            plugin='nvidia::isaac_ros::topic_tools::CameraDropNode',
            parameters=[{
                'X': dropped_fps,
                'Y': input_fps,
                'mode': 'mono+depth',
                'input_qos': input_qos,
                'output_qos': output_qos,
                'input_queue_size': 1,
                'output_queue_size': 1,
                'sync_queue_size': 5,
                'max_latency_threshold': 0.1,
                'enforce_max_latency': enforce_max_latency,
            }],
            remappings=[
                ('image_1', image_input_topic),
                ('camera_info_1', camera_info_input_topic),
                ('image_1_drop', dropped_image_topic_name),
                ('camera_info_1_drop', dropped_camera_info_topic_name),
                ('depth_1', depth_topic_name),
                ('depth_1_drop', dropped_depth_topic_name)
            ]
        ))

    resize_node = ComposableNode(
        name='rtdetr_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_width': image_width,
            'input_height': image_height,
            'output_width': RT_DETR_MODEL_INPUT_WIDTH,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True,
            'use_latest_camera_info': True,
            'drop_old_messages': False
        }],
        remappings=[
            ('image', dropped_image_topic_name),
            ('camera_info', dropped_camera_info_topic_name),
        ],
    )

    pad_node = ComposableNode(
        name='rtdetr_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RT_DETR_MODEL_INPUT_WIDTH,
            'output_image_height': RT_DETR_MODEL_INPUT_HEIGHT,
            'padding_type': 'BOTTOM_RIGHT',
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[('image', 'resize/image')])

    image_to_tensor_node = ComposableNode(
        name='rtdetr_image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'normalized_tensor'),
        ])

    interleave_to_planar_node = ComposableNode(
        name='rtdetr_interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [RT_DETR_MODEL_INPUT_HEIGHT, RT_DETR_MODEL_INPUT_WIDTH, 3],
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')])

    reshape_node = ComposableNode(
        name='rtdetr_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, RT_DETR_MODEL_INPUT_HEIGHT, RT_DETR_MODEL_INPUT_WIDTH],
            'output_tensor_shape': [1, 3, RT_DETR_MODEL_INPUT_HEIGHT, RT_DETR_MODEL_INPUT_WIDTH],
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[('encoded_tensor', 'reshaped_tensor')])

    tensor_rt_node = ComposableNode(
        name='rtdetr_tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'verbose': False,
            'force_engine_update': False,
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{
            'confidence_threshold': rt_detr_confidence_threshold,
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[
            ('detections_output', unscaled_detections_rtdetr_topic),
        ],
    )

    # Rt-detr outputs detections as per image size of 640x640, so we need to scale them to the
    # actual image size
    image_width_int = int(context.perform_substitution(image_width))
    image_height_int = int(context.perform_substitution(image_height))

    non_composable_nodes = [
        Node(
            package='isaac_ros_rtdetr',
            executable='detection_scaler.py',
            name='detection_scaler_up',
            parameters=[{
                'network_image_width': RT_DETR_MODEL_INPUT_WIDTH,
                'network_image_height': RT_DETR_MODEL_INPUT_HEIGHT,
                'rgb_image_width': image_width_int,
                'rgb_image_height': image_height_int,
                'scale_type': 'scale_up',
                'sub_pub_message_type': 'multi'
            }],
            remappings=[
                ('input', unscaled_detections_rtdetr_topic),
                # This is injested by the object detection server node after it has been scaled up.
                ('output', detections_2d_array_output_topic),
            ]
        )
    ]

    mask_resize_camera_info_topic = 'resize/camera_info'

    if rtdetr_is_object_following == 'True':
        # This is required to filter out other object ids as well as output a detection 2d
        # instead of a detection 2d array, which is required by the Mask generation node
        detection2_d_array_filter_node = ComposableNode(
            name='detection2_d_array_filter',
            package='isaac_ros_foundationpose',
            plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
            parameters=[{
                'desired_class_id': str(context.perform_substitution(object_class_id))}
            ],
            remappings=[('detection2_d_array', unscaled_detections_rtdetr_topic),  # subscriber
                        ('detection2_d', filtered_detection2_d_topic)]  # publisher
        )
    else:
        # This node takes as input foundation pose server output and scales it down to the
        # RT-DETR input size to enable mask generation.
        scale_down_node = Node(
            package='isaac_ros_rtdetr',
            executable='detection_scaler.py',
            name='detection_scaler_down',
            parameters=[{
                'network_image_width': RT_DETR_MODEL_INPUT_WIDTH,
                'network_image_height': RT_DETR_MODEL_INPUT_HEIGHT,
                'rgb_image_width': image_width_int,
                'rgb_image_height': image_height_int,
                'scale_type': 'scale_down',
                'sub_pub_message_type': 'single'
            }],
            remappings=[
                ('input', detection2_d_topic),
                ('output', filtered_detection2_d_topic),
            ]
        )
        non_composable_nodes.append(scale_down_node)
        mask_resize_camera_info_topic = 'reshaped_camera_info_topic'

        camera_info_scaler_node = Node(
            package='isaac_ros_rtdetr',
            executable='camera_info_scaler.py',
            name='camera_info_scaler',
            parameters=[{
                'scale_factor': 0.5
            }],
            remappings=[('input', foundationpose_server_input_camera_info_topic),
                        ('output', mask_resize_camera_info_topic)]
        )
        non_composable_nodes.append(camera_info_scaler_node)

    rt_detr_model_input_width = RT_DETR_MODEL_INPUT_WIDTH
    rgb_image_to_rt_detr_ratio = image_width_int / rt_detr_model_input_width
    mask_width = int(image_width_int/rgb_image_to_rt_detr_ratio)
    mask_height = int(image_height_int/rgb_image_to_rt_detr_ratio)

    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': mask_width,
            'mask_height': mask_height,
            'input_qos': output_qos,
            'output_qos': output_qos
        }],
        remappings=[('detection2_d', filtered_detection2_d_topic),
                    ('segmentation', 'rt_detr_segmentation')]
    )

    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': mask_width,
            'input_height': mask_height,
            'output_width': depth_image_width,
            'output_height': depth_image_height,
            'keep_aspect_ratio': False,
            'disable_padding': False,
            'input_qos': output_qos,
            'output_qos': output_qos,
            'use_latest_camera_info': True,
            'drop_old_messages': False
        }],
        remappings=[
            ('image', 'rt_detr_segmentation'),
            ('camera_info', mask_resize_camera_info_topic),
            ('resize/image', 'segmentation'),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    final_nodes = [
        resize_node,
        pad_node,
        image_to_tensor_node,
        interleave_to_planar_node,
        reshape_node,
        rtdetr_preprocessor_node,
        tensor_rt_node,
        rtdetr_decoder_node,
        detection2_d_to_mask_node,
        resize_mask_node,
    ]

    if rtdetr_is_object_following == 'True':
        final_nodes.append(detection2_d_array_filter_node)

    final_nodes += drop_node_nodes
    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=final_nodes)

    final_launch = GroupAction(
        actions=[
            load_composable_nodes,
        ],)

    return [final_launch] + non_composable_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'rtdetr_engine_file_path',
            default_value='/tmp/rtdetr.plan',
            description='The absolute path to the RTDETR engine plan.',
        ),
        DeclareLaunchArgument(
            'rt_detr_confidence_threshold',
            default_value='0.7',
            description='The minimum score for a bounding box to be published.',
        ),
        DeclareLaunchArgument(
            'input_fps',
            default_value='30',
            description='FPS for input message to the drop node'
        ),
        DeclareLaunchArgument(
            'object_class_id',
            default_value='22',
            description='The RT-DETR class ID of the object'
        ),
        DeclareLaunchArgument(
            'dropped_fps',
            default_value='28',
            description='FPS that are dropped by the drop node'),
        DeclareLaunchArgument(
            'input_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for RT-DETR input'),
        DeclareLaunchArgument(
            'output_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for RT-DETR output'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
