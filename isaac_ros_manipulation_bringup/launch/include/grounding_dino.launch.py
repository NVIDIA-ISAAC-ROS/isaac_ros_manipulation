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

from isaac_ros_launch_utils.all_types import (
    ComposableNode, DeclareLaunchArgument, GroupAction,
    LaunchConfiguration, LaunchDescription, LoadComposableNodes, OpaqueFunction
)
import isaac_ros_manipulation_ros_python_utils.constants as constants

from launch_ros.actions import Node

NETWORK_BATCH_SIZE = 1
NETWORK_NUM_CHANNELS = 3


def launch_setup(context, *args, **kwargs):
    confidence_threshold = LaunchConfiguration('grounding_dino_confidence_threshold')
    default_prompt = LaunchConfiguration('grounding_dino_default_prompt')
    engine_file_path = LaunchConfiguration('grounding_dino_engine_file_path')
    is_object_following = str(context.perform_substitution(LaunchConfiguration(
        'grounding_dino_is_object_following', default='False')))
    model_file_path = LaunchConfiguration('grounding_dino_model_file_path')

    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')
    depth_image_height = LaunchConfiguration('depth_image_height')
    depth_image_width = LaunchConfiguration('depth_image_width')
    depth_topic_name = LaunchConfiguration('depth_topic_name')
    encoder_image_mean = LaunchConfiguration('grounding_dino_encoder_image_mean')
    encoder_image_stddev = LaunchConfiguration('grounding_dino_encoder_image_stddev')
    image_height = LaunchConfiguration('image_height')
    image_input_topic = LaunchConfiguration('image_input_topic')
    image_width = LaunchConfiguration('image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    network_image_width = LaunchConfiguration('network_image_width')

    dropped_fps = LaunchConfiguration('dropped_fps')
    input_fps = LaunchConfiguration('input_fps')

    input_qos = LaunchConfiguration('input_qos')
    output_qos = LaunchConfiguration('output_qos')

    detections_2d_array_output_topic = LaunchConfiguration(
        'detections_2d_array_output_topic', default='detections_output')

    # This one is used by the foundation pose server to send in detections to generate
    # a mask and route to foundation pose node.
    detection2_d_topic = LaunchConfiguration('detection2_d_topic')

    unscaled_detections_grounding_dino_topic = 'unscaled_detections_grounding_dino'
    filtered_detection2_d_topic = 'filtered_detection2_d'

    if str(context.perform_substitution(model_file_path)) == '':
        raise ValueError('grounding_dino_model_file_path is not set.')
    if str(context.perform_substitution(engine_file_path)) == '':
        raise ValueError('grounding_dino_engine_file_path is not set.')

    dropped_image_topic_name = image_input_topic
    dropped_camera_info_topic_name = camera_info_input_topic
    dropped_depth_topic_name = depth_topic_name

    enforce_max_latency = True
    if context.perform_substitution(LaunchConfiguration('use_sim_time')) == 'true':
        enforce_max_latency = False

    if is_object_following == 'True':
        dropped_image_topic_name = '/grounding_dino/image_dropped'
        dropped_camera_info_topic_name = '/grounding_dino/camera_info_dropped'
        dropped_depth_topic_name = '/grounding_dino/depth_dropped'
        drop_node = ComposableNode(
            name='grounding_dino_drop_node',
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
        )

    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': image_width,
            'input_height': image_height,
            'output_width': network_image_width,
            'output_height': network_image_height,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True,
            'use_latest_camera_info': True,
            'drop_old_messages': False,
            'input_qos': input_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', dropped_image_topic_name),
            ('camera_info', dropped_camera_info_topic_name),
        ],
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': network_image_width,
            'output_image_height': network_image_height,
            'padding_type': 'BOTTOM_RIGHT',
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[(
            'image', 'resize/image'
        )]
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': True,
            'tensor_name': 'image',
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'image_tensor'),
        ]
    )

    normalize_node = ComposableNode(
        name='normalize_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': encoder_image_mean,
            'stddev': encoder_image_stddev,
            'input_tensor_name': 'image',
            'output_tensor_name': 'image',
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('tensor', 'image_tensor'),
        ],
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [
                network_image_height, network_image_width, NETWORK_NUM_CHANNELS],
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'images',
            'input_tensor_shape': [
                NETWORK_NUM_CHANNELS, network_image_height, network_image_width],
            'output_tensor_shape': [
                NETWORK_BATCH_SIZE, NETWORK_NUM_CHANNELS,
                network_image_height, network_image_width],
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )

    text_tokenizer_node = Node(
        package='isaac_ros_grounding_dino',
        executable='isaac_ros_grounding_dino_text_tokenizer.py',
        name='grounding_dino_text_tokenizer',
        output='screen',
    )

    grounding_dino_preprocessor = ComposableNode(
        name='grounding_dino_preprocessor',
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode',
        parameters=[{
            'default_prompt': default_prompt,
            'input_image_tensor_name': 'images',
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image_tensor', 'reshaped_tensor')
        ]
    )

    grounding_dino_inference_node = ComposableNode(
        name='grounding_dino_inference',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'input_tensor_names': [
                'images', 'input_ids', 'attention_mask', 'position_ids',
                'token_type_ids', 'text_token_mask'
            ],
            'input_binding_names': [
                'inputs', 'input_ids', 'attention_mask', 'position_ids',
                'token_type_ids', 'text_token_mask'
            ],
            'output_tensor_names': ['scores', 'boxes'],
            'output_binding_names': ['pred_logits', 'pred_boxes'],
            'verbose': False,
            'force_engine_update': False,
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
    )

    grounding_dino_decoder_node = ComposableNode(
        name='grounding_dino_decoder',
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'image_width': network_image_width,
            'image_height': network_image_height,
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('detections_output', unscaled_detections_grounding_dino_topic),
        ],
    )

    scale_up_node = Node(
        package='isaac_ros_rtdetr',
        executable='detection_scaler.py',
        name='detection_scaler_up',
        parameters=[{
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'rgb_image_width': image_width,
            'rgb_image_height': image_height,
            'scale_type': 'scale_up',
            'sub_pub_message_type': 'multi'
        }],
        remappings=[
            ('input', unscaled_detections_grounding_dino_topic),
            # This is injested by object detection server node after it has been scaled up.
            ('output', detections_2d_array_output_topic),
        ]
    )

    if is_object_following == 'True':
        # When object following, Grounding DINO only accepts one category per prompt.
        # The filter node is required to output a detection 2d instead of a detection 2d array
        detection2_d_array_filter_node = ComposableNode(
            name='detection2_d_array_filter',
            package='isaac_ros_foundationpose',
            plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
            parameters=[{
                'desired_class_id': '0',  # The first index of the Grounding DINO prompt
                'input_qos': output_qos,
                'output_qos': output_qos,
                'input_qos_depth': 1,
                'output_qos_depth': 1,
            }],
            remappings=[
                ('detection2_d_array', unscaled_detections_grounding_dino_topic),
                ('detection2_d', filtered_detection2_d_topic),
            ]
        )
    else:
        # This node takes as input foundation pose server output and scales it down to the
        # Grounding DINO network input size to enable mask generation.
        scale_down_node = Node(
            package='isaac_ros_rtdetr',
            executable='detection_scaler.py',
            name='detection_scaler_down',
            parameters=[{
                'network_image_height': network_image_height,
                'network_image_width': network_image_width,
                'rgb_image_width': image_width,
                'rgb_image_height': image_height,
                'scale_type': 'scale_down',
                'sub_pub_message_type': 'single'
            }],
            remappings=[
                ('input', detection2_d_topic),
                ('output', filtered_detection2_d_topic),
            ]
        )

    image_width_int = int(context.perform_substitution(image_width))
    image_height_int = int(context.perform_substitution(image_height))
    network_image_width_int = int(context.perform_substitution(network_image_width))
    network_image_height_int = int(context.perform_substitution(network_image_height))
    width_scale = image_width_int / network_image_width_int
    height_scale = image_height_int / network_image_height_int
    rgb_image_to_network_ratio = max(width_scale, height_scale)
    mask_width = int(image_width_int / rgb_image_to_network_ratio)
    mask_height = int(image_height_int / rgb_image_to_network_ratio)

    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': mask_width,
            'mask_height': mask_height,
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[('detection2_d', filtered_detection2_d_topic),
                    ('segmentation', 'grounding_dino_segmentation')]
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
            'use_latest_camera_info': True,
            'drop_old_messages': False,
            'input_qos': output_qos,
            'output_qos': output_qos,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'grounding_dino_segmentation'),
            ('camera_info', 'resize/camera_info'),
            ('resize/image', 'segmentation'),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    composable_nodes = [
        resize_node,
        pad_node,
        image_to_tensor_node,
        normalize_node,
        interleave_to_planar_node,
        reshape_node,
        grounding_dino_preprocessor,
        grounding_dino_inference_node,
        grounding_dino_decoder_node,
        detection2_d_to_mask_node,
        resize_mask_node,
    ]

    non_composable_nodes = [
        scale_up_node,
        text_tokenizer_node,
    ]

    if is_object_following == 'True':
        composable_nodes.extend([
            detection2_d_array_filter_node,
            drop_node,
        ])
    else:
        non_composable_nodes.append(scale_down_node)

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_nodes)

    final_launch = GroupAction(actions=[load_composable_nodes])

    return [final_launch] + non_composable_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'dropped_fps',
            default_value='28',
            description='FPS that are dropped by the drop node'),
        DeclareLaunchArgument(
            'grounding_dino_model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'grounding_dino_encoder_image_mean',
            default_value='[0.485, 0.456, 0.406]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'grounding_dino_encoder_image_stddev',
            default_value='[0.229, 0.224, 0.225]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'grounding_dino_engine_file_path',
            default_value='',
            description='The absolute path to the Grounding DINO engine plan.'),
        DeclareLaunchArgument(
            'grounding_dino_confidence_threshold',
            default_value='0.5',
            description='The minimum score for a bounding box to be published.'),
        DeclareLaunchArgument(
            'grounding_dino_default_prompt',
            default_value='object',
            description='The text prompt for Grounding DINO object detection.'),
        DeclareLaunchArgument(
            'image_width',
            default_value='1920',
            description='The input image width'),
        DeclareLaunchArgument(
            'image_height',
            default_value='1080',
            description='The input image height'),
        DeclareLaunchArgument(
            'input_fps',
            default_value='30',
            description='FPS for input message to the drop node'),
        DeclareLaunchArgument(
            'input_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for Grounding DINO input'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='960',
            description='The network image width'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='544',
            description='The network image height'),
        DeclareLaunchArgument(
            'output_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for Grounding DINO output'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
