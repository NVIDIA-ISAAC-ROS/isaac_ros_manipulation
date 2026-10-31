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
    ComposableNode, DeclareLaunchArgument, GroupAction, LaunchConfiguration,
    LaunchDescription, LoadComposableNodes, OpaqueFunction, Shutdown
)
import isaac_ros_manipulation_ros_python_utils.constants as constants

from launch_ros.actions import Node


SAM_MODEL_INPUT_WIDTH = 1024  # SAM model accept 1024x1024 image
SAM_MODEL_INPUT_HEIGHT = 1024


def launch_setup(context, *args, **kwargs):
    input_qos = LaunchConfiguration('input_qos')
    output_qos = LaunchConfiguration('output_qos')

    image_input_topic = LaunchConfiguration('image_input_topic')
    depth_topic_name = LaunchConfiguration('depth_topic_name')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')

    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    depth_image_width = LaunchConfiguration('depth_image_width')
    depth_image_height = LaunchConfiguration('depth_image_height')

    segment_anything_input_points_topic = LaunchConfiguration(
        'segment_anything_input_points_topic')
    segment_anything_input_detections_topic = LaunchConfiguration(
        'segment_anything_input_detections_topic')

    segment_anything_output_detections_topic = LaunchConfiguration(
        'segment_anything_output_detections_topic')
    segment_anything_output_binary_mask_topic = LaunchConfiguration(
        'segment_anything_output_binary_mask_topic')

    image_image_width_int = int(context.perform_substitution(image_width))
    image_image_height_int = int(context.perform_substitution(image_height))

    is_point_triggered = str(context.perform_substitution(LaunchConfiguration(
        'segment_anything_is_point_triggered', default='False')))

    enable_debug_output = str(context.perform_substitution(LaunchConfiguration(
        'segment_anything_enable_debug_output', default='False')))

    # Triton parameters
    sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
    sam_max_batch_size = LaunchConfiguration('sam_max_batch_size')

    # DNN Image Encoder parameters
    sam_encoder_image_mean = LaunchConfiguration('sam_encoder_image_mean')
    sam_encoder_image_stddev = LaunchConfiguration('sam_encoder_image_stddev')

    dropped_image_topic_name = image_input_topic
    dropped_camera_info_topic_name = camera_info_input_topic
    dropped_depth_topic_name = depth_topic_name

    composable_nodes = []

    if is_point_triggered == 'True':
        dropped_image_topic_name = '/segment_anything/image_dropped'
        dropped_camera_info_topic_name = '/segment_anything/camera_info_dropped'
        dropped_depth_topic_name = '/segment_anything/depth_dropped'
        composable_nodes.append(ComposableNode(
            name='point_triggered_node',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingPointTriggeredNode',
            namespace='segment_anything',
            parameters=[{
                'mode': 'mono+depth',
                'is_sam2': False,
                'max_rate_hz': 2.0,
                'input_qos': input_qos,
                'output_qos': output_qos,
                'input_queue_size': 1,
                'output_queue_size': 1,
                'sync_queue_size': 5,
            }],
            remappings=[
                ('image_1', image_input_topic),
                ('camera_info_1', camera_info_input_topic),
                ('image_1_triggered', dropped_image_topic_name),
                ('camera_info_1_triggered', dropped_camera_info_topic_name),
                ('depth_1', depth_topic_name),
                ('depth_1_triggered', dropped_depth_topic_name),
                ('point', segment_anything_input_points_topic),
                ('detection_2d', segment_anything_input_detections_topic)
            ]
        ))

    composable_nodes.append(
        ComposableNode(
            name='input_resize_node',
            namespace='segment_anything',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            parameters=[{
                'input_width': image_width,
                'input_height': image_height,
                'output_width': SAM_MODEL_INPUT_WIDTH,
                'output_height': SAM_MODEL_INPUT_HEIGHT,
                'keep_aspect_ratio': True,
                'disable_padding': True,
                'encoding_desired': 'rgb8',
                'input_qos': output_qos if is_point_triggered == 'True' else input_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('image', dropped_image_topic_name),
                ('camera_info', dropped_camera_info_topic_name),
                ('resize/image', 'resized_image'),
                ('resize/camera_info', 'resized_camera_info'),
            ],
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='pad_node',
            namespace='segment_anything',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::PadNode',
            parameters=[{
                'output_image_width': SAM_MODEL_INPUT_WIDTH,
                'output_image_height': SAM_MODEL_INPUT_HEIGHT,
                'padding_type': 'BOTTOM_RIGHT',
                'input_qos': output_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('image', 'resized_image'),
            ],
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='image_to_tensor_node',
            namespace='segment_anything',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
            parameters=[{
                'scale': True,
                'tensor_name': 'image',
                'input_qos': output_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('image', 'padded_image'),
                ('tensor', 'image_tensor'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='normalize_node',
            namespace='segment_anything',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
            parameters=[{
                'mean': sam_encoder_image_mean,
                'stddev': sam_encoder_image_stddev,
                'input_tensor_name': 'image',
                'output_tensor_name': 'image',
                'input_qos': output_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('tensor', 'image_tensor'),
                ('normalized_tensor', 'normalized_tensor'),
            ],
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='interleaved_to_planar_node',
            namespace='segment_anything',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
            parameters=[{
                'input_tensor_shape': [SAM_MODEL_INPUT_HEIGHT, SAM_MODEL_INPUT_WIDTH, 3],
                'input_qos': output_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('interleaved_tensor', 'normalized_tensor'),
                ('planar_tensor', 'planar_tensor'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='reshape_node',
            namespace='segment_anything',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
            parameters=[{
                'output_tensor_name': 'input_tensor',
                'input_tensor_shape': [3, SAM_MODEL_INPUT_HEIGHT, SAM_MODEL_INPUT_WIDTH],
                'output_tensor_shape': [1, 3, SAM_MODEL_INPUT_HEIGHT, SAM_MODEL_INPUT_WIDTH],
                'input_qos': output_qos,
                'output_qos': output_qos
            }],
            remappings=[
                ('tensor', 'planar_tensor'),
                ('reshaped_tensor', 'tensor_pub'),
            ],
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='dummy_mask_pub_node',
            namespace='segment_anything',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::DummyMaskPublisher',
            remappings=[('tensor_pub', 'tensor_pub'),
                        ('mask', 'mask')]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='data_encoder_node',
            namespace='segment_anything',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode',
            parameters=[{
                'prompt_input_type': 'point',  # You can use a bbox as well but no support yet.
                'has_input_mask': False,  # Whether input mask is valid or not.
                'max_batch_size': sam_max_batch_size,
                'orig_img_dims': [image_image_height_int, image_image_width_int],
            }],
            remappings=[
                ('prompts', segment_anything_input_detections_topic),
                ('mask', 'mask'),
                ('tensor', 'encoded_data'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='triton_node',
            namespace='segment_anything',
            package='isaac_ros_triton',
            plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
            parameters=[{
                'model_name': 'segment_anything',
                'model_repository_paths': sam_model_repository_paths,
                'max_batch_size': 1,
                'input_tensor_names': ['input_tensor', 'points', 'labels',
                                       'input_mask', 'has_input_mask', 'orig_img_dims'],
                'input_binding_names': ['images', 'point_coords', 'point_labels',
                                        'mask_input', 'has_mask_input', 'orig_im_size'],
                'output_tensor_names': ['masks', 'iou', 'low_res_mask'],
                'output_binding_names': ['masks', 'iou_predictions', 'low_res_masks'],
            }],
            remappings=[
                ('tensor_pub', 'encoded_data'),
                ('tensor_sub', 'tensor_sub'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='decoder_node',
            namespace='segment_anything',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
            parameters=[{
                'mask_width': image_width,
                'mask_height': image_height,
                'max_batch_size': sam_max_batch_size,
            }],
            remappings=[
                ('segment_anything/raw_segmentation_mask', 'raw_segmentation_mask'),
            ],
        )
    )

    # If segment anything, also start rqt viewer and the point prompt generator
    non_composable_nodes = []

    image_view_topic = image_input_topic
    if enable_debug_output == 'True':
        # Convert the binary mask to a colored mask and overlay it on the original image for debug
        image_view_topic = '/segment_anything/colored_segmentation_mask'
        non_composable_nodes.append(
            Node(
                name='colored_mask_converter',
                namespace='segment_anything',
                package='isaac_ros_segment_anything',
                executable='colored_mask_converter_node.py',
                remappings=[
                    ('image', image_input_topic),
                    ('binary_segmentation_mask', segment_anything_output_binary_mask_topic),
                    ('colored_segmentation_mask', 'colored_segmentation_mask'),
                ],
                on_exit=Shutdown(),
            )
        )

    headless = str(context.perform_substitution(
        LaunchConfiguration('headless'))).lower() == 'true'
    if not headless:
        non_composable_nodes.append(
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='segment_anything_rqt_viewer',
                arguments=[image_view_topic],
                on_exit=Shutdown(),
            )
        )

    tensor_to_image_node = ComposableNode(
        name='tensor_to_image',
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::TensorToImageNode',
        namespace='segment_anything',
        remappings=[
            ('segmentation_tensor', 'raw_segmentation_mask'),
            ('detection_array', segment_anything_output_detections_topic),
            ('binary_mask', 'original_binary_mask'),
        ]
    )

    sam_resize_mask_node = ComposableNode(
        name='sam_resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace='segment_anything',
        parameters=[{
            'input_width': image_width,
            'input_height': image_height,
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
            ('image', 'original_binary_mask'),
            ('camera_info', dropped_camera_info_topic_name),
            ('resize/image', segment_anything_output_binary_mask_topic),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    composable_nodes.append(tensor_to_image_node)
    composable_nodes.append(sam_resize_mask_node)

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_nodes)

    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],)

    return [final_launch] + non_composable_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'image_input_topic',
            default_value='image',
            description='The input topic name for the image'
        ),
        DeclareLaunchArgument(
            'depth_topic_name',
            default_value='depth',
            description='The input topic name for the depth'
        ),
        DeclareLaunchArgument(
            'camera_info_input_topic',
            default_value='camera_info',
            description='The input topic name for the camera info'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='The width of the image'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='640',
            description='The height of the image'
        ),
        DeclareLaunchArgument(
            'depth_image_width',
            default_value='640',
            description='The width of the depth image'
        ),
        DeclareLaunchArgument(
            'depth_image_height',
            default_value='640',
            description='The height of the depth image'
        ),
        DeclareLaunchArgument(
            'sam_encoder_image_mean',
            default_value='[0.485, 0.456, 0.406]',
            description='The mean for image normalization'
        ),
        DeclareLaunchArgument(
            'sam_encoder_image_stddev',
            default_value='[0.229, 0.224, 0.225]',
            description='The standard deviation for image normalization'
        ),
        DeclareLaunchArgument(
            'sam_model_repository_paths',
            default_value='["/tmp/models"]',
            description='The absolute path to the repository of models'
        ),
        DeclareLaunchArgument(
            'sam_max_batch_size',
            default_value='20',
            description='The maximum allowed batch size of the model'
        ),
        DeclareLaunchArgument(
            'segment_anything_is_point_triggered',
            default_value='False',
            description='Whether object following is enabled or not'
        ),
        DeclareLaunchArgument(
            'segment_anything_enable_debug_output',
            default_value='False',
            description='Enables visualizing of segmentation mask for debugging'
        ),
        DeclareLaunchArgument(
            'input_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for RT-DETR input'),
        DeclareLaunchArgument(
            'output_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for RT-DETR output'),
        DeclareLaunchArgument(
            'segment_anything_input_points_topic',
            default_value='input_points',
            description='The input topic name for the points'
        ),
        DeclareLaunchArgument(
            'segment_anything_input_detections_topic',
            default_value='input_detections',
            description='The input topic name for the detections'
        ),
        DeclareLaunchArgument(
            'segment_anything_output_binary_mask_topic',
            default_value='binary_segmentation_mask',
            description='The output topic name for the binary mask'
        ),
        DeclareLaunchArgument(
            'segment_anything_output_detections_topic',
            default_value='output_detections',
            description='The output topic name for the detections'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run in headless mode without GUI elements like rqt_image_view'
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
