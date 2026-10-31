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
    LaunchDescription, LoadComposableNodes, OpaqueFunction
)
import isaac_ros_manipulation_ros_python_utils.constants as constants

from launch_ros.actions import Node


SAM2_MODEL_INPUT_WIDTH = 1024  # SAM2 model accept 1024x1024 image
SAM2_MODEL_INPUT_HEIGHT = 1024


def launch_setup(context, *args, **kwargs):
    input_qos = LaunchConfiguration('input_qos')
    output_qos = LaunchConfiguration('output_qos')

    image_input_topic = LaunchConfiguration('image_input_topic')
    depth_topic_name = LaunchConfiguration('depth_topic_name')
    camera_info_input_topic = LaunchConfiguration('camera_info_input_topic')

    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')

    segment_anything_input_points_topic = LaunchConfiguration(
        'segment_anything_input_points_topic')
    segment_anything2_output_detections_topic = LaunchConfiguration(
        'segment_anything2_output_detections_topic')
    segment_anything2_output_binary_mask_topic = LaunchConfiguration(
        'segment_anything2_output_binary_mask_topic')

    image_image_width_int = int(context.perform_substitution(image_width))
    image_image_height_int = int(context.perform_substitution(image_height))

    is_point_triggered = str(context.perform_substitution(LaunchConfiguration(
        'segment_anything2_is_point_triggered', default='False')))

    # Triton parameters
    sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
    sam2_max_batch_size = LaunchConfiguration('sam2_max_batch_size')

    # DNN Image Encoder parameters
    sam2_encoder_image_mean = LaunchConfiguration('sam2_encoder_image_mean')
    sam2_encoder_image_stddev = LaunchConfiguration('sam2_encoder_image_stddev')

    dropped_image_topic_name = image_input_topic
    dropped_camera_info_topic_name = camera_info_input_topic
    dropped_depth_topic_name = depth_topic_name

    composable_nodes = []
    non_composable_nodes = []

    enforce_max_latency = True
    if context.perform_substitution(LaunchConfiguration('use_sim_time')) == 'true':
        enforce_max_latency = False

    if is_point_triggered == 'True':
        dropped_image_topic_name = '/segment_anything2/image_dropped'
        dropped_camera_info_topic_name = '/segment_anything2/camera_info_dropped'
        dropped_depth_topic_name = '/segment_anything2/depth_dropped'
        composable_nodes.append(
            ComposableNode(
                name='point_triggered_node',
                package='isaac_ros_segment_anything',
                plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingPointTriggeredNode',
                namespace='segment_anything2',
                parameters=[{
                    'mode': 'mono+depth',
                    'is_sam2': True,
                    'max_rate_hz': 2.0,
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
                    ('image_1_triggered', dropped_image_topic_name),
                    ('camera_info_1_triggered', dropped_camera_info_topic_name),
                    ('depth_1', depth_topic_name),
                    ('depth_1_triggered', dropped_depth_topic_name),
                    ('point', segment_anything_input_points_topic),
                ]
            )
        )

    composable_nodes.append(
        ComposableNode(
            name='input_resize_node',
            namespace='segment_anything2',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            parameters=[{
                'input_width': image_width,
                'input_height': image_height,
                'output_width': SAM2_MODEL_INPUT_WIDTH,
                'output_height': SAM2_MODEL_INPUT_HEIGHT,
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
            namespace='segment_anything2',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::PadNode',
            parameters=[{
                'output_image_width': SAM2_MODEL_INPUT_WIDTH,
                'output_image_height': SAM2_MODEL_INPUT_HEIGHT,
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
            namespace='segment_anything2',
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
            namespace='segment_anything2',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
            parameters=[{
                'mean': sam2_encoder_image_mean,
                'stddev': sam2_encoder_image_stddev,
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
            namespace='segment_anything2',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
            parameters=[{
                'input_tensor_shape': [SAM2_MODEL_INPUT_HEIGHT, SAM2_MODEL_INPUT_WIDTH, 3],
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
            namespace='segment_anything2',
            package='isaac_ros_tensor_proc',
            plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
            parameters=[{
                'output_tensor_name': 'input_tensor',
                'input_tensor_shape': [3, SAM2_MODEL_INPUT_HEIGHT, SAM2_MODEL_INPUT_WIDTH],
                'output_tensor_shape': [1, 3, SAM2_MODEL_INPUT_HEIGHT, SAM2_MODEL_INPUT_WIDTH],
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
            name='data_encoder_node',
            namespace='segment_anything2',
            package='isaac_ros_segment_anything2',
            plugin='nvidia::isaac_ros::segment_anything2::SegmentAnything2DataEncoderNode',
            parameters=[{
                'max_num_objects': sam2_max_batch_size,
                'orig_img_dims': [image_image_height_int, image_image_width_int],
            }],
            remappings=[
                ('image', 'tensor_pub'),
                ('memory', 'tensor_sub'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='triton_node',
            namespace='segment_anything2',
            package='isaac_ros_triton',
            plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
            parameters=[{
                'model_name': 'segment_anything2',
                'model_repository_paths': sam_model_repository_paths,
                'max_batch_size': 1,
                'input_tensor_names': [
                    'image', 'bbox_coords', 'point_coords', 'point_labels',
                    'mask_memory', 'obj_ptr_memory', 'original_size', 'permutation'],
                'input_binding_names': [
                    'image', 'bbox_coords', 'point_coords', 'point_labels',
                    'mask_memory', 'obj_ptr_memory', 'original_size', 'permutation'],
                'output_tensor_names': [
                    'high_res_masks', 'object_score_logits', 'maskmem_features',
                    'maskmem_pos_enc', 'obj_ptr_features'],
                'output_binding_names': [
                    'high_res_masks', 'object_score_logits', 'maskmem_features',
                    'maskmem_pos_enc', 'obj_ptr_features'],
            }],
            remappings=[
                ('tensor_pub', 'encoded_data'),
            ]
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='decoder_node',
            namespace='segment_anything2',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
            parameters=[{
                'mask_width': image_width,
                'mask_height': image_height,
                'max_batch_size': sam2_max_batch_size,
            }],
            remappings=[
                ('segment_anything/raw_segmentation_mask', 'raw_segmentation_mask'),
            ],
        )
    )

    composable_nodes.append(
        ComposableNode(
            name='tensor_to_image',
            package='isaac_ros_segment_anything',
            plugin='nvidia::isaac_ros::segment_anything::TensorToImageNode',
            namespace='segment_anything2',
            remappings=[
                ('segmentation_tensor', 'raw_segmentation_mask'),
                ('detection_array', segment_anything2_output_detections_topic),
                ('binary_mask', segment_anything2_output_binary_mask_topic),
            ]
        )
    )

    headless = str(context.perform_substitution(
        LaunchConfiguration('headless'))).lower() == 'true'
    if not headless:
        non_composable_nodes.append(
            Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='segment_anything2_rqt_viewer',
            )
        )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_nodes)

    final_launch = GroupAction(actions=[load_composable_nodes])

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
            'sam2_encoder_image_mean',
            default_value='[0.485, 0.456, 0.406]',
            description='The mean for image normalization'
        ),
        DeclareLaunchArgument(
            'sam2_encoder_image_stddev',
            default_value='[0.229, 0.224, 0.225]',
            description='The standard deviation for image normalization'
        ),
        DeclareLaunchArgument(
            'sam_model_repository_paths',
            default_value='["/tmp/models"]',
            description='The absolute path to the repository of models'
        ),
        DeclareLaunchArgument(
            'sam2_max_batch_size',
            default_value='5',
            description='The maximum allowed batch size of the model'
        ),
        DeclareLaunchArgument(
            'segment_anything2_is_point_triggered',
            default_value='False',
            description='Whether object following is enabled or not.'
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
            'segment_anything2_output_binary_mask_topic',
            default_value='binary_segmentation_mask',
            description='The output topic name for the binary mask'
        ),
        DeclareLaunchArgument(
            'segment_anything2_output_detections_topic',
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
