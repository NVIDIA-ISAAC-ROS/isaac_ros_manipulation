
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
import os
from typing import List

from ament_index_python.packages import get_package_share_directory
# flake8: noqa: I100
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    CameraType, ObjectDetectionType, PoseEstimationType,
    SegmentationType, WorkflowType
)
from isaac_ros_manipulation_ros_python_utils.config import (
    CameraConfig, DopeConfig, FoundationPoseConfig, ObjectSelectionConfig
)

from launch_ros.actions import Node
from launch.actions import (
    IncludeLaunchDescription
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.descriptions import ComposableNode


def get_foundation_pose_nodes(camera_config: CameraConfig,
                              workflow_type: WorkflowType,
                              pose_estimation_config: FoundationPoseConfig
                              ) -> List[Node]:
    """
    Return foundation pose nodes for RT-DETR and Foundation Pose nodes.

    Args
    ----
        camera_config (CameraConfig): Camera config
        workflow_type (WorkflowType): Workflow type
        pose_estimation_config (PoseEstimationConfig): Pose estimation config

    Returns
    -------
        List[Node]: RTDETR and Foundation Pose

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )

    is_object_following = 'False' if workflow_type == WorkflowType.PICK_AND_PLACE else 'True'
    if workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
        is_object_following = 'False'

    # We use a qos profile of default since we use pose servers in pick and place
    # and for object following we use a drop node that limits input hz to rt-detr + foundation pose
    # hence, we should use RELIABLE so that there are no drops.
    # If we use BEST EFFORT for object following, we should remove the drop node. But we do not
    # do this because it causes a lot of load on the Orin AGX system which degrades realtime
    # performance for more granular predictions. Other components are affected as well such as
    # the cuMotion Planner, nvblox, robot segmenter etc.

    object_detection_config = pose_estimation_config.object_detection_config
    segmentation_config = pose_estimation_config.segmentation_config
    if segmentation_config is not None:
        segmentation_config.update_server_topic_names(camera_config)
    object_detection_config.update_server_topic_names(camera_config)
    pose_estimation_config.update_server_topic_names(camera_config)
    enable_dnn_depth_in_realsense = pose_estimation_config.enable_dnn_depth_in_realsense
    pose_estimation_depth_input_topic = pose_estimation_config.fp_in_depth_topic_name
    pose_estimation_depth_image_width = pose_estimation_config.fp_in_depth_image_width
    pose_estimation_depth_image_height = pose_estimation_config.fp_in_depth_image_height

    refine_iterations = '3'
    symmetry_axes = '["x_180", "y_180", "z_180"]'
    if workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
        # The plug and socket are not rotationally symmetric. Treating 180-degree
        # flips as equivalent can rotate the grasp's local +Z offset beside the plug.
        symmetry_axes = '[""]'

    pose_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/foundationpose.launch.py']
        ),
        launch_arguments={
            'camera_type': str(camera_config.camera_type),
            'rgb_image_width': camera_config.rgb_image_width,
            'rgb_image_height': camera_config.rgb_image_height,
            'depth_image_width': pose_estimation_depth_image_width,
            'depth_image_height': pose_estimation_depth_image_height,
            'rgb_image_topic': pose_estimation_config.foundation_pose_rgb_image_topic,
            'rgb_camera_info_topic': pose_estimation_config.foundation_pose_rgb_camera_info,
            'realsense_depth_image_topic': pose_estimation_config.realsense_depth_image_topic,
            'foundation_pose_server_depth_topic_name':
                pose_estimation_config.foundation_pose_depth_image_topic,
            'mesh_file_path': pose_estimation_config.foundation_pose_mesh_file_path,
            'texture_path': pose_estimation_config.foundation_pose_texture_path,
            'refine_engine_file_path':
                pose_estimation_config.foundation_pose_refine_engine_file_path,
            'score_engine_file_path':
                pose_estimation_config.foundation_pose_score_engine_file_path,
            'is_object_following': is_object_following,
            'foundationpose_sensor_qos_config': 'DEFAULT',
            'discard_old_messages': pose_estimation_config.discard_old_messages,
            'discard_msg_older_than_ms': pose_estimation_config.discard_msg_older_than_ms,
            'segmentation_mask_camera_info_topic':
                pose_estimation_config.foundation_pose_rgb_camera_info,
            # This should be correct if pick and place is enabled or not.
            'segmentation_mask_topic': pose_estimation_config.segmentation_mask_topic,
            'output_pose_estimate_topic': pose_estimation_config.fp_in_pose_estimate_topic_name,
            'refine_iterations': refine_iterations,
            'symmetry_axes': symmetry_axes,
            'enable_dnn_depth_in_realsense': 'True' if enable_dnn_depth_in_realsense else 'False',
            'realsense_depth_camera_info_topic': pose_estimation_config.depth_camera_info_for_ess,
        }.items(),
    )

    object_detection_nodes = []
    # Currently, RT-DETR is run all the time even when PICK_AND_PLACE is ON.
    # We should stop this to conserve GPU resources for PICK and PLACE.
    if pose_estimation_config.object_detection_type == ObjectDetectionType.RTDETR:
        object_detection_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/rtdetr.launch.py']
            ),
            launch_arguments={
                'image_width': camera_config.rgb_image_width,
                'image_height': camera_config.rgb_image_height,
                'depth_image_width': pose_estimation_depth_image_width,
                'depth_image_height': pose_estimation_depth_image_height,
                # TODO(kchahal):This should ideally come from perception config.
                'image_input_topic': object_detection_config.rtdetr_camera_input,
                'depth_topic_name': pose_estimation_depth_input_topic,
                'camera_info_input_topic': object_detection_config.rtdetr_camera_info_input,
                'rtdetr_engine_file_path': object_detection_config.rtdetr_engine_file_path,
                'rt_detr_confidence_threshold':
                    object_detection_config.rt_detr_confidence_threshold,
                'detections_2d_array_output_topic': object_detection_config.detections_2d_array_output_topic,
                'rtdetr_is_object_following': is_object_following,
                'object_class_id': object_detection_config.object_class_id,
                'detection2_d_topic': pose_estimation_config.foundation_pose_detections_topic,
                # These are numbers found on the test machine, this might vary for your machine
                'input_fps': pose_estimation_config.input_fps,
                'dropped_fps': pose_estimation_config.dropped_fps,
                'input_qos': pose_estimation_config.input_qos,
                'output_qos': 'DEFAULT',
                'foundationpose_server_input_camera_info_topic':
                    pose_estimation_config.fp_out_camera_info_topic_name
            }.items(),
        ))

    elif pose_estimation_config.object_detection_type == ObjectDetectionType.GROUNDING_DINO:
        object_detection_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/grounding_dino.launch.py']
            ),
            launch_arguments={
                'camera_info_input_topic': object_detection_config.grounding_dino_camera_info_input,
                'depth_image_height': pose_estimation_depth_image_height,
                'depth_image_width': pose_estimation_depth_image_width,
                'depth_topic_name': pose_estimation_depth_input_topic,
                'detection2_d_topic':
                    pose_estimation_config.foundation_pose_detections_topic,
                'detections_2d_array_output_topic':
                    object_detection_config.detections_2d_array_output_topic,
                'dropped_fps': pose_estimation_config.dropped_fps,
                'grounding_dino_confidence_threshold':
                    object_detection_config.grounding_dino_confidence_threshold,
                'grounding_dino_default_prompt':
                    object_detection_config.grounding_dino_default_prompt,
                'grounding_dino_engine_file_path':
                    object_detection_config.grounding_dino_engine_file_path,
                'grounding_dino_is_object_following': is_object_following,
                'grounding_dino_model_file_path':
                    object_detection_config.grounding_dino_model_file_path,
                'image_height': camera_config.rgb_image_height,
                'image_input_topic': object_detection_config.grounding_dino_camera_input,
                'image_width': camera_config.rgb_image_width,
                'input_fps': pose_estimation_config.input_fps,
                'input_qos': pose_estimation_config.input_qos,
                'output_qos': 'DEFAULT',
                'network_image_height': object_detection_config.network_image_height,
                'network_image_width': object_detection_config.network_image_width
            }.items(),
        ))

    if pose_estimation_config.segmentation_type == SegmentationType.SEGMENT_ANYTHING:
        object_detection_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/segment_anything.launch.py']
            ),
            launch_arguments={
                'image_width': camera_config.rgb_image_width,
                'image_height': camera_config.rgb_image_height,
                'image_input_topic': segmentation_config.sam_main_launch_file_image_topic,
                # This is not used when PICK_AND_PLACE is ON.
                'depth_topic_name': pose_estimation_depth_input_topic,
                'camera_info_input_topic':
                    segmentation_config.sam_main_launch_file_camera_info_topic,
                'segment_anything_is_point_triggered': str(segmentation_config.is_point_triggered),
                # These are numbers found on the test machine, this might vary for your machine
                'input_qos': pose_estimation_config.input_qos,
                'output_qos': 'DEFAULT',
                'sam_model_repository_paths': segmentation_config.sam_model_repository_paths,
                # This will only send data if object following is on.
                'segment_anything_input_points_topic':
                    segmentation_config.segment_anything_input_points_topic,
                'segment_anything_input_detections_topic':
                    segmentation_config.sam_out_detections_topic_name,
                # Below values is published by servers if object following is OFF.
                # And if it is ON, then input points is transformed to detection on this topic
                # internally in this launch file.
                'segment_anything_output_binary_mask_topic':
                    segmentation_config.sam_in_mask_topic_name,
                'segment_anything_output_detections_topic':
                    segmentation_config.sam_in_detections_topic_name,
                'headless': str(segmentation_config.headless),
            }.items(),
        ))
    elif pose_estimation_config.segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
        object_detection_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/segment_anything2.launch.py']
            ),
            launch_arguments={
                'image_width': camera_config.rgb_image_width,
                'image_height': camera_config.rgb_image_height,
                'image_input_topic': segmentation_config.sam2_main_launch_file_image_topic,
                # This is not used when PICK_AND_PLACE is ON.
                'depth_topic_name': pose_estimation_depth_input_topic,
                'camera_info_input_topic':
                    segmentation_config.sam2_main_launch_file_camera_info_topic,
                'segment_anything2_is_point_triggered': str(segmentation_config.is_point_triggered),
                # These are numbers found on the test machine, this might vary for your machine
                'input_qos': pose_estimation_config.input_qos,
                'output_qos': 'DEFAULT',
                'sam_model_repository_paths': segmentation_config.sam_model_repository_paths,
                # This will only send data if object following is on.
                'segment_anything_input_points_topic':
                    segmentation_config.segment_anything_input_points_topic,
                # Below values is published by servers if object following is OFF.
                # And if it is ON, then input points is transformed to detection on this topic
                # internally in this launch file.
                'segment_anything2_output_binary_mask_topic':
                    segmentation_config.sam2_in_mask_topic_name,
                'segment_anything2_output_detections_topic':
                    segmentation_config.sam2_in_detections_topic_name,
                'headless': str(segmentation_config.headless),
            }.items(),
        ))
    elif pose_estimation_config.segmentation_type != SegmentationType.NONE:
        raise NotImplementedError(
            f'Segmentation type {pose_estimation_config.segmentation_type} not supported')

    return [pose_estimator] + object_detection_nodes


def get_dope_nodes(camera_config: CameraConfig, dope_config: DopeConfig,
                   workflow_type: WorkflowType) -> List[Node]:
    """
    Get DOPE nodes.

    Returns
    -------
        List[Node]: Node

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )
    image_input_topic = camera_config.color_camera_topic_name
    camera_info_input_topic = camera_config.color_camera_info_topic_name
    if workflow_type == WorkflowType.PICK_AND_PLACE:
        image_input_topic = 'dope_server/image_rect'
        camera_info_input_topic = 'dope_server/camera_info'
    elif workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
        raise NotImplementedError(
            f'{workflow_type.value} is not supported yet for DOPE')

    pose_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/dope.launch.py']
        ),
        launch_arguments={
            'image_input_topic': image_input_topic,
            'camera_info_input_topic': camera_info_input_topic,
            'input_image_width': camera_config.rgb_image_width,
            'input_image_height': camera_config.rgb_image_height,
            'dope_network_image_width': camera_config.rgb_image_width,
            'dope_network_image_height': camera_config.rgb_image_height,
            'dope_model_file_path': dope_config.dope_model_file_path,
            'dope_engine_file_path': dope_config.dope_engine_file_path,
            'dope_enable_tf_publishing': dope_config.dope_enable_tf_publishing,
            'rotation_y_axis': dope_config.dope_rotation_y_axis,
            'input_fps': dope_config.input_fps,
            'dropped_fps': dope_config.dropped_fps,
            'dope_input_qos': dope_config.dope_input_qos
        }.items(),
    )

    return [pose_estimator]


def get_object_detection_servers(camera_config: CameraConfig,
                                 pose_estimation_config: FoundationPoseConfig
                                 ) -> List[IncludeLaunchDescription]:
    """
    Object detection servers.

    Args
    ----
        camera_config (CameraConfig): Camera config
        pose_estimation_config (FoundationPoseConfig): Pose estimation config

    Returns
    -------
        List[IncludeLaunchDescription]: Include launch description

    """
    object_detection_type = pose_estimation_config.object_detection_type
    segmentation_type = pose_estimation_config.segmentation_type
    pose_estimation_type = pose_estimation_config.pose_estimation_type

    object_detection_config = pose_estimation_config.object_detection_config
    segmentation_config = pose_estimation_config.segmentation_config

    # Add objectinfo servers
    isaac_ros_manipulation_servers_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_servers'), 'launch')
    perception_servers = []
    # In Isaac Sim, we use DEFAULT QoS as we can apply backpressure to the input topic in sim
    # and make sim tick slower so that it can injest all the data
    sub_qos = 'DEFAULT' if camera_config.camera_type == CameraType.ISAAC_SIM else 'SENSOR_DATA'
    # Below is always default as we do not want to drop messages after this barrier of the servers
    pub_qos = 'DEFAULT'
    if object_detection_type == ObjectDetectionType.RTDETR:
        object_detection_server_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_servers_include_dir, '/object_detection_server.launch.py']),
            launch_arguments={
                'obj_input_img_topic_name': object_detection_config.rtdetr_in_img_topic_name,
                'obj_output_img_topic_name': object_detection_config.rtdetr_out_img_topic_name,
                'obj_input_camera_info_topic_name':
                    object_detection_config.rtdetr_in_camera_info_topic_name,
                'obj_output_camera_info_topic_name':
                    object_detection_config.rtdetr_out_camera_info_topic_name,
                'obj_output_detections_topic_name':
                    object_detection_config.rtdetr_out_detections_topic_name,
                'obj_input_detections_topic_name':
                    object_detection_config.rtdetr_in_detections_topic_name,
                'obj_input_qos': sub_qos,
                'obj_result_and_output_qos': pub_qos
            }.items()
        )
        perception_servers.append(object_detection_server_launch)
    elif object_detection_type == ObjectDetectionType.GROUNDING_DINO:
        object_detection_server_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_servers_include_dir, '/object_detection_server.launch.py']),
            launch_arguments={
                'obj_input_img_topic_name':
                    object_detection_config.grounding_dino_in_img_topic_name,
                'obj_output_img_topic_name':
                    object_detection_config.grounding_dino_out_img_topic_name,
                'obj_input_camera_info_topic_name':
                    object_detection_config.grounding_dino_in_camera_info_topic_name,
                'obj_output_camera_info_topic_name':
                    object_detection_config.grounding_dino_out_camera_info_topic_name,
                'obj_output_detections_topic_name':
                    object_detection_config.grounding_dino_out_detections_topic_name,
                'obj_input_detections_topic_name':
                    object_detection_config.grounding_dino_in_detections_topic_name,
                'obj_input_qos': sub_qos,
                'obj_result_and_output_qos': pub_qos
            }.items()
        )
        perception_servers.append(object_detection_server_launch)

    if (
        segmentation_type == SegmentationType.SEGMENT_ANYTHING or
        object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING
    ):
        if segmentation_type != SegmentationType.SEGMENT_ANYTHING:
            raise NotImplementedError(f'Segmentation type {segmentation_type} not supported, '
                                      'If object detection is SEGMENT_ANYTHING, then segmentation '
                                      'must be SEGMENT_ANYTHING')

        segmentation_server_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_servers_include_dir, '/segment_anything_server.launch.py']),
            launch_arguments={
                'is_sam2': 'False',
                'sam_in_img_topic_name': segmentation_config.sam_in_img_topic_name,
                'sam_out_img_topic_name': segmentation_config.sam_out_img_topic_name,
                'sam_in_camera_info_topic_name': segmentation_config.sam_in_camera_info_topic_name,
                'sam_out_camera_info_topic_name':
                    segmentation_config.sam_out_camera_info_topic_name,
                'sam_in_segmentation_mask_topic_name': segmentation_config.sam_in_mask_topic_name,
                'sam_in_detections_topic_name': segmentation_config.sam_in_detections_topic_name,
                'sam_out_detections_topic_name': segmentation_config.sam_out_detections_topic_name,
                'sam_input_qos': sub_qos,
                'sam_result_and_output_qos': pub_qos
            }.items()
        )
        perception_servers.append(segmentation_server_launch)
    elif (
        segmentation_type == SegmentationType.SEGMENT_ANYTHING2 or
        object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2
    ):
        if segmentation_type != SegmentationType.SEGMENT_ANYTHING2:
            raise NotImplementedError(f'Segmentation type {segmentation_type} not supported, '
                                      'If object detection is SEGMENT_ANYTHING2, then segmentation '
                                      'must be SEGMENT_ANYTHING2')

        segmentation_server_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_servers_include_dir, '/segment_anything_server.launch.py']),
            launch_arguments={
                'is_sam2': 'True',
                'sam_in_img_topic_name': segmentation_config.sam2_in_img_topic_name,
                'sam_out_img_topic_name': segmentation_config.sam2_out_img_topic_name,
                'sam_in_camera_info_topic_name': segmentation_config.sam2_in_camera_info_topic_name,
                'sam_out_camera_info_topic_name':
                    segmentation_config.sam2_out_camera_info_topic_name,
                'sam_in_segmentation_mask_topic_name': segmentation_config.sam2_in_mask_topic_name,
                'sam_in_detections_topic_name': segmentation_config.sam2_in_detections_topic_name,
                'sam_input_qos': sub_qos,
                'sam_result_and_output_qos': pub_qos
            }.items()
        )
        perception_servers.append(segmentation_server_launch)

    if pose_estimation_type == PoseEstimationType.FOUNDATION_POSE:
        pose_estimation_server_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_servers_include_dir, '/foundation_pose_server.launch.py']),
            launch_arguments={
                'fp_in_img_topic_name': pose_estimation_config.fp_in_img_topic_name,
                'fp_in_depth_topic_name': pose_estimation_config.fp_in_depth_topic_name,
                'fp_in_camera_info_topic_name':
                    pose_estimation_config.fp_in_camera_info_topic_name,
                'fp_out_img_topic_name': pose_estimation_config.fp_out_img_topic_name,
                'fp_out_depth_topic_name': pose_estimation_config.fp_out_depth_topic_name,
                'fp_out_camera_info_topic_name':
                    pose_estimation_config.fp_out_camera_info_topic_name,
                'fp_out_bbox_topic_name': pose_estimation_config.fp_out_detections_topic_name,
                'fp_in_pose_estimate_topic_name':
                    pose_estimation_config.fp_in_pose_estimate_topic_name,
                'fp_out_pose_estimate_topic_name':
                    pose_estimation_config.fp_out_pose_estimate_topic_name,
                'fp_out_segmented_mask_topic_name':
                    pose_estimation_config.fp_out_segmented_mask_topic_name,
                'fp_input_qos': sub_qos,
                'fp_result_and_output_qos': pub_qos
            }.items()
        )
        perception_servers.append(pose_estimation_server_launch)
    elif pose_estimation_type == PoseEstimationType.DOPE:
        # This is not supported yet, but it should not be too difficult to add for those
        # who want to use DOPE servers. one would need to support ability of DOPE to output
        # detections in 2D array format.
        raise NotImplementedError('Add functionality for Dope servers to run this config')

    if segmentation_type == SegmentationType.SEGMENT_ANYTHING:
        segmentation_backend = 'SEGMENT_ANYTHING'
    elif segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
        segmentation_backend = 'SEGMENT_ANYTHING2'
    else:
        segmentation_backend = 'NONE'

    if object_detection_type == ObjectDetectionType.GROUNDING_DINO:
        object_detection_backend = 'GROUNDING_DINO'
    elif object_detection_type == ObjectDetectionType.RTDETR:
        object_detection_backend = 'RT_DETR'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING:
        object_detection_backend = 'SEGMENT_ANYTHING'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2:
        object_detection_backend = 'SEGMENT_ANYTHING2'
    else:
        raise NotImplementedError(f'Object detection type {object_detection_type} not supported')

    pose_estimation_backend = 'FOUNDATION_POSE' \
        if pose_estimation_type == PoseEstimationType.FOUNDATION_POSE else 'NONE'

    objectinfo_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_ros_manipulation_servers_include_dir, '/object_info_server.launch.py']),
        launch_arguments={
            'segmentation_backend': segmentation_backend,
            'object_detection_backend': object_detection_backend,
            'pose_estimation_backend': pose_estimation_backend
        }.items()
    )
    perception_servers.append(objectinfo_server_launch)

    return perception_servers


def get_object_selection_server(
    object_selection_config: ObjectSelectionConfig
) -> [IncludeLaunchDescription]:
    """
    Get object selection server.

    Args
    ----
        object_selection_config (ObjectSelectionConfig): Object selection config

    Returns
    -------
        IncludeLaunchDescription: Include launch description for object selection server

    """
    isaac_ros_manipulation_servers_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_servers'), 'launch')
    object_selection_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_ros_manipulation_servers_include_dir, '/object_selection_server.launch.py']),
        launch_arguments={
            'selection_policy': str(object_selection_config.selection_policy)
        }.items()
    )
    return [object_selection_server_launch]


def create_camera_drop_node(node_name: str,
                            namespace: str,
                            input_fps: int,
                            output_fps: int,
                            topic_remappings: dict,
                            input_qos: str = 'SENSOR_DATA',
                            output_qos: str = 'SENSOR_DATA',
                            max_latency_threshold: float = 0.1,
                            enforce_max_latency: bool = True,
                            mode: str = 'stereo',
                            sync_queue_size: int = 100) -> ComposableNode:
    """
    Create a ComposableNode for a camera drop node with configurable topic remappings.

    Args
    ----
        node_name (str): Name of the drop node
        namespace (str): Namespace for the node
        input_fps (int): Input frames per second
        output_fps (int): Output frames per second
        input_qos (str, optional): Input QoS setting. Defaults to 'SENSOR_DATA'.
        output_qos (str, optional): Output QoS setting. Defaults to 'SENSOR_DATA'.
        mode (str, optional): Drop node mode ('stereo' or 'mono'). Defaults to 'stereo'.
        sync_queue_size (int, optional): Synchronization queue size. Defaults to 100.
        topic_remappings (dict, optional): Dictionary of topic remappings.
                                           If not provided, uses default remappings.
        max_latency_threshold (float, optional): Maximum latency threshold. Defaults to 0.1.
        enforce_max_latency (bool, optional): Enforce max latency. Defaults to True.

    Returns
    -------
        ComposableNode: Drop node for camera

    """
    # Calculate drop rate (X out of Y frames)
    X = output_fps
    Y = input_fps

    # Create the drop node
    drop_node = ComposableNode(
        name=node_name,
        package='isaac_ros_topic_tools',
        plugin='nvidia::isaac_ros::topic_tools::CameraDropNode',
        namespace=namespace,
        parameters=[{
            'input_qos': input_qos,
            'output_qos': output_qos,
            'X': X,
            'Y': Y,
            'mode': mode,
            'sync_queue_size': sync_queue_size,
            'max_latency_threshold': max_latency_threshold,
            'enforce_max_latency': enforce_max_latency,
        }],
        remappings=[(src, dst) for src, dst in topic_remappings.items()]
    )

    return drop_node
