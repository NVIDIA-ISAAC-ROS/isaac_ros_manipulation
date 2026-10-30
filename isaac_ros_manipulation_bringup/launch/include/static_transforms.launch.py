# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List

import isaac_ros_launch_utils as lu
from isaac_ros_launch_utils.all_types import Action, LaunchDescription
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    CameraType, TrackingType,
)


# Dictionary containing the calibration of various camera setups.
# Every item of the dictionary represents the calibration of a single setup
calibrations_dict = {
    'rosie_ur10e_test_bench': {
        'world_pose_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-1.0344, -1.41359, 0.586203],
            'rotation': [-0.13085, 0.161075, 0.629893, 0.748443],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-0.414329, 0.242295, 0.129507],
            'rotation': [-0.0533118, 0.00148028, 0.998085, -0.0313258],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'hubble_ur10e_test_bench': {
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-1.24786, 0.22986, 0.262594],
            'rotation': [-0.00930892, 0.482916, 0.0158122, 0.875474],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-1.62286, 0.704411, 0.451326],
            'rotation': [0.0595995, 0.157255, -0.300931, 0.938701],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_3': {
            'parent_frame': 'world',
            'child_frame': 'camera_3_link',
            'translation': [-1.64437, -0.43066, 0.452024],
            'rotation': [0.0286703, 0.169328, 0.152123, 0.973326],  # [qx, qy ,qz, qw]
        },
        'world_pose_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'flexiv_test_bench': {
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [0.931374, -0.246224, 0.290686],
            'rotation': [-0.188498, 0.035703, 0.978276, 0.078546],  # [qx, qy, qz, qw]
        },
        'world_pose_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [0.5, -0.15, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [0.5, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'zurich_test_bench': {
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [2.131679, 0.563435, 0.775389],
            'rotation': [-0.324828, 0.001799, 0.945766, -0.002907],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-0.250322, 0.598947, 0.864349],
            'rotation': [0.039936, 0.349054, -0.058430, 0.934426],  # [qx, qy ,qz, qw]
        },
        'world_pose_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'hubble_ur5e_test_bench': {
        'world_pose_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-0.751471, 0.649335, 0.506522],
            'rotation': [0.0433558, 0.140372, -0.509666, 0.847736],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-0.651394, -0.763559, 0.342156],
            'rotation': [-0.031729, 0.0792618, 0.653756, 0.751874],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.3, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'galileo_ur10e_test_bench': {
        'world_pose_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-0.48429, 1.15513, 0.452704],
            'rotation': [0.0879499, 0.18293, -0.480816, 0.853005],  # [qx, qy ,qz, qw]
        },
        'world_pose_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [0.32545, 1.14985, 0.331019],
            'rotation': [0.148563, 0.0800837, -0.741462, 0.649423],  # [qx, qy ,qz, qw]
        },
        'world_pose_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_pose_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.4, 0.6, 0.2],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_pose_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [0.2, 0.6, 0.2],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
}


def static_transform_from_dict(transform_dict):
    return lu.static_transform(
        parent=transform_dict['parent_frame'],
        child=transform_dict['child_frame'],
        translation=transform_dict['translation'],
        orientation_quaternion=transform_dict['rotation'])


def add_static_transforms(args: lu.ArgumentContainer) -> List[Action]:
    camera_type = CameraType[args.camera_type]
    tracking_type = TrackingType[args.tracking_type]
    num_cameras = int(args.num_cameras)
    broadcast_world_base_link = bool(args.broadcast_world_base_link)

    # Get the calibration dict
    if args.calibration_name not in calibrations_dict and camera_type is not CameraType.ISAAC_SIM:
        return [
            lu.log_info([
                "Calibration with name '",
                str(args.calibration_name), "' does not exits. Not loading static transforms."
            ])
        ]

    actions = []
    if camera_type is CameraType.ISAAC_SIM:
        return actions

    transforms = calibrations_dict[args.calibration_name]

    # Get world transform
    if broadcast_world_base_link:
        actions.append(static_transform_from_dict(transforms['world_pose_base_link']))

    # Get camera transforms
    if camera_type is CameraType.REALSENSE:
        actions.append(static_transform_from_dict(transforms['world_pose_realsense_1']))
        if num_cameras > 1:
            actions.append(static_transform_from_dict(transforms['world_pose_realsense_2']))
            assert num_cameras <= 2, 'Running more than 2 cameras not allowed.'
    else:
        raise Exception(f'CameraType {camera_type} not implemented.')

    # Get target and grasp frames
    if tracking_type is TrackingType.OBJECT_FOLLOWING:
        actions.append(static_transform_from_dict(transforms['object_pose_grasp_frame']))
    elif tracking_type is TrackingType.POSE_TO_POSE:
        actions.append(static_transform_from_dict(transforms['world_pose_target_frame_1']))
        actions.append(static_transform_from_dict(transforms['world_pose_target_frame_2']))
    elif tracking_type is not TrackingType.NONE:
        raise Exception(f'TrackingType {tracking_type} not implemented.')

    actions.append(
        lu.log_info([
            "Successfully loaded the static transforms of the '",
            str(args.calibration_name),
            "' calibration.",
        ]))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('num_cameras', 1)
    args.add_arg('broadcast_world_base_link', False)
    args.add_arg('camera_type')
    args.add_arg('tracking_type')
    args.add_arg('calibration_name', '')
    args.add_arg('gripper_type', 'robotiq_2f_140')
    args.add_arg('robot_type', 'UR')

    args.add_opaque_function(add_static_transforms)
    return LaunchDescription(args.get_launch_actions())
