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

import enum


class GripperType(enum.Enum):
    """Define a list of grippers supported by Manipulator."""

    ROBOTIQ_2F_140 = 'robotiq_2f_140'
    ROBOTIQ_2F_85 = 'robotiq_2f_85'
    GRAV = 'grav'

    @classmethod
    def get_gripper_type(self, gripper_type: str):
        """
        Get the gripper type.

        Args
        ----
            gripper_type (str): Get str version of gripper type

        Returns
        -------
            GripperType: returns typed value

        """
        if gripper_type == GripperType.ROBOTIQ_2F_140.value:
            return GripperType.ROBOTIQ_2F_140
        elif gripper_type == GripperType.ROBOTIQ_2F_85.value:
            return GripperType.ROBOTIQ_2F_85
        elif gripper_type == GripperType.GRAV.value:
            return GripperType.GRAV
        else:
            raise NotImplementedError(f'Gripper type {gripper_type} not supported')


class EnumMeta(enum.EnumMeta):
    """Enum metaclass to add meaningful error messages for KeyErrors and AttributeErrors."""

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError as error:
            raise KeyError(f'The key {name} is not part of the {self.__name__} enum '
                           f'(valid options are {self.names()}). KeyError: {error}')

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            raise AttributeError(f'The attribute {name} is not part of the {self.__name__} enum. '
                                 f'AttributeError: {error}')


class ManipulatorEnum(enum.Enum, metaclass=EnumMeta):
    """Base class for manipulator enums defining common functions."""

    @classmethod
    def names(cls):
        return [mode.name for mode in cls]

    def __str__(self):
        return self.name


class CameraType(ManipulatorEnum):
    """Enum defining the camera that manipulator should be run with."""

    REALSENSE = 1
    HAWK = 2
    ISAAC_SIM = 3


class DepthType(ManipulatorEnum):
    """Enum defining the depth estimation model."""

    ESS_FULL = 1
    ESS_LIGHT = 2
    REALSENSE = 3
    ISAAC_SIM = 4
    FOUNDATION_STEREO_LOW_RES = 5
    FOUNDATION_STEREO_HIGH_RES = 6


class TrackingType(ManipulatorEnum):
    """Enum defining the tracking type that manipulator should be run in."""

    NONE = 0
    POSE_TO_POSE = 1
    OBJECT_FOLLOWING = 2
    DISPLAYPORT_INSERTION = 4


class PoseEstimationType(ManipulatorEnum):
    """Enum defining the pose estimation model to be used if TrackingType is follow_object."""

    FOUNDATION_POSE = 1
    DOPE = 2


class SegmentationType(ManipulatorEnum):
    """Enum defining the segmentation model to be used."""

    SEGMENT_ANYTHING = 1
    SEGMENT_ANYTHING2 = 2
    NONE = 3


class ObjectDetectionType(ManipulatorEnum):
    """Enum defining the object detection model to be used."""

    RTDETR = 1
    SEGMENT_ANYTHING = 2
    SEGMENT_ANYTHING2 = 3
    GROUNDING_DINO = 4
    DOPE = 5


class ObjectSelectionType(ManipulatorEnum):
    """Enum defining the object selection type to be used."""

    FIRST = 'first'
    RANDOM = 'random'
    HIGHEST_SCORE = 'highest_score'


class ObjectAttachmentShape(enum.Enum):
    """Object attachment types."""

    SPHERE = 'SPHERE'
    CUBOID = 'CUBOID'
    CUSTOM_MESH = 'CUSTOM_MESH'


class Mode(enum.Enum):
    """
    Enumeration for operation modes.

    Values
    ------
        ONCE: Execute the action once.
        CYCLE: Continuously cycle through attach and detach actions.

    """

    ONCE = 'once'
    CYCLE = 'cycle'


class AttachState(enum.Enum):
    """
    Enumeration for attachment states.

    Values
    ------
        ATTACH: Attach the object.
        DETACH: Detach the object.

    """

    ATTACH = True
    DETACH = False


class ObjectSelectionPolicy(enum.Enum):
    """
    Enumeration for selection policies.

    Values
    ------
        FIRST: Select the first object.
        RANDOM: Select a random object.
        HIGHEST_SCORE: Select the object with the highest score.
    """

    FIRST = 'first'
    RANDOM = 'random'
    HIGHEST_SCORE = 'highest_score'


class WorkflowType(enum.Enum):
    """Workflow types that are supported by Isaac Manipulator."""

    POSE_TO_POSE = 'POSE_TO_POSE'
    PICK_AND_PLACE = 'PICK_AND_PLACE'
    OBJECT_FOLLOWING = 'OBJECT_FOLLOWING'
    DISPLAYPORT_CABLE_INSERTION = 'DISPLAYPORT_CABLE_INSERTION'


class RobotType(ManipulatorEnum):
    """
    Robot families supported by Isaac Manipulator.

    Used to drive type-safe dispatch in shared launch utilities (e.g. TF frame
    prefix and arm joint name derivation) instead of sniffing for
    robot-family-specific launch args (``robot_sn`` for Flexiv, ``tf_prefix``
    for UR).
    """

    UR = 'UR'
    FLEXIV = 'FLEXIV'


class ObjectStatus(enum.Enum):
    """Object status enum for orchestration workflows."""

    NOT_READY = 'NOT_READY'                # Detection done, no pose estimation yet
    SELECTED = 'SELECTED'                  # Chosen by the Selector; awaiting pose estimation
    READY_FOR_MOTION = 'READY_FOR_MOTION'  # Perception completed; queued for motion
    IN_MOTION = 'IN_MOTION'                # Currently being processed
    DONE = 'DONE'                          # Finished
    FAILED = 'FAILED'                      # Motion attempt failed
