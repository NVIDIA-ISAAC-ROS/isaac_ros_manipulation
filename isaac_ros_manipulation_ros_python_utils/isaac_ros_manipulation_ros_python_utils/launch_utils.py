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
import ast
import math
from typing import List, Tuple

from geometry_msgs.msg import Pose, Vector3
import isaac_ros_manipulation_ros_python_utils.constants as constants
# flake8: noqa: I100
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    CameraType,
    DepthType,
    GripperType,
    ObjectAttachmentShape,
    ObjectDetectionType,
    ObjectSelectionType,
    PoseEstimationType,
    RobotType,
    SegmentationType,
    WorkflowType,
)
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration


def get_dnn_stereo_depth_resolution(depth_type: DepthType) -> Tuple[str, str]:
    """
    Get the DNN stereo depth model resolution for the given depth type.

    Returns
    -------
        str, str: width, height

    """
    dnn_depth_resolution = {
        DepthType.ESS_FULL: (constants.ESS_INPUT_IMAGE_WIDTH,
                             constants.ESS_INPUT_IMAGE_HEIGHT),
        DepthType.ESS_LIGHT: (constants.ESS_LIGHT_INPUT_IMAGE_WIDTH,
                              constants.ESS_LIGHT_INPUT_IMAGE_HEIGHT),
        DepthType.FOUNDATION_STEREO_LOW_RES: (
            constants.FOUNDATION_STEREO_LOW_RES_INPUT_IMAGE_WIDTH,
            constants.FOUNDATION_STEREO_LOW_RES_INPUT_IMAGE_HEIGHT),
        DepthType.FOUNDATION_STEREO_HIGH_RES: (
            constants.FOUNDATION_STEREO_HIGH_RES_INPUT_IMAGE_WIDTH,
            constants.FOUNDATION_STEREO_HIGH_RES_INPUT_IMAGE_HEIGHT),
    }
    if depth_type not in dnn_depth_resolution:
        raise ValueError(
            f'depth_type {depth_type} is not a valid DNN stereo depth type. '
            f'Valid types: {list(dnn_depth_resolution.keys())}')
    width, height = dnn_depth_resolution[depth_type]
    return str(width), str(height)


def get_str_variable(context: LaunchContext, variable_name: str) -> str:
    """
    Return a string from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        str: Returns string representation

    """
    return str(
        context.perform_substitution(LaunchConfiguration(variable_name))
    )


def get_float_variable(context: LaunchContext, variable_name: str) -> float:
    """
    Return a float from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        float: Returns float representation

    """
    return float(
        context.perform_substitution(LaunchConfiguration(variable_name))
    )


def get_float_list_variable(context: LaunchContext, variable_name: str) -> List[float]:
    """
    Return a float list from a launch variable.

    YAML list params are passed through launch as their Python string representation.
    """
    str_var = str(context.perform_substitution(LaunchConfiguration(variable_name))).strip()
    if not str_var or str_var in ('[]', "''", 'None'):
        return []

    try:
        parsed = ast.literal_eval(str_var)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f'{variable_name} must be a list of floats, got {str_var!r}') from exc

    if not isinstance(parsed, list):
        raise ValueError(f'{variable_name} must be a list, got {type(parsed).__name__}')

    return [float(value) for value in parsed]


def normalize_pose_quaternion(pose_values: List[float],
                              param_name: str = 'pose') -> List[float]:
    """Validate and normalize a pose array ``[x, y, z, qx, qy, qz, qw]``."""
    if len(pose_values) != 7:
        raise ValueError(
            f'{param_name} must contain 7 values [x, y, z, qx, qy, qz, qw], '
            f'got {len(pose_values)}')

    x, y, z, qx, qy, qz, qw = (float(value) for value in pose_values)
    if not all(math.isfinite(value) for value in (x, y, z, qx, qy, qz, qw)):
        raise ValueError(f'{param_name} values must be finite numbers')

    quat_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if quat_norm <= 1e-9:
        raise ValueError(f'{param_name} quaternion must be non-zero')

    return [x, y, z, qx / quat_norm, qy / quat_norm, qz / quat_norm, qw / quat_norm]


def get_bool_variable(context: LaunchContext, variable_name: str) -> bool:
    """
    Return a boolean from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        bool: Returns boolean representation

    """
    str_var = str(context.perform_substitution(LaunchConfiguration(variable_name)))
    return True if str_var == 'true' else False


def get_camera_type(camera_type_str: str) -> CameraType:
    """
    Get the pythonic camera type.

    Args
    ----
        camera_type_str (str): Str from user

    Raises
    ------
        NotImplementedError: If str is not supported

    Returns
    -------
        CameraType: Camera type variable

    """
    if camera_type_str == str(CameraType.ISAAC_SIM):
        return CameraType.ISAAC_SIM
    elif camera_type_str == str(CameraType.REALSENSE):
        return CameraType.REALSENSE
    else:
        raise NotImplementedError(f'Camera type {camera_type_str} not supported')


def get_object_attachment_type(object_attachment_type: str) -> ObjectAttachmentShape:
    """
    Get pythonic object attachment type.

    Args
    ----
        object_attachment_type (str): Object attachment types

    Returns
    -------
        ObjectAttachmentShape: Return value

    """
    if object_attachment_type == ObjectAttachmentShape.CUBOID.value:
        return ObjectAttachmentShape.CUBOID
    elif object_attachment_type == ObjectAttachmentShape.CUSTOM_MESH.value:
        return ObjectAttachmentShape.CUSTOM_MESH
    elif object_attachment_type == ObjectAttachmentShape.SPHERE.value:
        return ObjectAttachmentShape.SPHERE
    else:
        raise NotImplementedError(f'Object attachment type {object_attachment_type} not supported')


def get_robot_type(context: LaunchContext) -> RobotType:
    """
    Resolve the ``robot_type`` launch arg to a ``RobotType`` enum.

    The launch arg value must match one of the ``RobotType`` members
    (case-sensitive, e.g. ``UR`` or ``FLEXIV``).

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        RobotType: Robot type object

    Raises
    ------
        NotImplementedError: If the ``robot_type`` launch arg value is not a
            supported ``RobotType``

    """
    robot_type_str = get_str_variable(context, 'robot_type')
    if robot_type_str == str(RobotType.UR):
        return RobotType.UR
    elif robot_type_str == str(RobotType.FLEXIV):
        return RobotType.FLEXIV
    else:
        raise NotImplementedError(f'Robot type {robot_type_str} not supported !')


def get_workflow_type(workflow_type_str: str) -> WorkflowType:
    """
    Get workflow type object.

    Args
    ----
        workflow_type_str (str): Workflow type str

    Returns
    -------
        WorkflowType: Workflow type object

    """
    if workflow_type_str == WorkflowType.OBJECT_FOLLOWING.value:
        return WorkflowType.OBJECT_FOLLOWING
    elif workflow_type_str == WorkflowType.PICK_AND_PLACE.value:
        return WorkflowType.PICK_AND_PLACE
    elif workflow_type_str == WorkflowType.POSE_TO_POSE.value:
        return WorkflowType.POSE_TO_POSE
    elif workflow_type_str == WorkflowType.DISPLAYPORT_CABLE_INSERTION.value:
        return WorkflowType.DISPLAYPORT_CABLE_INSERTION
    else:
        raise NotImplementedError(f'Workflow type {workflow_type_str} not supported !')


def get_pose_estimation_type(pose_estimation_str: str) -> PoseEstimationType:
    """
    Pose estimation type of the context string.

    Args
    ----
        pose_estimation_str (str): Pose estimator type

    Returns
    -------
        PoseEstimationType: Pose estimation type object

    """
    if pose_estimation_str == str(PoseEstimationType.DOPE):
        return PoseEstimationType.DOPE
    elif pose_estimation_str == str(PoseEstimationType.FOUNDATION_POSE):
        return PoseEstimationType.FOUNDATION_POSE
    else:
        raise NotImplementedError(f'Pose estimation type {pose_estimation_str} not supported')


def extract_pose_from_parameter(pose_list: list) -> Pose:
    """
    Extract pose from a launch param.

    Args
    ----
        pose_dict (dict): The pose dict

    Raises
    ------
        ValueError: Value Error if key not found or missing key
        ValueError: Value error if invalid value that is not float

    Returns
    -------
        Pose: The extracted pose

    """
    try:
        pose = Pose()
        x, y, z, rot_x, rot_y, rot_z, rot_w = pose_list
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        pose.orientation.x = float(rot_x)
        pose.orientation.y = float(rot_y)
        pose.orientation.z = float(rot_z)
        pose.orientation.w = float(rot_w)
        return pose
    except ValueError as e:
        raise ValueError(f'Invalid value in pose: {e}')


def extract_vector3_from_parameter(scale_dict: list) -> Vector3:
    """
    Extract a Vector3 from a launch parameter.

    Args
    ----
        scale_dict (dict): The scale dictionary.

    Raises
    ------
        ValueError: If a value cannot be converted to a float.

    Returns
    -------
        Vector3: The extracted Vector3 message.

    """
    try:
        scale = Vector3()

        scale.x = float(scale_dict[0])
        scale.y = float(scale_dict[1])
        scale.z = float(scale_dict[2])

        return scale
    except ValueError as e:
        raise ValueError(f'Invalid value in scale: {e}')


def get_depth_type(depth_type_str: str) -> DepthType:
    """
    Get the pythonic depth type.

    Args
    ----
        depth_type_str (str): Depth type str

    Returns
    -------
        DepthType: Depth type object

    """
    if depth_type_str == str(DepthType.ESS_FULL):
        return DepthType.ESS_FULL
    elif depth_type_str == str(DepthType.ESS_LIGHT):
        return DepthType.ESS_LIGHT
    elif depth_type_str == str(DepthType.FOUNDATION_STEREO_LOW_RES):
        return DepthType.FOUNDATION_STEREO_LOW_RES
    elif depth_type_str == str(DepthType.FOUNDATION_STEREO_HIGH_RES):
        return DepthType.FOUNDATION_STEREO_HIGH_RES
    elif depth_type_str == str(DepthType.REALSENSE):
        return DepthType.REALSENSE
    elif depth_type_str == str(DepthType.ISAAC_SIM):
        return DepthType.ISAAC_SIM
    else:
        raise NotImplementedError(f'Depth type {depth_type_str} not supported')


def get_object_detection_namespace(object_detection_type: ObjectDetectionType) -> str:
    """
    Get the namespace for the object detection type.

    Args
    ----
        object_detection_type (ObjectDetectionType): Object detection type

    Returns
    -------
        str: Namespace for the object detection type

    """
    if object_detection_type == ObjectDetectionType.GROUNDING_DINO:
        return 'grounding_dino'
    elif object_detection_type == ObjectDetectionType.RTDETR:
        return 'rtdetr'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING:
        return 'segment_anything'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2:
        return 'segment_anything2'
    else:
        raise NotImplementedError(f'Object detection type {object_detection_type} not supported')


def get_object_detection_type(object_detection_type_str: str) -> ObjectDetectionType:
    """
    Get the pythonic object detection type.

    Args
    ----
        object_detection_type_str (str): Object detection type str

    Returns
    -------
        ObjectDetectionType: Object detection type object

    """
    if object_detection_type_str == str(ObjectDetectionType.DOPE):
        return ObjectDetectionType.DOPE
    elif object_detection_type_str == str(ObjectDetectionType.GROUNDING_DINO):
        return ObjectDetectionType.GROUNDING_DINO
    elif object_detection_type_str == str(ObjectDetectionType.RTDETR):
        return ObjectDetectionType.RTDETR
    elif object_detection_type_str == str(ObjectDetectionType.SEGMENT_ANYTHING):
        return ObjectDetectionType.SEGMENT_ANYTHING
    elif object_detection_type_str == str(ObjectDetectionType.SEGMENT_ANYTHING2):
        return ObjectDetectionType.SEGMENT_ANYTHING2
    else:
        raise NotImplementedError(
            f'Object detection type {object_detection_type_str} not supported')


def get_object_selection_type(object_selection_type_str: str) -> ObjectSelectionType:
    """
    Get the pythonic object selection type.

    Args
    ----
        object_selection_type_str (str): Object selection type str

    Returns
    -------
        ObjectSelectionType: Object selection type object

    """
    if object_selection_type_str == str(ObjectSelectionType.FIRST):
        return ObjectSelectionType.FIRST
    elif object_selection_type_str == str(ObjectSelectionType.RANDOM):
        return ObjectSelectionType.RANDOM
    elif object_selection_type_str == str(ObjectSelectionType.HIGHEST_SCORE):
        return ObjectSelectionType.HIGHEST_SCORE
    else:
        raise NotImplementedError(
            f'Object selection type {object_selection_type_str} not supported')


def get_segmentation_type(segmentation_type_str: str) -> SegmentationType:
    """
    Get the pythonic segmentation type.

    Args
    ----
        segmentation_type_str (str): Segmentation type str

    Returns
    -------
        SegmentationType: Segmentation type object

    """
    if segmentation_type_str == str(SegmentationType.SEGMENT_ANYTHING):
        return SegmentationType.SEGMENT_ANYTHING
    elif segmentation_type_str == str(SegmentationType.SEGMENT_ANYTHING2):
        return SegmentationType.SEGMENT_ANYTHING2
    elif segmentation_type_str == str(SegmentationType.NONE):
        return SegmentationType.NONE
    else:
        raise NotImplementedError(
            f'Segmentation type {segmentation_type_str} not supported')


def _flexiv_frame_prefix(context: LaunchContext) -> str:
    try:
        robot_sn = get_str_variable(context, 'robot_sn')
    except Exception:
        robot_sn = ''
    if not robot_sn:
        try:
            robot_launch_file_path = get_str_variable(context, 'robot_launch_file_path')
        except Exception:
            robot_launch_file_path = ''
        if robot_launch_file_path.endswith('flexiv_rizon_sim_driver.launch.py'):
            return ''
    if not robot_sn:
        raise ValueError(f'Robot SN not found for Flexiv Rizon, please set the robot_sn')
    return f'{robot_sn}_'


def _ur_frame_prefix(context: LaunchContext) -> str:
    try:
        tf_prefix = get_str_variable(context, 'tf_prefix')
    except Exception:
        # This is not required for UR robots, so we can return an empty string
        return ''
    return tf_prefix or ''


def compute_frame_prefix(context: LaunchContext) -> str:
    """
    Derive the TF frame prefix for the running robot.

    The prefix is resolved using the canonical launch arg for the given robot
    family (``robot_sn`` for Flexiv, ``tf_prefix`` for UR). The robot family
    is resolved from the ``robot_type`` launch arg via :func:`get_robot_type`.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        str: TF frame prefix (empty string if the family has none)

    Raises
    ------
        ValueError: If the resolved ``robot_type`` is not a supported
            ``RobotType``

    """
    robot_type = get_robot_type(context)
    if robot_type is RobotType.FLEXIV:
        return _flexiv_frame_prefix(context)
    if robot_type is RobotType.UR:
        return _ur_frame_prefix(context)
    raise ValueError(f'Robot type {robot_type} not supported')


_UR_ARM_JOINTS: Tuple[str, ...] = (
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
)


def compute_joint_names(context: LaunchContext) -> List[str]:
    """
    Return arm joint names for the running robot.

    Flexiv Rizon exposes ``joint1``..``joint7`` and UR exposes the 6 standard
    shoulder/elbow/wrist joints. The final names are prefixed with
    :func:`compute_frame_prefix`. The robot family is resolved from the
    ``robot_type`` launch arg via :func:`get_robot_type`.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        List[str]: Prefixed arm joint names

    Raises
    ------
        ValueError: If the resolved ``robot_type`` is not a supported
            ``RobotType``

    """
    robot_type = get_robot_type(context)
    prefix = compute_frame_prefix(context)
    if robot_type is RobotType.FLEXIV:
        return [f'{prefix}joint{i}' for i in range(1, 8)]
    if robot_type is RobotType.UR:
        return [f'{prefix}{j}' for j in _UR_ARM_JOINTS]
    raise ValueError(f'Robot type {robot_type} not supported')


def compute_gripper_action_name(gripper_type: str) -> str:
    """Return gripper GripperCommand action name based on gripper type."""
    if gripper_type == 'grav':
        return '/flexiv_gripper_node/gripper_action'
    return '/robotiq_gripper_controller/gripper_cmd'


def compute_gripper_positions(gripper_type: str) -> tuple:
    """
    Return (open_position, close_position) for the given gripper type.

    Grav gripper: 0.0 = fully closed, 0.1 = fully open.
    Robotiq gripper: 0.0 = fully open, 0.65 = fully closed.
    """
    if gripper_type == 'grav':
        return (0.09, 0.0)
    return (0.0, 0.65)


def compute_gripper_settle_time(gripper_type: str) -> float:
    """
    Return settle time (seconds) to wait after a gripper action completes.

    The Flexiv Grav action server returns before the gripper finishes moving,
    so we need an extra delay.  Robotiq's controller blocks until done.
    """
    if gripper_type == 'grav':
        return 2.0
    return 0.0
