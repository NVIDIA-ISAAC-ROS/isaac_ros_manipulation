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

from dataclasses import dataclass
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils.depth_routing import (
    select_foundation_pose_depth_routing,
)
from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    compute_frame_prefix, compute_gripper_action_name, compute_gripper_positions,
    compute_gripper_settle_time, compute_joint_names,
    constants, get_bool_variable, get_camera_type, get_depth_type, get_dnn_stereo_depth_resolution,
    get_float_list_variable, get_float_variable, get_object_attachment_type,
    get_object_detection_namespace,
    get_object_detection_type, get_object_selection_type, get_pose_estimation_type,
    get_robot_type, get_segmentation_type, get_str_variable, get_workflow_type,
    normalize_pose_quaternion
)
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
    WorkflowType
)
from launch import LaunchContext
from launch.substitutions import LaunchConfiguration, SubstitutionFailure
from sensor_msgs.msg import JointState
import yaml


SERVER_BASED_PERCEPTION_WORKFLOWS = (
    WorkflowType.PICK_AND_PLACE,
    WorkflowType.DISPLAYPORT_CABLE_INSERTION,
)


# Distinguishes an undeclared impedance service name (derive it) from an explicit
# '' (skip the impedance call).
_DERIVE_IMPEDANCE_SERVICE = '__derive_from_robot_sn__'


def _get_optional_str(context: LaunchContext, name: str, default: str = '') -> str:
    """
    Resolve a launch argument that may or may not be declared.

    Thin wrapper around :func:`get_str_variable` that swallows the exception
    raised when ``name`` is not declared in the current launch file. Used by
    :class:`DriverConfig` for fields that are only relevant to a subset of
    vendor launch files (e.g. URDF / SRDF paths unused by vendors that rely
    on third-party description packages).

    Args
    ----
        context (LaunchContext): Active launch context.
        name (str): Launch argument name to resolve.

    Returns
    -------
        str: The declared value, or ``default`` if the argument is not declared.

    """
    try:
        return get_str_variable(context, name)
    except Exception:
        return default


def _get_optional_float(context: LaunchContext, name: str, default: float) -> float:
    """
    Resolve an optional float launch argument, returning ``default`` if undeclared.

    Float counterpart of :func:`_get_optional_str`, used for tuning knobs that are
    only defined in a subset of workflow param files.
    """
    try:
        value = get_str_variable(context, name).strip()
        if value == '':
            return default
        return float(value)
    except (SubstitutionFailure, ValueError):
        return default


def _get_optional_bool(context: LaunchContext, name: str, default: bool) -> bool:
    """Resolve an optional bool launch argument, returning ``default`` if undeclared."""
    try:
        return get_bool_variable(context, name)
    except SubstitutionFailure:
        return default


def _get_optional_float_list(context: LaunchContext, name: str,
                             default: List[float]) -> List[float]:
    """Resolve an optional float-list launch argument, returning ``default`` if undeclared."""
    try:
        return get_float_list_variable(context, name)
    except SubstitutionFailure:
        return list(default)


def _reject_relative_path(name: str, value: str) -> None:
    """
    Fail fast when a path that is resolved against the process CWD is relative.

    Only use this for fields that are handed to a consumer which does no
    package-relative resolution of its own. Fields consumed by
    ``MoveItConfigsBuilder`` (``srdf_path``, ``joint_limits_file_path``,
    ``kinematics_file_path``, ``moveit_controllers_file_path``) resolve against
    the robot-description package share directory and must not be passed here.
    """
    if value and not os.path.isabs(value):
        raise ValueError(
            f'{name} must be an absolute path, got {value!r}. This value is resolved '
            'against the working directory of the launching process, so a relative path '
            'happens to work from a source checkout but fails from an installed '
            'workspace, a Debian package or a container. Write it as '
            '"$(ros2 pkg prefix --share <package>)/...".')


def load_yaml_params(
    params_file_path: str,
    package_name: str = 'isaac_ros_manipulation_bringup'
) -> Dict:
    """
    Load the YAML parameters for manipulator workflow config.

    Args
    ----
        params_file_path (str): The file path for the manipulator workflow config
        package_name (str): The name of the package to load the parameters from

    Returns
    -------
        Dict: Return dict with values in the file.

    Raises
    ------
        FileNotFoundError: If the parameter file is not found.

    """
    # If path is just a file name, then we need to get the full path
    if not os.path.exists(params_file_path):
        params_file_path = os.path.join(
            get_package_share_directory(package_name),
            'params',
            params_file_path
        )

    if not os.path.exists(params_file_path):
        raise FileNotFoundError(f'Parameter file not found: {params_file_path}')

    # Load the YAML file as text
    with open(params_file_path, 'r') as file:
        yaml_content = file.read()

    # Simple function to replace command expressions with their output
    def process_commands(text):
        pattern = r'\$\(([^)]+)\)'

        def replace_cmd(match):
            cmd = match.group(1)
            return subprocess.getoutput(cmd)

        return re.sub(pattern, replace_cmd, text)

    def process_commands_env_variables(text):
        """
        Replace environment variables in the text.

        Supports both ${VAR} and $VAR syntax.
        Variables that are not set will be left as-is.
        """
        # First, handle ${VAR} format (easier to match unambiguously)
        def replace_braced_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        text = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace_braced_var, text)

        # Then handle $VAR format (but not $(commands) which have already been processed)
        # Match $VAR where VAR is alphanumeric/underscore and NOT followed by (
        def replace_unbraced_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        text = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)(?![(\{])', replace_unbraced_var, text)

        return text

    # Process the YAML content
    processed_yaml = process_commands(yaml_content)

    # Make sure to replace all the env variables in the processed YAML
    processed_yaml = process_commands_env_variables(processed_yaml)

    # Parse the processed YAML
    return yaml.safe_load(processed_yaml)


class Config:
    """Config that tracks Configuration variables."""

    use_sim_time: bool
    headless: bool

    def __init__(self, context: LaunchContext):
        self.use_sim_time = get_bool_variable(context, 'use_sim_time')
        self.headless = _get_optional_bool(context, 'headless', False)


# Per-gripper collision link tables, used to disable links during the retract/approach
# phases of pick-and-place. Co-located with DriverConfig (rather than in any vendor
# driver_utils package) so it can be exposed as a property on DriverConfig without
# introducing an import cycle with the per-vendor driver_utils packages.
_GRIPPER_COLLISION_LINKS: Dict[GripperType, List[str]] = {
    GripperType.ROBOTIQ_2F_140: [
        'left_outer_knuckle',
        'left_inner_knuckle',
        'left_outer_finger',
        'left_inner_finger',
        'left_inner_finger_pad',
        'right_outer_knuckle',
        'right_inner_knuckle',
        'right_outer_finger',
        'right_inner_finger',
        'right_inner_finger_pad',
    ],
    GripperType.ROBOTIQ_2F_85: [
        'robotiq_85_base_link',
        'robotiq_85_left_finger_link',
        'robotiq_85_left_finger_tip_link',
        'robotiq_85_left_inner_knuckle_link',
        'robotiq_85_left_knuckle_link',
        'robotiq_85_right_finger_link',
        'robotiq_85_right_finger_tip_link',
        'robotiq_85_right_inner_knuckle_link',
        'robotiq_85_right_knuckle_link',
    ],
    GripperType.GRAV: [
        'grav_base_link',
        'left_outer_bar',
        'left_inner_bar',
        'left_finger_mount',
        'left_finger_tip',
        'right_outer_bar',
        'right_inner_bar',
        'right_finger_mount',
        'right_finger_tip',
    ],
}


class DriverConfig(Config):
    """
    Config that tracks all variables needed to run drivers for the robot/Isaac Sim.

    Owns the vendor-agnostic robot identity (robot_type, TF frames, joint names)
    and the gripper-derived fields shared by every workflow (action name, open/close
    positions, settle time, collision link list). Per-vendor driver packages extend
    this class to add hardware-specific fields (e.g. ``robot_ip``, ``ur_type``).
    """

    robot_type: RobotType
    gripper_type: GripperType
    urdf_path: str
    srdf_path: str
    joint_limits_file_path: str
    kinematics_file_path: str
    moveit_controllers_file_path: str
    ros2_controllers_file_path: str
    frame_prefix: str
    base_frame: str
    gripper_frame: str
    grasp_frame: str
    insertion_frame: str
    joint_names: List[str]
    # Serial number of the physical robot. Currently only populated for Flexiv
    # (via the ``robot_sn`` launch arg); empty string for every other family.
    robot_sn: str
    gripper_action_name: str
    gripper_open_position: float
    gripper_close_position: float
    gripper_settle_time_sec: float
    gripper_max_effort: float
    gripper_collision_links: List[str]

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.robot_type = get_robot_type(context)
        self.gripper_type = GripperType.get_gripper_type(
            get_str_variable(context, 'gripper_type'))

        # Robot-description and MoveIt path fields are optional at the
        # ``DriverConfig`` layer: some vendor integrations (e.g. the real
        # Flexiv Rizon driver) pull URDF / SRDF / controller configs from
        # third-party packages and never consume these paths. Wrap each lookup
        # so a missing launch arg resolves to '' instead of raising, and let
        # the vendor subclass or utility tell us whether the path is actually
        # required for its code path.
        self.urdf_path = _get_optional_str(context, 'urdf_path')
        self.srdf_path = _get_optional_str(context, 'srdf_path')
        self.joint_limits_file_path = _get_optional_str(
            context, 'joint_limits_file_path')
        self.kinematics_file_path = _get_optional_str(
            context, 'kinematics_file_path')
        self.moveit_controllers_file_path = _get_optional_str(
            context, 'moveit_controllers_file_path')
        self.ros2_controllers_file_path = _get_optional_str(
            context, 'ros2_controllers_file_path')
        # Unlike the MoveIt fields above, which MoveItConfigsBuilder resolves against
        # the robot-description package share directory, this one is handed straight to
        # launch_ros ParameterFile and so is resolved against the process CWD.
        _reject_relative_path(
            'ros2_controllers_file_path', self.ros2_controllers_file_path)

        self.frame_prefix = compute_frame_prefix(context)
        self.base_frame = f'{self.frame_prefix}base_link'
        self.gripper_frame = f'{self.frame_prefix}gripper_frame'
        self.grasp_frame = f'{self.frame_prefix}grasp_frame'
        self.insertion_frame = f'{self.frame_prefix}insertion_frame'
        self.joint_names = compute_joint_names(context)

        # ``robot_sn`` is a Flexiv-specific launch arg. Keep the resolution here
        # (rather than pushing it into ``FlexivRizonDriverConfig``) so every
        # DriverConfig consumer can reach for ``driver_config.robot_sn`` without
        # branching on robot family.
        self.robot_sn = _get_optional_str(context, 'robot_sn')

        gripper_type_str = self.gripper_type.value
        self.gripper_action_name = compute_gripper_action_name(gripper_type_str)
        default_open_position, default_close_position = compute_gripper_positions(
            gripper_type_str)
        self.gripper_open_position = _get_optional_float(
            context, 'gripper_open_position', default_open_position)
        self.gripper_close_position = _get_optional_float(
            context, 'gripper_close_position', default_close_position)
        self.gripper_settle_time_sec = compute_gripper_settle_time(gripper_type_str)
        # Grasp force (GripperCommand max_effort) used when closing the gripper.
        # Optional launch arg; defaults to 10.0 to preserve prior behavior. On the
        # real Flexiv Grav gripper this maps to grasp force (higher = tighter);
        # the sim gripper driver ignores it.
        self.gripper_max_effort = _get_optional_float(context, 'gripper_max_effort', 10.0)
        try:
            self.gripper_collision_links = _GRIPPER_COLLISION_LINKS[self.gripper_type]
        except KeyError as exc:
            raise NotImplementedError(
                f'Gripper type {self.gripper_type} not supported') from exc


def build_driver_config(context: LaunchContext) -> DriverConfig:
    """
    Build the vendor-specific :class:`DriverConfig` for the running robot.

    Dispatches on the ``robot_type`` launch arg to the correct vendor subclass
    (currently ``UrRobotiqDriverConfig`` or ``FlexivRizonDriverConfig``). The
    vendor imports are performed lazily inside this function to avoid import
    cycles: the vendor driver_utils packages depend on this package for the
    ``DriverConfig`` base class, so this package cannot import them at module
    load time.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        DriverConfig: Vendor-specific driver config instance

    Raises
    ------
        NotImplementedError: If the ``robot_type`` launch arg value is not a
            supported ``RobotType``

    """
    robot_type = get_robot_type(context)
    if robot_type is RobotType.UR:
        # UR support is kept for internal use only and is not part of the release.
        try:
            from isaac_ros_manipulation_ur_driver_utils.config import UrRobotiqDriverConfig
        except ModuleNotFoundError as exc:
            if exc.name not in ('isaac_ros_manipulation_ur_driver_utils',
                                'isaac_ros_manipulation_ur_driver_utils.config'):
                raise
            raise NotImplementedError(
                f'Robot type {robot_type.value} is not part of this release; use '
                f'{RobotType.FLEXIV.value}') from exc
        return UrRobotiqDriverConfig(context)
    if robot_type is RobotType.FLEXIV:
        from isaac_ros_manipulation_flexiv_driver_utils.config import (
            FlexivRizonDriverConfig,
        )
        return FlexivRizonDriverConfig(context)
    raise NotImplementedError(f'Robot type {robot_type} not supported')


class CameraConfig(Config):
    """Config that tracks all variables needed to run cameras for the robot/Isaac Sim."""

    num_cameras: str
    depth_type: DepthType
    camera_type: CameraType

    # These are overridden by the child classes
    color_camera_topic_name: str = 'N/A'
    depth_camera_topic_name: str = 'N/A'
    color_camera_info_topic_name: str = 'N/A'
    depth_camera_info_topic_name: str = 'N/A'
    realsense_depth_camera_topic_name: str = 'N/A'
    realsense_depth_camera_info_topic_name: str = 'N/A'
    realsense_depth_image_width: str = 'N/A'
    realsense_depth_image_height: str = 'N/A'
    rgb_image_width: str = 'N/A'
    rgb_image_height: str = 'N/A'
    depth_image_width: str = 'N/A'
    depth_image_height: str = 'N/A'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.num_cameras = _get_optional_str(context, 'num_cameras', '1')
        self.camera_type = get_camera_type(
            _get_optional_str(context, 'camera_type', str(CameraType.REALSENSE)))
        self.enable_dnn_depth_in_realsense = _get_optional_bool(
            context, 'enable_dnn_depth_in_realsense', False)
        if self.camera_type == CameraType.ISAAC_SIM:
            self.depth_type = DepthType.ISAAC_SIM
        elif self.camera_type == CameraType.REALSENSE:
            if self.enable_dnn_depth_in_realsense:
                self.depth_type = get_depth_type(get_str_variable(context, 'depth_type'))
            else:
                self.depth_type = DepthType.REALSENSE
        else:
            raise ValueError(f'Camera type {self.camera_type} not supported')


class IsaacSimCameraConfig(CameraConfig):
    """Contains image topic names coming from Isaac Sim."""

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = CameraType.ISAAC_SIM
        self.color_camera_topic_name = '/front_stereo_camera/left/image_raw'
        self.depth_camera_topic_name = '/front_stereo_camera/depth/ground_truth'
        self.color_camera_info_topic_name = '/front_stereo_camera/left/camera_info'
        self.depth_camera_info_topic_name = '/front_stereo_camera/left/camera_info'
        self.rgb_image_width = str(constants.HAWK_IMAGE_WIDTH)
        self.rgb_image_height = str(constants.HAWK_IMAGE_HEIGHT)
        self.depth_image_width = str(constants.HAWK_IMAGE_WIDTH)
        self.depth_image_height = str(constants.HAWK_IMAGE_HEIGHT)
        self.num_cameras = '1'
        self.use_sim_time = True


class RealsenseCameraConfig(CameraConfig):
    """Contains image topic names coming from RealSense camera."""

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = CameraType.REALSENSE
        self.color_camera_topic_name = '/camera_1/color/image_raw'
        self.realsense_depth_camera_topic_name = '/camera_1/aligned_depth_to_color/image_raw'
        self.color_camera_info_topic_name = '/camera_1/color/camera_info'
        self.realsense_depth_camera_info_topic_name = \
            '/camera_1/aligned_depth_to_color/camera_info'
        self.depth_camera_topic_name = self.realsense_depth_camera_topic_name
        self.depth_camera_info_topic_name = self.realsense_depth_camera_info_topic_name
        self.rgb_image_width = str(constants.REALSENSE_IMAGE_WIDTH)
        self.rgb_image_height = str(constants.REALSENSE_IMAGE_HEIGHT)
        self.realsense_depth_image_width = str(constants.REALSENSE_IMAGE_WIDTH)
        self.realsense_depth_image_height = str(constants.REALSENSE_IMAGE_HEIGHT)
        self.depth_image_width = str(constants.REALSENSE_IMAGE_WIDTH)
        self.depth_image_height = str(constants.REALSENSE_IMAGE_HEIGHT)

        if self.enable_dnn_depth_in_realsense:
            self.ess_depth_camera_topic_name = '/camera_1/depth_image'
            self.ess_depth_camera_info_topic_name = '/camera_1/rgb/camera_info'

            # Also align the images from realsense to be this resized one.
            self.depth_camera_topic_name = '/camera_1/depth_image'
            self.depth_camera_info_topic_name = '/camera_1/rgb/camera_info'

            depth_image_width, depth_image_height = \
                get_dnn_stereo_depth_resolution(self.depth_type)
            self.depth_image_width = depth_image_width
            self.depth_image_height = depth_image_height


def get_camera_config(context: LaunchContext, camera_type: CameraType) -> CameraConfig:
    """
    Get camera config.

    Args
    ----
        camera_type (CameraType): Camera type

    Returns
    -------
        CameraConfig: Camera config

    """
    if camera_type == CameraType.ISAAC_SIM:
        return IsaacSimCameraConfig(context)
    elif camera_type == CameraType.REALSENSE:
        return RealsenseCameraConfig(context)
    else:
        raise NotImplementedError(f'Camera type is not supported {camera_type}')


class SensorConfig(Config):
    """Config tracks variables needed to perform workflows in Isaac Sim and on the real robot."""

    robot_type: RobotType
    camera_type: CameraType
    gripper_type: GripperType
    num_cameras: str
    setup: str
    depth_type: str
    depth_engine_file_path: str
    workflow_type: WorkflowType
    enable_dnn_depth_in_realsense: bool

    _DEPTH_ENGINE_KEY = {
        str(DepthType.ESS_FULL): 'ess_full_engine_file_path',
        str(DepthType.ESS_LIGHT): 'ess_light_engine_file_path',
        str(DepthType.FOUNDATION_STEREO_LOW_RES): 'foundationstereo_low_res_engine_file_path',
        str(DepthType.FOUNDATION_STEREO_HIGH_RES): 'foundationstereo_high_res_engine_file_path',
    }

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.robot_type = get_robot_type(context)
        self.gripper_type = get_str_variable(context, 'gripper_type')
        self.camera_type = get_camera_type(
            _get_optional_str(context, 'camera_type', str(CameraType.REALSENSE)))
        self.num_cameras = _get_optional_str(context, 'num_cameras', '1')
        self.setup = get_str_variable(context, 'setup')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.depth_type = get_str_variable(context, 'depth_type')
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')
        engine_key = self._DEPTH_ENGINE_KEY.get(self.depth_type, '')
        self.depth_engine_file_path = (
            get_str_variable(context, engine_key) if engine_key else '')

        needs_depth_engine = engine_key and (
            self.camera_type != CameraType.REALSENSE
            or self.enable_dnn_depth_in_realsense
        )
        if needs_depth_engine and not os.path.isfile(self.depth_engine_file_path):
            raise FileNotFoundError(
                f"depth_type is '{self.depth_type}' but the engine file "
                f"'{engine_key}' does not exist: "
                f"'{self.depth_engine_file_path}'. "
                f'Run the model setup script or verify the path in '
                f'your workflow configuration YAML.'
            )

        if self.use_sim_time and self.gripper_type == 'robotiq_2f_85':
            raise ValueError(f'Gripper type {self.gripper_type} not supported for Isaac sim')


class PoseEstimationConfig(Config):
    """Config that tracks all variables needed to perform pose estimation."""

    pose_estimation_type: PoseEstimationType
    object_detection_type: ObjectDetectionType
    segmentation_type: SegmentationType

    def __init__(self, context: LaunchContext):
        super().__init__(context)

        self.pose_estimation_type = get_pose_estimation_type(
            get_str_variable(context, 'pose_estimation_type'))
        self.object_detection_type = get_object_detection_type(
            get_str_variable(context, 'object_detection_type'))
        self.segmentation_type = get_segmentation_type(
            get_str_variable(context, 'segmentation_type'))

    def update_server_topic_names(self, camera_config: CameraConfig):
        """
        Update the topic names for the servers.

        Args
        ----
            camera_config (CameraConfig): Camera config

        Raises
        ------
            NotImplementedError: Pose estimation config does not implement this method

        """
        raise NotImplementedError('Pose estimation config does not implement this method')


class ObjectDetectionConfig(Config):
    """Config that tracks all variables needed to perform object detection."""

    detections_2d_array_output_topic: str
    is_sam_detection_backend: bool
    object_detection_type: ObjectDetectionType
    name_space_prefix: str
    workflow_type: WorkflowType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.object_detection_type = get_object_detection_type(
            get_str_variable(context, 'object_detection_type'))
        self.name_space_prefix = get_object_detection_namespace(self.object_detection_type)
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))

        self.is_sam_detection_backend = (
            self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING or
            self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2
        )

        if self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'


class ObjectSelectionConfig(Config):
    """Config that tracks all variables needed to perform object selection."""

    selection_policy: ObjectSelectionType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.selection_policy = get_object_selection_type(
            get_str_variable(context, 'selection_policy'))


class SegmentAnythingConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Segment Anything object detection."""

    # This is used when action servers are not being used; manual inputs to this topic are needed.
    # This will be picked up by the SegmentAnythingTrigger node.
    segment_anything_input_points_topic: str
    segment_anything_input_detections_topic: str
    sam_model_repository_paths: LaunchConfiguration
    segment_anything_enable_debug_output: str

    # Topic names for SAM (Segment Anything) server
    sam_in_img_topic_name: str
    sam_out_img_topic_name: str

    sam_in_camera_info_topic_name: str
    sam_out_camera_info_topic_name: str

    # The servers will publish detections 2d array on this topic.
    # When servers are on, SegmentAnythingTrigger node is not operational, the servers will output
    # time synced messages onto image, camera info and detection 2d queues which are injested by
    # SegmentAnythingEncoder node.
    sam_out_detections_topic_name: str

    # Output of SAM, the detections 2d array of the mask generation and the mask piped into the
    # servers for caching purposes. These messages are send via action responses to ObjectInfo
    # Server.
    sam_in_detections_topic_name: str
    sam_in_mask_topic_name: str

    # True when using object following workflow and the object detection backend is SAM.
    is_point_triggered: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.segment_anything_input_points_topic = get_str_variable(
            context, 'segment_anything_input_points_topic')
        self.segment_anything_input_detections_topic = get_str_variable(
            context, 'segment_anything_input_detections_topic')
        self.sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
        self.segment_anything_enable_debug_output = get_str_variable(
            context, 'segment_anything_enable_debug_output')

        # Topic names for SAM (Segment Anything) server
        self.sam_in_img_topic_name = 'UPDATED_LATER'
        self.sam_out_img_topic_name = '/segment_anything_server/image'
        self.sam_in_camera_info_topic_name = 'UPDATED_LATER'
        self.sam_out_camera_info_topic_name = '/segment_anything_server/camera_info'
        self.sam_out_detections_topic_name = \
            '/segment_anything_server/detections_initial_guess'
        self.sam_in_mask_topic_name = '/segment_anything/binary_segmentation_mask'
        self.sam_in_detections_topic_name = '/segment_anything/detections'

        # Main launch file input topic names
        self.sam_main_launch_file_image_topic = 'UPDATED_LATER'
        self.sam_main_launch_file_camera_info_topic = 'UPDATED_LATER'

        self.is_point_triggered = (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            self.is_sam_detection_backend
        )

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.sam_in_img_topic_name = camera_config.color_camera_topic_name
        self.sam_in_camera_info_topic_name = camera_config.color_camera_info_topic_name

        self.sam_main_launch_file_image_topic = camera_config.color_camera_topic_name
        self.sam_main_launch_file_camera_info_topic = camera_config.color_camera_info_topic_name
        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            self.sam_main_launch_file_image_topic = self.sam_out_img_topic_name
            self.sam_main_launch_file_camera_info_topic = self.sam_out_camera_info_topic_name

        # If using object following workflow and using an object detection model as the detection
        # backend, pass the output 2D detections to the SAM graph.
        if (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            not self.is_sam_detection_backend
        ):
            self.sam_out_detections_topic_name = \
                f'/{self.detections_2d_array_output_topic}'
            self.sam_main_launch_file_image_topic = f'/{self.name_space_prefix}/image_dropped'
            self.sam_main_launch_file_camera_info_topic = \
                f'/{self.name_space_prefix}/camera_info_dropped'


class SegmentAnything2Config(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Segment Anything 2 object detection."""

    # This is used when action servers are not being used; manual inputs to this topic are needed.
    # This will be picked up by the SegmentAnything2Trigger node.
    segment_anything_input_points_topic: str
    sam_model_repository_paths: LaunchConfiguration

    # Topic names for SAM2 (Segment Anything 2) server
    sam2_in_img_topic_name: str
    sam2_out_img_topic_name: str

    sam2_in_camera_info_topic_name: str
    sam2_out_camera_info_topic_name: str

    # Output of SAM2, the detections 2d array of the mask generation and the mask piped into the
    # servers for caching purposes. These messages are send via action responses to ObjectInfo
    # Server.
    sam2_in_detections_topic_name: str
    sam2_in_mask_topic_name: str

    # True when using object following workflow and the object detection backend is SAM.
    is_point_triggered: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.segment_anything_input_points_topic = get_str_variable(
            context, 'segment_anything_input_points_topic')
        self.sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')

        # Topic names for SAM2 (Segment Anything 2) server
        self.sam2_in_img_topic_name = 'UPDATED_LATER'
        self.sam2_out_img_topic_name = '/segment_anything_server/image'
        self.sam2_in_camera_info_topic_name = 'UPDATED_LATER'
        self.sam2_out_camera_info_topic_name = '/segment_anything_server/camera_info'
        self.sam2_in_mask_topic_name = '/segment_anything2/binary_segmentation_mask'
        self.sam2_in_detections_topic_name = '/segment_anything2/detections'

        # Main launch file input topic names
        self.sam2_main_launch_file_image_topic = 'UPDATED_LATER'
        self.sam2_main_launch_file_camera_info_topic = 'UPDATED_LATER'

        self.is_point_triggered = (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            self.is_sam_detection_backend
        )

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.sam2_in_img_topic_name = camera_config.color_camera_topic_name
        self.sam2_in_camera_info_topic_name = camera_config.color_camera_info_topic_name

        self.sam2_main_launch_file_image_topic = camera_config.color_camera_topic_name
        self.sam2_main_launch_file_camera_info_topic = camera_config.color_camera_info_topic_name
        if self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.sam2_main_launch_file_image_topic = self.sam2_out_img_topic_name
            self.sam2_main_launch_file_camera_info_topic = self.sam2_out_camera_info_topic_name


class DopeConfig(PoseEstimationConfig):
    """Config that tracks all variables needed to perform DOPE detection and pose estimation."""

    dope_model_file_path: str
    dope_engine_file_path: str
    input_fps: str
    dropped_fps: str
    dope_input_qos: str
    dope_rotation_y_axis: str = '-90.0'
    dope_enable_tf_publishing: str = 'true'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.dope_engine_file_path = get_str_variable(context, 'dope_engine_file_path')
        self.dope_model_file_path = get_str_variable(context, 'dope_model_file_path')
        self.input_fps = get_str_variable(context, 'pose_estimation_input_fps')
        self.dropped_fps = get_str_variable(context, 'pose_estimation_dropped_fps')
        self.dope_input_qos = get_str_variable(context, 'pose_estimation_input_qos')


class RtDetrConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform RT-DETR object detection."""

    rt_detr_confidence_threshold: str
    rtdetr_engine_file_path: str
    object_class_id: str

    # Object Detection Server for RT-DETR
    rtdetr_in_img_topic_name: str
    rtdetr_out_img_topic_name: str

    rtdetr_in_camera_info_topic_name: str
    rtdetr_out_camera_info_topic_name: str

    # Not sure why we need this:
    rtdetr_out_detections_topic_name: str

    # Output of RT-DETR
    rtdetr_in_detections_topic_name: str

    # Input into RTDEtr launcg file.
    rtdetr_camera_input: str
    rtdetr_camera_info_input: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.rt_detr_confidence_threshold = get_str_variable(
            context, 'rt_detr_confidence_threshold'
        )
        self.rtdetr_engine_file_path = get_str_variable(
            context, 'rtdetr_engine_file_path'
        )
        self.object_class_id = get_str_variable(context, 'object_class_id')

        self.rtdetr_in_img_topic_name = 'UPDATED_LATER'
        self.rtdetr_out_img_topic_name = '/object_detection_server/image_rect'
        # Figure this out, pipe into rtdetr as well not hardcoded.
        self.rtdetr_in_detections_topic_name = '/detections'
        self.rtdetr_out_detections_topic_name = '/object_detection_server/detections_output'
        self.rtdetr_in_camera_info_topic_name = 'UPDATED_LATER'
        self.rtdetr_out_camera_info_topic_name = '/object_detection_server/camera_info'

        # Input into the main rtdetr.launch.py file
        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'

        self.rtdetr_camera_input = 'UPDATED_LATER'
        self.rtdetr_camera_info_input = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.rtdetr_in_img_topic_name = camera_config.color_camera_topic_name
        self.rtdetr_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            self.rtdetr_camera_input = self.rtdetr_out_img_topic_name
            self.rtdetr_camera_info_input = self.rtdetr_out_camera_info_topic_name
        else:
            self.rtdetr_camera_input = camera_config.color_camera_topic_name
            self.rtdetr_camera_info_input = camera_config.color_camera_info_topic_name


class GroundingDinoConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Grounding DINO object detection."""

    grounding_dino_confidence_threshold: str
    grounding_dino_engine_file_path: str
    grounding_dino_model_file_path: str
    grounding_dino_default_prompt: str
    grounding_dino_in_camera_info_topic_name: str
    grounding_dino_in_detections_topic_name: str
    grounding_dino_in_img_topic_name: str
    grounding_dino_out_camera_info_topic_name: str
    grounding_dino_out_detections_topic_name: str
    grounding_dino_out_img_topic_name: str
    network_image_height: str
    network_image_width: str

    # Input into grounding dino launch file
    grounding_dino_camera_input: str
    grounding_dino_camera_info_input: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.grounding_dino_confidence_threshold = get_str_variable(
            context, 'grounding_dino_confidence_threshold'
        )
        self.grounding_dino_engine_file_path = get_str_variable(
            context, 'grounding_dino_engine_file_path'
        )
        self.grounding_dino_model_file_path = get_str_variable(
            context, 'grounding_dino_model_file_path'
        )
        self.grounding_dino_default_prompt = get_str_variable(
            context, 'grounding_dino_default_prompt'
        )
        self.network_image_width = get_str_variable(
            context, 'grounding_dino_network_image_width'
        )
        self.network_image_height = get_str_variable(
            context, 'grounding_dino_network_image_height'
        )

        # Inputs into the object detection server.
        self.grounding_dino_in_camera_info_topic_name = 'UPDATED_LATER'
        self.grounding_dino_in_img_topic_name = 'UPDATED_LATER'
        # Inputs into the main grounding_dino.launch.py file
        self.grounding_dino_in_detections_topic_name = '/detections'
        self.grounding_dino_out_camera_info_topic_name = '/object_detection_server/camera_info'
        self.grounding_dino_out_detections_topic_name = \
            '/object_detection_server/detections_output'
        self.grounding_dino_out_img_topic_name = '/object_detection_server/image_rect'

        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'

        self.grounding_dino_camera_input = 'UPDATED_LATER'
        self.grounding_dino_camera_info_input = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.grounding_dino_in_img_topic_name = camera_config.color_camera_topic_name
        self.grounding_dino_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            self.grounding_dino_camera_input = self.grounding_dino_out_img_topic_name
            self.grounding_dino_camera_info_input = self.grounding_dino_out_camera_info_topic_name
        else:
            self.grounding_dino_camera_input = camera_config.color_camera_topic_name
            self.grounding_dino_camera_info_input = camera_config.color_camera_info_topic_name


class FoundationPoseConfig(PoseEstimationConfig):
    """Config that tracks all variables needed to perform FoundationPose pose estimation."""

    foundation_pose_mesh_file_path: str
    foundation_pose_texture_path: str
    foundation_pose_refine_engine_file_path: str
    foundation_pose_score_engine_file_path: str
    foundation_pose_segmentation_mask_topic: str
    object_detection_config: Union[
        GroundingDinoConfig, RtDetrConfig, SegmentAnythingConfig, SegmentAnything2Config]
    segmentation_config: Union[SegmentAnythingConfig, SegmentAnything2Config, None]
    input_fps: str
    dropped_fps: str
    input_qos: str
    foundation_pose_detection2_d_topic: str = 'detections_output'
    workflow_type: WorkflowType
    discard_messages: str = 'True'
    discard_msg_older_than_ms: str = '1000'

    # Topic names to push perception input into foundation pose launch file
    foundation_pose_rgb_image_topic: str
    foundation_pose_depth_image_topic: str
    foundation_pose_rgb_camera_info: str

    # Topic names for foundation pose servers.
    fp_in_img_topic_name: str  # Input image topic name
    fp_out_img_topic_name: str  # Ouput from servers routed into perception pipeline

    fp_in_camera_info_topic_name: str  # Input camera info topic name
    fp_out_camera_info_topic_name: str  # Output topic name routed into perception pipeline

    fp_in_depth_topic_name: str  # Input depth topic name
    fp_in_depth_image_width: str
    fp_in_depth_image_height: str
    fp_out_depth_topic_name: str  # Output depth topic name routed into perception pipeline

    fp_out_detections_topic_name: str  # Output detections topic name routed into perception
    # Output segmented mask topic name routed into perception, only when SEGMENT ANYTHING
    # backend for segmentaiton is used. For RTDETR seg mask is generated per pose estimation call.
    fp_out_segmented_mask_topic_name: str
    # Output pose estimate topic name, just outputs the computed pose in this topic
    # TODO(kchahal): Do we need this ?
    fp_out_pose_estimate_topic_name: str

    enable_dnn_depth_in_realsense: bool
    depth_camera_info_for_ess: str
    depth_type: DepthType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.foundation_pose_mesh_file_path = get_str_variable(
            context, 'foundation_pose_mesh_file_path')
        self.foundation_pose_texture_path = get_str_variable(
            context, 'foundation_pose_texture_path')
        self.foundation_pose_score_engine_file_path = get_str_variable(
            context, 'foundation_pose_score_engine_file_path')
        self.foundation_pose_refine_engine_file_path = get_str_variable(
            context, 'foundation_pose_refine_engine_file_path'
        )
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.depth_type = get_depth_type(get_str_variable(context, 'depth_type'))

        if self.workflow_type == WorkflowType.OBJECT_FOLLOWING:
            # FoundationPose object following needs timestamp-matched color, depth, and
            # segmentation. Keep this pose-estimation path on native RealSense depth; the
            # camera, nvblox, and cuMotion paths may still use DNN depth.
            self.enable_dnn_depth_in_realsense = False

        self.pose_estimation_type = get_pose_estimation_type(
            get_str_variable(context, 'pose_estimation_type'))
        self.dropped_fps = get_str_variable(context, 'pose_estimation_dropped_fps')
        self.input_fps = get_str_variable(context, 'pose_estimation_input_fps')
        # In Isaac Sim, we use DEFAULT QoS as we can apply backpressure to the input topic in
        # and make sim tick slower so that it can injest all the data. For real robot, we use
        # SENSOR_DATA so that we do not slow the pipeline down in the pursuit of no message drops.
        self.input_qos = get_str_variable(context, 'pose_estimation_input_qos')

        if self.object_detection_type == ObjectDetectionType.GROUNDING_DINO:
            self.object_detection_config = GroundingDinoConfig(context)
            self.foundation_pose_segmentation_mask_topic = 'segmentation'
        elif self.object_detection_type == ObjectDetectionType.RTDETR:
            self.object_detection_config = RtDetrConfig(context)
            self.foundation_pose_segmentation_mask_topic = 'segmentation'
        elif self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING:
            self.object_detection_config = SegmentAnythingConfig(context)
            self.foundation_pose_segmentation_mask_topic = \
                '/segment_anything/binary_segmentation_mask'
        elif self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2:
            self.object_detection_config = SegmentAnything2Config(context)
            self.foundation_pose_segmentation_mask_topic = \
                '/segment_anything2/binary_segmentation_mask'
        else:
            raise NotImplementedError(
                f'Object detection type {self.object_detection_type} not supported')

        self.segmentation_config = None
        if self.segmentation_type == SegmentationType.SEGMENT_ANYTHING:
            self.segmentation_config = SegmentAnythingConfig(context)
        elif self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
            self.segmentation_config = SegmentAnything2Config(context)

        self.foundation_pose_detection2_d_topic = 'detections_output'

        # Topic names for foundation pose servers.
        self.fp_in_img_topic_name = 'UPDATED_LATER'
        self.fp_out_img_topic_name = '/foundation_pose_server/image'
        self.fp_in_camera_info_topic_name = 'UPDATED_LATER'  # Verify this for SAM case
        self.fp_out_camera_info_topic_name = '/foundation_pose_server/camera_info'
        self.fp_in_depth_topic_name = 'UPDATED_LATER'
        self.fp_in_depth_image_width = 'UPDATED_LATER'
        self.fp_in_depth_image_height = 'UPDATED_LATER'
        self.fp_out_depth_topic_name = '/foundation_pose_server/depth'
        # This is correct for foundation pose.
        self.fp_in_pose_estimate_topic_name = '/pose_estimation/output'
        self.fp_out_segmented_mask_topic_name = '/foundation_pose_server/segmented_mask'
        self.fp_out_pose_estimate_topic_name = '/foundation_pose_server/pose_estimation/output'
        self.fp_out_detections_topic_name = '/foundation_pose_server/bbox'

        self.discard_old_messages = 'True'
        if (
            self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS
            or self.use_sim_time
        ):
            # Do not discard messages in FoundationPoseSynchronizer as we only send messages per
            # client request, we do not want to drop that data.
            self.discard_old_messages = 'False'

        if self.workflow_type in SERVER_BASED_PERCEPTION_WORKFLOWS:
            if (
                self.segmentation_type == SegmentationType.SEGMENT_ANYTHING or
                self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2
            ):
                self.segmentation_mask_topic = self.fp_out_segmented_mask_topic_name
            else:
                self.segmentation_mask_topic = 'segmentation'
        else:
            if self.segmentation_type == SegmentationType.SEGMENT_ANYTHING:
                self.segmentation_mask_topic = self.segmentation_config.sam_in_mask_topic_name
            elif self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
                self.segmentation_mask_topic = self.segmentation_config.sam2_in_mask_topic_name
            else:
                self.segmentation_mask_topic = 'segmentation'

        self.name_space_prefix = get_object_detection_namespace(self.object_detection_type)

        self.depth_camera_info_for_ess = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.fp_in_img_topic_name = camera_config.color_camera_topic_name
        self.fp_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        depth_routing = select_foundation_pose_depth_routing(
            camera_config=camera_config,
            workflow_type=self.workflow_type,
            enable_dnn_depth_in_realsense=self.enable_dnn_depth_in_realsense,
        )
        self.fp_in_depth_topic_name = depth_routing.input_depth_topic
        self.fp_in_depth_image_width = depth_routing.input_depth_image_width
        self.fp_in_depth_image_height = depth_routing.input_depth_image_height

        self.foundation_pose_rgb_image_topic = camera_config.color_camera_topic_name
        self.foundation_pose_rgb_camera_info = camera_config.color_camera_info_topic_name
        self.foundation_pose_depth_image_topic = self.fp_in_depth_topic_name
        self.foundation_pose_detections_topic = self.foundation_pose_detection2_d_topic
        self.realsense_depth_image_topic = self.foundation_pose_depth_image_topic
        if self.workflow_type not in SERVER_BASED_PERCEPTION_WORKFLOWS:
            # A new node is added that drops input to acceptable input hz into perception pipelines
            # It has more hz than the server based approach but not at the same rate as camera
            # streams. This is done to manage compute on the Orin AGX system.
            if camera_config.camera_type is CameraType.REALSENSE:
                self.realsense_depth_image_topic = f'/{self.name_space_prefix}/depth_dropped'
                self.foundation_pose_rgb_image_topic = f'/{self.name_space_prefix}/image_dropped'
                self.foundation_pose_rgb_camera_info = \
                    f'/{self.name_space_prefix}/camera_info_dropped'
            elif camera_config.camera_type is CameraType.ISAAC_SIM:
                self.foundation_pose_rgb_image_topic = f'/{self.name_space_prefix}/image_dropped'
                self.foundation_pose_depth_image_topic = f'/{self.name_space_prefix}/depth_dropped'
                self.foundation_pose_rgb_camera_info = \
                    f'/{self.name_space_prefix}/camera_info_dropped'
            else:
                raise Exception(f'CameraType {camera_config.camera_type} not implemented.')
        else:
            self.foundation_pose_depth_image_topic = self.fp_out_depth_topic_name
            self.foundation_pose_rgb_camera_info = self.fp_out_camera_info_topic_name
            self.foundation_pose_rgb_image_topic = self.fp_out_img_topic_name
            self.foundation_pose_detections_topic = self.fp_out_detections_topic_name
            self.realsense_depth_image_topic = self.fp_out_depth_topic_name

        if camera_config.camera_type is CameraType.REALSENSE:
            self.foundation_pose_depth_image_topic = self.realsense_depth_image_topic + '_metric'

        self.depth_camera_info_for_ess = depth_routing.camera_info_topic


class OrchestrationConfig(Config):
    """Config that tracks all variables needed for orchestration script."""

    behavior_tree_config_file: str
    blackboard_config_file: str
    print_ascii_tree: str
    manual_mode: str
    log_level: str
    frame_prefix: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.behavior_tree_config_file = get_str_variable(context, 'behavior_tree_config_file')
        self.blackboard_config_file = get_str_variable(context, 'blackboard_config_file')
        self.print_ascii_tree = _get_optional_str(context, 'print_ascii_tree', 'false')
        self.manual_mode = _get_optional_str(context, 'manual_mode', 'false')
        self.log_level = _get_optional_str(context, 'log_level', 'error')
        # TF prefix sourced once from the bringup config (Flexiv `robot_sn` /
        # UR `tf_prefix`) so the BT YAML can stay robot-agnostic.
        self.frame_prefix = compute_frame_prefix(context)


class WorkflowConfigParams(Config):
    """
    Config that tracks all variables needed to perform workflows.

    Every workflow (object following, pose-to-pose, pick-and-place,
    displayport cable insertion) runs against a concrete robot, so every
    :class:`WorkflowConfigParams` owns a
    vendor-specific :class:`DriverConfig` instance. Call sites that need
    robot-identity / frame / joint / gripper fields should reach for them through
    ``driver_config`` rather than duplicating the values on the workflow config.
    """

    workflow_type: WorkflowType
    driver_config: DriverConfig

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.driver_config = build_driver_config(context)


class ObjectFollowingConfig(WorkflowConfigParams):
    """Config that tracks all variables needed to perform object following."""

    time_dilation_factor: float
    target_tool_frame: str = 'goal_frame'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')


class PoseToPoseConfig(WorkflowConfigParams):
    """Config that tracks all variables needed to perform pose to pose workflows."""

    time_dilation_factor: float

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')


class PickAndPlaceConfig(WorkflowConfigParams):
    """
    Config that tracks all variables needed to perform pick and place workflows.

    Robot-identity, TF frames, joint names, and gripper-derived fields live on
    ``self.driver_config`` (inherited from :class:`WorkflowConfigParams`). This
    class only adds workflow-level knobs (retries, grasp files, attachment,
    home pose, orchestration). Call sites should read robot/gripper fields via
    ``workflow_config.driver_config.<field>``.
    """

    use_ground_truth_pose_in_sim: bool
    pick_and_place_planner_retries: int
    pick_and_place_retry_wait_time: float
    sim_gt_asset_frame_id: str
    grasps_file_path: str
    object_attachment_type: ObjectAttachmentShape
    object_attachment_scale: LaunchConfiguration
    use_pose_from_rviz: bool
    move_to_home_pose_after_place: bool
    home_pose: List[float]
    time_dilation_factor: str
    attach_object_mesh_file_path: str
    end_effector_mesh_resource_uri: str
    orchestration_config: OrchestrationConfig

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.use_ground_truth_pose_in_sim = _get_optional_bool(
            context, 'use_ground_truth_pose_in_sim', False)
        self.pick_and_place_planner_retries = int(get_str_variable(
            context, 'pick_and_place_planner_retries'
        ))
        self.pick_and_place_retry_wait_time = get_float_variable(
            context, 'pick_and_place_retry_wait_time'
        )
        self.sim_gt_asset_frame_id = _get_optional_str(context, 'sim_gt_asset_frame_id')
        self.grasps_file_path = get_str_variable(
            context, 'grasps_file_path'
        )
        self.object_attachment_type = get_object_attachment_type(get_str_variable(
            context, 'object_attachment_type'
        ))
        self.attach_object_mesh_file_path = _get_optional_str(
            context, 'attach_object_mesh_file_path')
        self.end_effector_mesh_resource_uri = _get_optional_str(
            context, 'end_effector_mesh_resource_uri')
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')
        self.object_attachment_scale = LaunchConfiguration('object_attachment_scale')
        self.use_pose_from_rviz = _get_optional_bool(context, 'use_pose_from_rviz', False)
        self.move_to_home_pose_after_place = _get_optional_bool(
            context, 'move_to_home_pose_after_place', False)
        self.home_pose = _get_optional_float_list(
            context, 'home_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.orchestration_config = OrchestrationConfig(context)


def get_home_joint_positions_from_env_yaml(env_yaml: Dict, yaml_file_path: str) -> List[float]:
    """
    Resolve home/reset arm joints from a policy env.yaml dump.

    Prefer a top-level ``initial_joint_pos`` list. Fall back to
    Isaac Lab ``scene.robot.init_state.joint_pos`` ordered by ``arm_joint_names``.
    """
    if 'initial_joint_pos' in env_yaml:
        return list(env_yaml['initial_joint_pos'])

    joint_pos = (
        env_yaml.get('scene', {})
        .get('robot', {})
        .get('init_state', {})
        .get('joint_pos')
    )
    if isinstance(joint_pos, dict) and joint_pos:
        arm_joint_names = env_yaml.get('arm_joint_names')
        if not arm_joint_names:
            arm_joint_names = sorted(
                (name for name in joint_pos.keys() if str(name).startswith('joint')),
                key=lambda name: int(''.join(ch for ch in str(name) if ch.isdigit()) or 0),
            )
        missing = [name for name in arm_joint_names if name not in joint_pos]
        if missing:
            raise ValueError(
                f'scene.robot.init_state.joint_pos in {yaml_file_path} is missing '
                f'joints {missing}')
        return [float(joint_pos[name]) for name in arm_joint_names]

    raise ValueError(
        f'Neither initial_joint_pos nor scene.robot.init_state.joint_pos found in '
        f'{yaml_file_path}')


class DisplayPortInsertionConfig(PickAndPlaceConfig):
    """Config that tracks all variables needed to perform DisplayPort cable insertion."""

    deploy_config_path: str
    goal_pose_topic: str
    deploy_decimation: int
    deploy_inference_controller_name: str
    deploy_safety_controller_name: str
    deploy_normal_controller_name: str
    deploy_controller_switch_timeout_sec: float
    deploy_blend_ratio: float
    deploy_max_blend_ratio_speed: float
    deploy_pre_goal_wait_sec: float
    deploy_external_policy: bool
    goal_pose_publish_rate_hz: float
    use_joint_space_planner: bool
    use_exact_joint_state_place_pose: bool
    socket_mesh_file_path: str
    cable_mesh_file_path: str
    manual_grasp_mode: bool
    use_manual_poses: bool
    manual_plug_pose: List[float]
    manual_socket_pose: List[float]
    plug_orientation_correction: List[float]
    socket_orientation_correction: List[float]
    socket_mate_offset: List[float]
    run_rl_inference: bool
    pick_retract_offset_distance: List[float]
    use_cartesian_pickup_retract: bool
    disable_pick_planning_scene: bool
    disable_place_planning_scene: bool
    pre_deploy_close_gripper: bool
    pre_deploy_gripper_close_position: float
    post_deploy_open_gripper: bool
    post_deploy_pre_open_wait_sec: float
    post_deploy_open_wait_sec: float
    post_deploy_retract_to_hover: bool
    post_deploy_retreat_lift_distance: float
    output_dir: str
    offset_for_hover_pose: float
    hover_pose_position: List[float]
    override_hover_position: bool
    hover_pose_orientation: List[float]
    timeout_for_insertion_action_call: float
    policy_home_joint_position: List[float]
    socket_tf_frame: str
    enable_controller_switching: bool
    home_joint_state: JointState
    seed_state_for_ik_solver_for_joint_space_planner: List[float]

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.goal_pose_topic = _get_optional_str(
            context, 'displayport_insertion_goal_pose_topic',
            '/displayport_insertion/goal_pose')

        self.use_joint_space_planner = get_bool_variable(
            context, 'displayport_insertion_use_joint_state_planner')
        self.use_exact_joint_state_place_pose = _get_optional_bool(
            context, 'displayport_insertion_use_exact_joint_state_place_pose', False)
        self.socket_mesh_file_path = _get_optional_str(
            context, 'displayport_insertion_socket_mesh_file_path')
        self.cable_mesh_file_path = _get_optional_str(
            context, 'displayport_insertion_cable_mesh_file_path')
        self.manual_grasp_mode = _get_optional_bool(
            context, 'displayport_insertion_manual_grasp_mode', True)
        self.use_manual_poses = get_bool_variable(
            context, 'displayport_insertion_use_manual_poses')
        self.manual_plug_pose = get_float_list_variable(
            context, 'displayport_insertion_manual_plug_pose')
        self.manual_socket_pose = get_float_list_variable(
            context, 'displayport_insertion_manual_socket_pose')
        if self.use_manual_poses:
            self.use_ground_truth_pose_in_sim = True
            self.manual_plug_pose = normalize_pose_quaternion(
                self.manual_plug_pose, 'displayport_insertion_manual_plug_pose')
            self.manual_socket_pose = normalize_pose_quaternion(
                self.manual_socket_pose, 'displayport_insertion_manual_socket_pose')
        # Fixed mesh-frame rotation [qx, qy, qz, qw] that maps the FoundationPose plug mesh
        # frame onto the convention the grasp file expects (+Z along the gripper approach
        # axis). Empty list keeps the estimated frame unchanged.
        self.plug_orientation_correction = _get_optional_float_list(
            context, 'displayport_insertion_plug_orientation_correction', [])
        if self.plug_orientation_correction:
            if len(self.plug_orientation_correction) != 4:
                raise ValueError(
                    'displayport_insertion_plug_orientation_correction must contain 4 values '
                    '[qx, qy, qz, qw] or be empty')
            normalized = normalize_pose_quaternion(
                [0.0, 0.0, 0.0] + list(self.plug_orientation_correction),
                'displayport_insertion_plug_orientation_correction')
            self.plug_orientation_correction = normalized[3:]
        else:
            self.plug_orientation_correction = [0.0, 0.0, 0.0, 1.0]

        self.socket_orientation_correction = _get_optional_float_list(
            context, 'displayport_insertion_socket_orientation_correction', [])
        if self.socket_orientation_correction:
            if len(self.socket_orientation_correction) != 4:
                raise ValueError(
                    'displayport_insertion_socket_orientation_correction must contain 4 values '
                    '[qx, qy, qz, qw]')
            normalized = normalize_pose_quaternion(
                [0.0, 0.0, 0.0] + list(self.socket_orientation_correction),
                'displayport_insertion_socket_orientation_correction')
            self.socket_orientation_correction = normalized[3:]
        else:
            self.socket_orientation_correction = [0.0, 0.0, 0.0, 1.0]

        # Mesh-origin-to-rim translation, in the corrected socket frame.
        self.socket_mate_offset = _get_optional_float_list(
            context, 'displayport_insertion_socket_mate_offset', []) or [0.0, 0.0, 0.0]
        if len(self.socket_mate_offset) != 3:
            raise ValueError(
                'displayport_insertion_socket_mate_offset must contain 3 values [x, y, z]')
        self.run_rl_inference = get_bool_variable(
            context, 'displayport_insertion_run_rl_inference')
        self.pick_retract_offset_distance = _get_optional_float_list(
            context, 'displayport_insertion_pick_retract_offset_distance', [])
        if self.pick_retract_offset_distance and len(self.pick_retract_offset_distance) != 3:
            raise ValueError(
                'displayport_insertion_pick_retract_offset_distance must contain 3 values '
                '[x, y, z] or be empty')
        self.use_cartesian_pickup_retract = _get_optional_bool(
            context, 'displayport_insertion_use_cartesian_pickup_retract', False)
        self.disable_pick_planning_scene = _get_optional_bool(
            context, 'displayport_insertion_disable_pick_planning_scene', False)
        self.disable_place_planning_scene = _get_optional_bool(
            context, 'displayport_insertion_disable_place_planning_scene', False)
        self.pre_deploy_close_gripper = _get_optional_bool(
            context, 'displayport_insertion_pre_deploy_close_gripper', False)
        self.pre_deploy_gripper_close_position = _get_optional_float(
            context,
            'displayport_insertion_pre_deploy_gripper_close_position',
            self.driver_config.gripper_close_position)
        self.post_deploy_open_gripper = _get_optional_bool(
            context, 'displayport_insertion_post_deploy_open_gripper', False)
        self.post_deploy_pre_open_wait_sec = _get_optional_float(
            context, 'displayport_insertion_post_deploy_pre_open_wait_sec', 0.0)
        if self.post_deploy_pre_open_wait_sec < 0.0:
            raise ValueError(
                'displayport_insertion_post_deploy_pre_open_wait_sec must be non-negative')
        self.post_deploy_open_wait_sec = _get_optional_float(
            context, 'displayport_insertion_post_deploy_open_wait_sec', 0.0)
        if self.post_deploy_open_wait_sec < 0.0:
            raise ValueError(
                'displayport_insertion_post_deploy_open_wait_sec must be non-negative')
        self.post_deploy_retract_to_hover = _get_optional_bool(
            context, 'displayport_insertion_post_deploy_retract_to_hover', False)
        self.post_deploy_retreat_lift_distance = _get_optional_float(
            context, 'displayport_insertion_post_deploy_retreat_lift_distance', 0.0)
        if self.post_deploy_retreat_lift_distance < 0.0:
            raise ValueError(
                'displayport_insertion_post_deploy_retreat_lift_distance must be non-negative')
        self.deploy_config_path = _get_optional_str(
            context, 'displayport_insertion_deploy_config_path')
        self.deploy_external_policy = _get_optional_bool(
            context, 'displayport_insertion_deploy_external_policy', False)
        self.requires_deploy = (
            self.run_rl_inference
            and not self.deploy_external_policy)
        if self.requires_deploy:
            if not self.deploy_config_path:
                raise ValueError(
                    'displayport_insertion_deploy_config_path is required when '
                    'displayport_insertion_run_rl_inference is true and insertion is enabled')
            if not os.path.isfile(self.deploy_config_path):
                raise ValueError(
                    f'displayport_insertion_deploy_config_path file not found: '
                    f'{self.deploy_config_path}')
        # Task-space policies emit Cartesian deltas, driven through the Flexiv Cartesian
        # motion controller. Optional, so joint-space workflow configs stay valid.
        self.use_task_space_policy = _get_optional_str(
            context, 'displayport_insertion_use_task_space_policy') == 'true'
        self.task_space_socket_pose_topic = ''
        self.task_space_enable_topic = ''
        self.task_space_cartesian_controller_name = ''
        self.task_space_cartesian_impedance_service_name = ''
        # Placeholder [0.0] rather than []: forwarded verbatim as node parameters, and
        # launch_ros cannot type an empty list. Only read on the task-space path.
        self.task_space_cartesian_stiffness = [0.0]
        self.task_space_cartesian_damping_ratio = [0.0]
        self.task_space_release_after_insertion = False
        self.task_space_return_home_after_insertion = False
        self.task_space_home_pose = [0.0]
        if self.use_task_space_policy:
            self.task_space_tcp_offset = get_str_variable(
                context, 'displayport_insertion_task_space_tcp_offset')
            self.task_space_adapter_tcp_offset = _get_optional_str(
                context, 'displayport_insertion_task_space_adapter_tcp_offset') or ''
            self.task_space_action_blend_ratio = _get_optional_float(
                context, 'displayport_insertion_task_space_action_blend_ratio', 0.35)
            if (self.task_space_action_blend_ratio <= 0.0
                    or self.task_space_action_blend_ratio > 1.0):
                raise ValueError(
                    'displayport_insertion_task_space_action_blend_ratio must be in '
                    'the range (0.0, 1.0]')
            self.task_space_socket_pose_topic = _get_optional_str(
                context, 'displayport_insertion_task_space_socket_pose_topic'
            ) or '/displayport_socket_pose'
            self.task_space_enable_topic = _get_optional_str(
                context, 'displayport_insertion_task_space_enable_topic'
            ) or '/displayport_task_space_policy/enable_passthrough'
            self.task_space_cartesian_controller_name = _get_optional_str(
                context, 'displayport_insertion_task_space_cartesian_controller_name'
            ) or 'cartesian_motion_controller'
            # Set once the Cartesian controller is active; rejected in a joint mode
            # Defaults to the name the Flexiv driver derives from the robot serial
            # Unset derives the name the Flexiv driver builds from robot_sn; an explicit
            # '' still disables the impedance call.
            self.task_space_cartesian_impedance_service_name = _get_optional_str(
                context,
                'displayport_insertion_task_space_cartesian_impedance_service_name',
                _DERIVE_IMPEDANCE_SERVICE)
            if self.task_space_cartesian_impedance_service_name == _DERIVE_IMPEDANCE_SERVICE:
                robot_sn = self.driver_config.robot_sn
                prefix = f'{robot_sn}_'.replace('-', '_') if robot_sn else ''
                self.task_space_cartesian_impedance_service_name = (
                    f'/{prefix}flexiv_hardware_services/set_cartesian_impedance'
                    if robot_sn else '')
            self.task_space_cartesian_stiffness = _get_optional_float_list(
                context, 'displayport_insertion_task_space_cartesian_stiffness', []) or [0.0]
            self.task_space_cartesian_damping_ratio = _get_optional_float_list(
                context, 'displayport_insertion_task_space_cartesian_damping_ratio', []) or [0.0]
            if self.task_space_cartesian_impedance_service_name:
                for name, values in (
                        ('stiffness', self.task_space_cartesian_stiffness),
                        ('damping_ratio', self.task_space_cartesian_damping_ratio)):
                    if len(values) != 6:
                        raise ValueError(
                            f'displayport_insertion_task_space_cartesian_{name} must contain '
                            '6 values when a Cartesian impedance service is configured')
            # Release the plug and retract to home once insertion finishes.
            self.task_space_release_after_insertion = _get_optional_str(
                context, 'displayport_insertion_task_space_release_after_insertion'
            ) == 'true'
            self.task_space_return_home_after_insertion = _get_optional_str(
                context, 'displayport_insertion_task_space_return_home_after_insertion'
            ) == 'true'
            # Retract target, tunable independently of the pick-and-place ``home_pose``.
            # Resolved here because ``home_pose`` on the base config is an unresolved
            # LaunchConfiguration, not a float list.
            self.task_space_home_pose = _get_optional_float_list(
                context, 'displayport_insertion_task_space_home_pose', [])
            if not self.task_space_home_pose:
                self.task_space_home_pose = get_float_list_variable(context, 'home_pose')
            if (self.task_space_return_home_after_insertion
                    and len(self.task_space_home_pose) != 7):
                raise ValueError(
                    'home_pose must contain 7 values [x, y, z, qx, qy, qz, qw] when '
                    'displayport_insertion_task_space_return_home_after_insertion is true')
        self.deploy_decimation = int(_get_optional_str(
            context, 'displayport_insertion_deploy_decimation') or '1')
        if self.deploy_decimation <= 0:
            raise ValueError('displayport_insertion_deploy_decimation must be positive')
        self.deploy_inference_controller_name = _get_optional_str(
            context, 'displayport_insertion_deploy_inference_controller_name'
        ) or 'deploy_inference_controller'
        self.deploy_safety_controller_name = _get_optional_str(
            context, 'displayport_insertion_deploy_safety_controller_name'
        ) or 'deploy_safety_controller'
        self.deploy_normal_controller_name = _get_optional_str(
            context, 'displayport_insertion_deploy_normal_controller_name'
        ) or 'rizon_arm_controller'
        self.deploy_controller_switch_timeout_sec = _get_optional_float(
            context, 'displayport_insertion_deploy_controller_switch_timeout_sec', 10.0)
        if self.deploy_controller_switch_timeout_sec <= 0.0:
            raise ValueError(
                'displayport_insertion_deploy_controller_switch_timeout_sec must be positive')
        self.deploy_blend_ratio = _get_optional_float(
            context, 'displayport_insertion_deploy_blend_ratio', 1.0)
        if self.deploy_blend_ratio < 0.0 or self.deploy_blend_ratio > 1.0:
            raise ValueError(
                'displayport_insertion_deploy_blend_ratio must be in the range [0.0, 1.0]')
        self.deploy_max_blend_ratio_speed = _get_optional_float(
            context, 'displayport_insertion_deploy_max_blend_ratio_speed', 1.0)
        if self.deploy_max_blend_ratio_speed <= 0.0 and self.deploy_blend_ratio > 0.0:
            raise ValueError(
                'displayport_insertion_deploy_max_blend_ratio_speed must be positive when '
                'displayport_insertion_deploy_blend_ratio is non-zero')
        self.deploy_pre_goal_wait_sec = _get_optional_float(
            context, 'displayport_insertion_deploy_pre_goal_wait_sec', 0.25)
        if self.deploy_pre_goal_wait_sec < 0.0:
            raise ValueError('displayport_insertion_deploy_pre_goal_wait_sec must be non-negative')
        self.goal_pose_publish_rate_hz = _get_optional_float(
            context, 'displayport_insertion_goal_pose_publish_rate_hz', 30.0)
        if self.goal_pose_publish_rate_hz <= 0.0:
            raise ValueError('displayport_insertion_goal_pose_publish_rate_hz must be positive')
        self.output_dir = _get_optional_str(context, 'displayport_insertion_output_dir')
        self.offset_for_hover_pose = get_float_variable(
            context, 'displayport_insertion_offset_for_hover_pose')
        # Optional fixed hover position [x, y, z] in the robot base frame. Empty list
        # keeps the socket-derived hover position (socket x/y, socket z + offset).
        self.hover_pose_position = _get_optional_float_list(
            context, 'displayport_insertion_hover_pose_position', [])
        if self.hover_pose_position:
            if len(self.hover_pose_position) != 3:
                raise ValueError(
                    'displayport_insertion_hover_pose_position must contain 3 values '
                    '[x, y, z] or be empty')
            self.override_hover_position = True
        else:
            # Placeholder avoids empty-list parameter typing issues; the override flag
            # tells the orchestrator to ignore it and derive the hover from the socket.
            self.override_hover_position = False
            self.hover_pose_position = [0.0, 0.0, 0.0]
        # Optional fixed hover orientation [qx, qy, qz, qw] in the robot base frame.
        # Empty list (or the [0, 0, 0, 0] sentinel) keeps the socket orientation.
        self.hover_pose_orientation = get_float_list_variable(
            context, 'displayport_insertion_hover_pose_orientation')
        if self.hover_pose_orientation:
            if len(self.hover_pose_orientation) != 4:
                raise ValueError(
                    'displayport_insertion_hover_pose_orientation must contain 4 values '
                    '[qx, qy, qz, qw] or be empty')
            normalized = normalize_pose_quaternion(
                [0.0, 0.0, 0.0] + list(self.hover_pose_orientation),
                'displayport_insertion_hover_pose_orientation')
            self.hover_pose_orientation = normalized[3:]
        else:
            # Zero-quaternion sentinel avoids empty-list parameter typing issues and
            # signals the orchestrator to keep the socket orientation.
            self.hover_pose_orientation = [0.0, 0.0, 0.0, 0.0]
        self.timeout_for_insertion_action_call = get_float_variable(
            context, 'displayport_insertion_timeout_for_insertion_action_call')
        self.socket_tf_frame = _get_optional_str(
            context, 'displayport_insertion_socket_tf_frame',
            'displayport_socket_frame')
        self.enable_controller_switching = _get_optional_bool(
            context, 'displayport_insertion_enable_controller_switching', True)

        if not self.manual_grasp_mode and not self.use_manual_poses:
            if not self.cable_mesh_file_path or not os.path.isfile(self.cable_mesh_file_path):
                raise ValueError(
                    'displayport_insertion_cable_mesh_file_path must point to an existing mesh '
                    'when displayport_insertion_manual_grasp_mode is false')
        if not self.use_manual_poses:
            if not self.socket_mesh_file_path or not os.path.isfile(self.socket_mesh_file_path):
                raise ValueError(
                    f'displayport_insertion_socket_mesh_file_path must point to an existing mesh: '
                    f'{self.socket_mesh_file_path}')

        home_target_position = get_float_list_variable(
            context, 'displayport_insertion_policy_home_joint_position')
        self.policy_home_joint_position = home_target_position
        joint_names = self.driver_config.joint_names
        if len(home_target_position) != len(joint_names):
            raise ValueError(
                f'displayport_insertion_policy_home_joint_position has '
                f'{len(home_target_position)} values but the selected robot has '
                f'{len(joint_names)} joints.')

        joint_state = JointState()
        joint_state.position = home_target_position
        joint_state.velocity = [0.0] * len(home_target_position)
        joint_state.effort = [0.0] * len(home_target_position)
        joint_state.name = list(joint_names)
        self.home_joint_state = joint_state
        self.seed_state_for_ik_solver_for_joint_space_planner = home_target_position


class DepthEstimationConfig(Config):
    """Config that tracks all variables needed to perform depth estimation."""

    depth_type: DepthType
    enable_dnn_depth_in_realsense: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.depth_type = _get_optional_str(context, 'depth_type', str(DepthType.REALSENSE))
        self.enable_dnn_depth_in_realsense = _get_optional_bool(
            context, 'enable_dnn_depth_in_realsense', False)


class CuMotionConfig(Config):
    """Config that tracks all variables needed to configure cuMotion."""

    cumotion_urdf_file_path: str
    cumotion_xrdf_file_path: str
    cumotion_joint_states_topic: str
    distance_threshold: float
    num_cameras: int
    moveit_collision_objects_scene_file: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.num_cameras = int(_get_optional_str(context, 'num_cameras', '1'))
        self.cumotion_urdf_file_path = get_str_variable(context, 'cumotion_urdf_file_path')
        self.cumotion_xrdf_file_path = get_str_variable(context, 'cumotion_xrdf_file_path')
        self.distance_threshold = _get_optional_str(context, 'distance_threshold', '0.15')
        self.moveit_collision_objects_scene_file = _get_optional_str(
            context, 'moveit_collision_objects_scene_file')
        self.cumotion_joint_states_topic = _get_optional_str(
            context, 'cumotion_joint_states_topic', '/joint_states')


class CoreConfig(Config):
    """Config that tracks all variables needed to perform core workflows."""

    camera_config: CameraConfig
    rviz_config_file: str
    object_selection_config: Optional[ObjectSelectionConfig]
    pose_estimation_config: Optional[PoseEstimationConfig]
    workflow_config: WorkflowConfigParams
    cumotion_config: CuMotionConfig
    # cuMotion and object attachment params
    time_sync_slop: str
    # nvblox configs
    setup: str
    enable_nvblox: bool
    nvblox_global_frame: str
    # Depth estimation params
    depth_estimation_config: DepthEstimationConfig
    # Visualization
    enable_rviz_visualization: bool
    enable_foxglove_visualization: bool
    # Nsight Profiling settings
    delay_to_start_nsight: str
    nsight_profile_duration: str
    nsight_profile_output_file_path: str
    enable_nsight_profiling: bool
    enable_system_wide_profiling: bool
    # MPS parameters
    enable_cuda_mps: bool
    cuda_mps_pipe_directory: str
    cuda_mps_active_thread_percentage_container: str
    cuda_mps_client_priority_container: str
    cuda_mps_client_priority_robot_segmenter: str
    cuda_mps_active_thread_percentage_robot_segmenter: str
    # Logging level
    log_level: str = 'error'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = get_camera_type(
            _get_optional_str(context, 'camera_type', str(CameraType.REALSENSE)))
        self.rviz_config_file = _get_optional_str(context, 'rviz_config_file')
        self.time_sync_slop = _get_optional_str(context, 'time_sync_slop', '0.1')
        self.depth_type = _get_optional_str(context, 'depth_type', str(DepthType.REALSENSE))
        self.use_sim_time = get_bool_variable(context, 'use_sim_time')
        self.num_cameras = int(_get_optional_str(context, 'num_cameras', '1'))
        self.setup = get_str_variable(context, 'setup')
        self.enable_nvblox = _get_optional_bool(context, 'enable_nvblox', False)
        self.nvblox_global_frame = _get_optional_str(context, 'nvblox_global_frame', 'base_link')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.uses_perception = not (
            self.workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION and
            _get_optional_bool(context, 'displayport_insertion_use_manual_poses', False)
        )
        self.pose_estimation_type = None
        if self.uses_perception:
            self.pose_estimation_type = get_pose_estimation_type(
                get_str_variable(context, 'pose_estimation_type'))
        self.delay_to_start_nsight = _get_optional_str(context, 'delay_to_start_nsight', '5')
        self.nsight_profile_duration = _get_optional_str(
            context, 'nsight_profile_duration', '20')
        self.nsight_profile_output_file_path = _get_optional_str(
            context, 'nsight_profile_output_file_path', '$ISAAC_ROS_WS/nsight_profile.nsys')
        self.enable_nsight_profiling = _get_optional_bool(
            context, 'enable_nsight_profiling', False)
        self.enable_system_wide_profiling = _get_optional_bool(
            context, 'enable_system_wide_profiling', False)

        self.enable_rviz_visualization = _get_optional_bool(
            context, 'enable_rviz_visualization', False)
        self.enable_foxglove_visualization = _get_optional_bool(
            context, 'enable_foxglove_visualization', False)
        # MPS parameters
        self.enable_cuda_mps = _get_optional_bool(context, 'enable_cuda_mps', False)
        self.cuda_mps_pipe_directory = _get_optional_str(
            context, 'cuda_mps_pipe_directory', '$ISAAC_ROS_WS/mps_pipe_dir')
        self.cuda_mps_active_thread_percentage_container = _get_optional_str(
            context, 'cuda_mps_active_thread_percentage_container', '100'
        )
        self.cuda_mps_client_priority_container = _get_optional_str(
            context, 'cuda_mps_client_priority_container', '1'
        )
        self.cuda_mps_client_priority_robot_segmenter = _get_optional_str(
            context, 'cuda_mps_client_priority_robot_segmenter', '1'
        )
        self.cuda_mps_active_thread_percentage_robot_segmenter = _get_optional_str(
            context, 'cuda_mps_active_thread_percentage_robot_segmenter', '100'
        )
        self.cuda_mps_client_priority_planner = _get_optional_str(
            context, 'cuda_mps_client_priority_planner', '0'
        )
        self.cuda_mps_active_thread_percentage_planner = _get_optional_str(
            context, 'cuda_mps_active_thread_percentage_planner', '100'
        )

        self.depth_estimation_config = DepthEstimationConfig(context)
        if self.camera_type == CameraType.ISAAC_SIM and self.num_cameras != 1:
            raise NotImplementedError(f'Camera type of isaac sim only support 1 camera'
                                      f', not {self.num_cameras}')
        self.camera_config = get_camera_config(context, self.camera_type)

        self.object_selection_config = None
        self.pose_estimation_config = None
        if self.uses_perception:
            self.object_selection_config = ObjectSelectionConfig(context)

            if self.pose_estimation_type == PoseEstimationType.DOPE:
                if self.workflow_type in (
                    WorkflowType.PICK_AND_PLACE,
                    WorkflowType.DISPLAYPORT_CABLE_INSERTION,
                ):
                    raise NotImplementedError(
                        'Dope is not supported for pick and place or displayport '
                        'cable insertion')
                self.pose_estimation_config = DopeConfig(context)
            elif self.pose_estimation_type == PoseEstimationType.FOUNDATION_POSE:
                self.pose_estimation_config = FoundationPoseConfig(context)
                # Now update the topic names for segmentation and object detection configs.
                self.pose_estimation_config.object_detection_config.update_server_topic_names(
                    self.camera_config)
                if self.pose_estimation_config.segmentation_type != SegmentationType.NONE:
                    self.pose_estimation_config.segmentation_config.update_server_topic_names(
                        self.camera_config)
                self.pose_estimation_config.update_server_topic_names(self.camera_config)
            else:
                raise NotImplementedError(f'Pose estimation type {self.pose_estimation_type} '
                                          f'not supported')

        if self.workflow_type == WorkflowType.POSE_TO_POSE:
            self.workflow_config = PoseToPoseConfig(context)
        elif self.workflow_type == WorkflowType.OBJECT_FOLLOWING:
            self.workflow_config = ObjectFollowingConfig(context)
        elif self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.workflow_config = PickAndPlaceConfig(context)
        elif self.workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION:
            self.workflow_config = DisplayPortInsertionConfig(context)
        else:
            raise NotImplementedError(f'Workflow type not supported {self.workflow_type} !')

        self.object_detection_type = None
        if self.pose_estimation_config is not None:
            self.object_detection_type = self.pose_estimation_config.object_detection_type

        self.cumotion_config = CuMotionConfig(context)

        if (
            self.workflow_type == WorkflowType.DISPLAYPORT_CABLE_INSERTION and
            self.uses_perception
        ):
            if self.object_detection_type != ObjectDetectionType.SEGMENT_ANYTHING:
                raise ValueError(
                    'Object detection type != SEGMENT_ANYTHING is not supported for '
                    f'{self.workflow_type.value}')


@dataclass
class SupportedObject:
    """Configuration for individual supported object."""

    class_id: str
    grasp_file_path: str
    mesh_file_path: str
    object_name: str

    @classmethod
    def from_dict(cls, object_name: str, params: Dict[str, Any]) -> 'SupportedObject':
        return cls(
            class_id=params.get('class_id', ''),
            grasp_file_path=params.get('grasp_file_path', ''),
            mesh_file_path=params.get('mesh_file_path', ''),
            object_name=object_name
        )


@dataclass
class SupportedObjectsConfig:
    """Configuration for supported objects containing multiple object types."""

    supported_objects: Dict[str, SupportedObject]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'SupportedObjectsConfig':
        supported_objects = {}

        for object_name, object_data in params.items():
            if not isinstance(object_data, dict):
                continue

            supported_object = SupportedObject.from_dict(object_name, object_data)
            supported_objects[supported_object.class_id] = supported_object

        return cls(
            supported_objects=supported_objects,
        )

    def get_object_by_class_id(self, class_id: str) -> Optional[SupportedObject]:
        """Get supported object configuration by class ID."""
        return self.supported_objects.get(class_id, None)
