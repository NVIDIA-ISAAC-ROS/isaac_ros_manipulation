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

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from geometry_msgs.msg import Pose
from isaac_ros_manipulation_interfaces.action import MultiObjectPickAndPlace
from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params, SupportedObjectsConfig


@dataclass
class WorkspaceBoundsConfig:
    """
    Configuration for workspace bounds filtering during pose estimation.

    Defines a 3D bounding box to filter detected objects outside the robot's
    reachable workspace. The box is specified by two opposite corners forming
    a diagonal, typically aligned parallel to the workspace table.

    Attributes
    ----------
    diagonal : List[List[float]]
        Two 3D points [[x1,y1,z1], [x2,y2,z2]] defining the box.
        Empty list [] disables filtering.

    """

    diagonal: List[List[float]]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'WorkspaceBoundsConfig':
        wb = params.get('workspace_bounds', {}) or {}
        diagonal = wb.get('diagonal')

        if diagonal and isinstance(diagonal, list) and len(diagonal) == 2:
            p1, p2 = diagonal[0], diagonal[1]
            if isinstance(p1, list) and isinstance(p2, list) and len(p1) == 3 and len(p2) == 3:
                # Convert to float and store the diagonal corners
                corner1 = [float(p1[i]) for i in range(3)]
                corner2 = [float(p2[i]) for i in range(3)]
                return cls(diagonal=[corner1, corner2])

        # Default empty if not properly specified
        # Empty diagonal means no workspace filtering
        return cls(diagonal=[])


@dataclass
class PlanToGraspConfig:
    """Configuration for plan_to_grasp behavior."""

    action_server_name: str
    link_name: str
    time_dilation_factor: float
    grasp_approach_offset_distance: List[float]
    grasp_translation_path_deviation_limit: float
    grasp_translation_terminal_deviation_limit: float
    grasp_enable_orientation_path_axis_constraint: bool
    grasp_orientation_terminal_deviation_limit: float
    grasp_orientation_path_axis_deviation_limit: float
    retract_offset_distance: List[float]
    retract_translation_path_deviation_limit: float
    retract_translation_terminal_deviation_limit: float
    retract_enable_orientation_path_axis_constraint: bool
    retract_orientation_terminal_deviation_limit: float
    retract_orientation_path_axis_deviation_limit: float
    grasp_approach_constraint_in_goal_frame: bool
    retract_constraint_in_goal_frame: bool
    update_planning_scene: bool
    world_frame: str
    enable_aabb_clearing: bool
    esdf_clearing_padding: List[float]
    aabb_clearing_shape: str
    aabb_clearing_shape_scale: List[float]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'PlanToGraspConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', 'cumotion/motion_plan'),
            link_name=params.get('link_name', 'base_link'),
            time_dilation_factor=params.get('time_dilation_factor', 0.2),
            grasp_approach_offset_distance=params.get(
                'grasp_approach_offset_distance', [0.0, 0.0, -0.15]
            ),
            grasp_translation_path_deviation_limit=params.get(
                'grasp_translation_path_deviation_limit', 0.001
            ),
            grasp_translation_terminal_deviation_limit=params.get(
                'grasp_translation_terminal_deviation_limit', -1.0
            ),
            grasp_enable_orientation_path_axis_constraint=params.get(
                'grasp_enable_orientation_path_axis_constraint', True
            ),
            grasp_orientation_terminal_deviation_limit=params.get(
                'grasp_orientation_terminal_deviation_limit', -1.0
            ),
            grasp_orientation_path_axis_deviation_limit=params.get(
                'grasp_orientation_path_axis_deviation_limit', -1.0
            ),
            retract_offset_distance=params.get(
                'retract_offset_distance', [0.0, 0.0, 0.15]
            ),
            retract_translation_path_deviation_limit=params.get(
                'retract_translation_path_deviation_limit', 0.001
            ),
            retract_translation_terminal_deviation_limit=params.get(
                'retract_translation_terminal_deviation_limit', -1.0
            ),
            retract_enable_orientation_path_axis_constraint=params.get(
                'retract_enable_orientation_path_axis_constraint', True
            ),
            retract_orientation_terminal_deviation_limit=params.get(
                'retract_orientation_terminal_deviation_limit', -1.0
            ),
            retract_orientation_path_axis_deviation_limit=params.get(
                'retract_orientation_path_axis_deviation_limit', -1.0
            ),
            grasp_approach_constraint_in_goal_frame=params.get(
                'grasp_approach_constraint_in_goal_frame', True
            ),
            retract_constraint_in_goal_frame=params.get(
                'retract_constraint_in_goal_frame', False
            ),
            update_planning_scene=params.get('update_planning_scene', True),
            world_frame=params.get('world_frame', 'world'),
            enable_aabb_clearing=params.get('enable_aabb_clearing', False),
            esdf_clearing_padding=params.get(
                'esdf_clearing_padding', [0.05, 0.05, 0.05]
            ),
            aabb_clearing_shape=params.get('aabb_clearing_shape', 'CUBOID'),
            aabb_clearing_shape_scale=params.get(
                'aabb_clearing_shape_scale', [0.15, 0.15, 0.20]
            )
        )


@dataclass
class PlanToPoseConfig:
    """Configuration for plan_to_pose behavior."""

    action_server_name: str
    link_name: str
    time_dilation_factor: float
    update_planning_scene: bool
    world_frame: str
    aabb_clearing_shape: str
    aabb_clearing_shape_scale: List[float]
    enable_aabb_clearing: bool
    esdf_clearing_padding: List[float]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'PlanToPoseConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', 'cumotion/motion_plan'),
            link_name=params.get('link_name', 'base_link'),
            time_dilation_factor=params.get('time_dilation_factor', 0.2),
            update_planning_scene=params.get('update_planning_scene', True),
            world_frame=params.get('world_frame', 'world'),
            aabb_clearing_shape=params.get('aabb_clearing_shape', 'SPHERE'),
            aabb_clearing_shape_scale=params.get(
                'aabb_clearing_shape_scale', [0.1, 0.1, 0.1]
            ),
            enable_aabb_clearing=params.get('enable_aabb_clearing', False),
            esdf_clearing_padding=params.get(
                'esdf_clearing_padding', [0.05, 0.05, 0.05]
            )
        )


@dataclass
class ReadGraspPosesConfig:
    """Configuration for read_grasp_poses behavior."""

    publish_grasp_poses: bool

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ReadGraspPosesConfig':
        return cls(
            publish_grasp_poses=params.get('publish_grasp_poses', True)
        )


@dataclass
class CloseGripperConfig:
    """Configuration for gripper behaviors (close)."""

    gripper_action_name: str
    close_position: float
    close_positions_by_class: Dict[str, float]
    max_effort: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'CloseGripperConfig':
        return cls(
            gripper_action_name=params.get(
                'gripper_action_name', '/robotiq_gripper_controller/gripper_cmd'
            ),
            close_position=params.get('close_position', 0.55),
            close_positions_by_class={
                str(class_id): float(position)
                for class_id, position in (
                    params.get('close_positions_by_class') or {}).items()
            },
            max_effort=params.get('max_effort', 10.0)
        )


@dataclass
class OpenGripperConfig:
    """Configuration for gripper behaviors (open)."""

    gripper_action_name: str
    open_position: float
    max_effort: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'OpenGripperConfig':
        return cls(
            gripper_action_name=params.get(
                'gripper_action_name', '/robotiq_gripper_controller/gripper_cmd'
            ),
            open_position=params.get('open_position', 0.0),
            max_effort=params.get('max_effort', 10.0)
        )


@dataclass
class ObjectAttachDetachConfig:
    """Configuration for object attachment/detachment behaviors."""

    action_server_name: str
    shape: str
    scale: List[float]
    gripper_frame: str
    grasp_frame: str

    @classmethod
    def from_dict(
        cls,
        params: Dict[str, Any],
        frame_prefix: str = '',
    ) -> 'ObjectAttachDetachConfig':
        gripper_frame = params.get('gripper_frame', 'gripper_frame')
        grasp_frame = params.get('grasp_frame', 'grasp_frame')
        return cls(
            action_server_name=params.get(
                'action_server_name', 'attach_object'),
            shape=params.get('shape', 'CUBOID'),
            scale=params.get('scale', [0.1, 0.1, 0.2]),
            gripper_frame=f'{frame_prefix}{gripper_frame}',
            grasp_frame=f'{frame_prefix}{grasp_frame}',
        )


@dataclass
class ExecuteTrajectoryConfig:
    """Configuration for execute_trajectory behavior."""

    action_server_name: str
    index: int

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ExecuteTrajectoryConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', 'execute_trajectory'),
            index=params.get('index', 0)
        )


@dataclass
class RetryConfig:
    """Configuration for retry parameters."""

    max_planning_retries: int
    max_controller_retries: int
    max_detection_retries: int
    max_pose_estimation_retries: int
    max_gripper_retries: int
    max_attachment_retries: int

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'RetryConfig':
        return cls(
            max_planning_retries=params.get('max_planning_retries', 3),
            max_controller_retries=params.get('max_controller_retries', 3),
            max_detection_retries=params.get('max_detection_retries', 3),
            max_pose_estimation_retries=params.get(
                'max_pose_estimation_retries', 3),
            max_gripper_retries=params.get('max_gripper_retries', 3),
            max_attachment_retries=params.get('max_attachment_retries', 3)
        )


@dataclass
class InteractiveMarkerConfig:
    """Configuration for interactive_marker behavior."""

    mesh_resource_uri: str
    reference_frame: str
    end_effector_frame: str
    user_confirmation_timeout: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'InteractiveMarkerConfig':
        return cls(
            mesh_resource_uri=params.get('mesh_resource_uri', ''),
            reference_frame=params.get('reference_frame', 'base_link'),
            end_effector_frame=params.get('end_effector_frame', 'gripper_frame'),
            user_confirmation_timeout=params.get('user_confirmation_timeout', 10.0)
        )


@dataclass
class DetectObjectConfig:
    """Configuration for detect_object behavior."""

    action_server_name: str
    detection_confidence_threshold: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'DetectObjectConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', '/get_objects'),
            detection_confidence_threshold=params.get(
                'detection_confidence_threshold', 0.5)
        )


@dataclass
class StaleDetectionConfig:
    """Configuration for stale detection handler."""

    timeout_duration: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'StaleDetectionConfig':
        return cls(
            timeout_duration=params.get('timeout_duration', 300.0)
        )


@dataclass
class MeshAssignerConfig:
    """Configuration for mesh_assigner behavior."""

    service_name: str

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'MeshAssignerConfig':
        return cls(
            service_name=params.get('service_name', '/add_mesh_to_object')
        )


@dataclass
class AssignObjectNameConfig:
    """Configuration for assign_object_name behavior."""

    service_name: str

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'AssignObjectNameConfig':
        return cls(
            service_name=params.get('service_name', 'assign_name_to_object')
        )


@dataclass
class PublishStaticPlanningSceneConfig:
    """Configuration for publish_static_planning_scene behavior."""

    service_name: str
    scene_file_path: str

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'PublishStaticPlanningSceneConfig':
        return cls(
            service_name=params.get('service_name', 'publish_static_planning_scene'),
            scene_file_path=params.get('scene_file_path', '')
        )


@dataclass
class ObjectSelectorConfig:
    """Configuration for object_selector behavior."""

    action_server_name: str

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ObjectSelectorConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', '/get_selected_object')
        )


@dataclass
class PoseEstimationConfig:
    """Configuration for pose_estimation behavior."""

    action_server_name: str
    workspace_bounds: WorkspaceBoundsConfig
    base_frame_id: str
    camera_frame_id: str

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'PoseEstimationConfig':
        return cls(
            action_server_name=params.get(
                'action_server_name', '/get_object_pose'),
            workspace_bounds=WorkspaceBoundsConfig.from_dict(params),
            base_frame_id=params.get(
                'base_frame_id', 'base_link'),
            camera_frame_id=params.get(
                'camera_frame_id', '')
        )


@dataclass
class SwitchControllersConfig:
    """Configuration for switch_controllers behavior."""

    controllers_to_activate: List[str]
    controllers_to_deactivate: List[str]
    strictness: int

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'SwitchControllersConfig':
        return cls(
            controllers_to_activate=params.get('controllers_to_activate', []),
            controllers_to_deactivate=params.get(
                'controllers_to_deactivate', []),
            strictness=params.get('strictness', 4)  # Default to FORCE_AUTO
        )


@dataclass
class ServerTimeoutConfig:
    """Configuration for server timeout settings."""

    startup_server_timeout_sec: Optional[float]
    runtime_retry_timeout_sec: float
    server_check_interval_sec: float

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ServerTimeoutConfig':
        """Create ServerTimeoutConfig from dictionary."""

        def _convert_timeout_value(value: Any) -> Optional[float]:
            """Convert timeout value, handling string 'None' from YAML."""
            if value is None or (isinstance(value, str) and value.lower() == 'none'):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        startup_timeout = params.get('startup_server_timeout_sec')

        return cls(
            startup_server_timeout_sec=_convert_timeout_value(startup_timeout),
            runtime_retry_timeout_sec=params.get('runtime_retry_timeout_sec', 10.0),
            server_check_interval_sec=params.get('server_check_interval_sec', 20.0)
        )


class BehaviorTreeConfig:
    """Base configuration class for behavior tree parameters."""

    def __init__(
        self,
        behavior_tree_params_file: str,
        package_name: str
    ):
        """
        Initialize base behavior tree configuration.

        Args:
        ----
        behavior_tree_params_file (str): Path to the behavior tree parameters YAML file.
        package_name (str): Name of the package to load parameters from.

        """
        self.params = load_yaml_params(
            behavior_tree_params_file,
            package_name=package_name
        )
        self.behavior_tree_params = self.params.get('behavior_tree_params', {})


class MultiObjPickPlaceConfig(BehaviorTreeConfig):
    """Configuration for multi-object pick and place behavior tree."""

    plan_to_grasp: PlanToGraspConfig
    plan_to_pose: PlanToPoseConfig
    read_grasp_poses: ReadGraspPosesConfig
    close_gripper: CloseGripperConfig
    open_gripper: OpenGripperConfig
    attach_object: ObjectAttachDetachConfig
    detach_object: ObjectAttachDetachConfig
    execute_trajectory: ExecuteTrajectoryConfig
    interactive_marker: InteractiveMarkerConfig
    detect_object: DetectObjectConfig
    stale_detection: StaleDetectionConfig
    mesh_assigner: MeshAssignerConfig
    assign_object_name: AssignObjectNameConfig
    publish_static_planning_scene: PublishStaticPlanningSceneConfig
    object_selector: ObjectSelectorConfig
    pose_estimation: PoseEstimationConfig
    arm_controllers: SwitchControllersConfig
    tool_controllers: SwitchControllersConfig
    retry_config: RetryConfig
    server_timeout_config: ServerTimeoutConfig

    def __init__(
        self,
        behavior_tree_params_file: str,
        package_name: str,
        frame_prefix: str = '',
    ):
        """
        Initialize multi-object pick and place behavior tree configuration.

        Args:
        ----
        behavior_tree_params_file (str): Path to the behavior tree parameters YAML file.
        package_name (str): Name of the package to load parameters from.
        frame_prefix (str): TF prefix from the launch chain, prepended to
            attach/detach gripper/grasp frame names. ``''`` for robots that
            publish bare frame names (e.g. UR).

        """
        super().__init__(behavior_tree_params_file, package_name)

        multi_obj_params = self.behavior_tree_params.get(
            'multi_object_pick_and_place', {})

        self.plan_to_grasp = PlanToGraspConfig.from_dict(
            multi_obj_params.get('plan_to_grasp', {}))
        self.plan_to_pose = PlanToPoseConfig.from_dict(
            multi_obj_params.get('plan_to_pose', {}))
        self.read_grasp_poses = ReadGraspPosesConfig.from_dict(
            multi_obj_params.get('read_grasp_poses', {}))
        self.close_gripper = CloseGripperConfig.from_dict(
            multi_obj_params.get('close_gripper', {}))
        self.open_gripper = OpenGripperConfig.from_dict(
            multi_obj_params.get('open_gripper', {}))
        self.attach_object = ObjectAttachDetachConfig.from_dict(
            multi_obj_params.get('attach_object', {}),
            frame_prefix=frame_prefix,
        )
        self.detach_object = ObjectAttachDetachConfig.from_dict(
            multi_obj_params.get('detach_object', {}),
            frame_prefix=frame_prefix,
        )
        self.execute_trajectory = ExecuteTrajectoryConfig.from_dict(
            multi_obj_params.get('execute_trajectory', {}))
        self.interactive_marker = InteractiveMarkerConfig.from_dict(
            multi_obj_params.get('interactive_marker', {}))
        self.detect_object = DetectObjectConfig.from_dict(
            multi_obj_params.get('detect_object', {}))
        self.stale_detection = StaleDetectionConfig.from_dict(
            multi_obj_params.get('stale_detection', {}))
        self.mesh_assigner = MeshAssignerConfig.from_dict(
            multi_obj_params.get('mesh_assigner', {}))
        self.assign_object_name = AssignObjectNameConfig.from_dict(
            multi_obj_params.get('assign_object_name', {}))
        self.publish_static_planning_scene = PublishStaticPlanningSceneConfig.from_dict(
            multi_obj_params.get('publish_static_planning_scene', {}))
        self.object_selector = ObjectSelectorConfig.from_dict(
            multi_obj_params.get('object_selector', {}))
        self.pose_estimation = PoseEstimationConfig.from_dict(
            multi_obj_params.get('pose_estimation', {}))

        # Parse switch_controllers with arm and tool subsections
        switch_controllers_params = multi_obj_params.get('switch_controllers', {})
        self.arm_controllers = SwitchControllersConfig.from_dict(
            switch_controllers_params.get('arm', {}))
        self.tool_controllers = SwitchControllersConfig.from_dict(
            switch_controllers_params.get('tool', {}))

        self.retry_config = RetryConfig.from_dict(
            multi_obj_params.get('retry_config', {}))
        self.server_timeout_config = ServerTimeoutConfig.from_dict(
            multi_obj_params.get('server_timeout_config', {}))


class BlackboardConfig:
    """Configuration for blackboard parameters."""

    max_num_next_object: int
    use_drop_pose_from_rviz: bool
    object_frame_name: str
    abort_motion: bool
    publish_grasp_poses: bool
    home_pose: Pose
    selected_object_id: Optional[int]
    active_obj_id: Optional[int]
    goal_drop_pose: Optional[Pose]
    rviz_drop_pose: Optional[Pose]
    object_info_cache: Optional[Any]
    next_object_id: collections.deque
    target_poses: List[Pose]
    class_ids: List[str]
    mode: int
    supported_objects_config: SupportedObjectsConfig
    workflow_status: int
    workflow_summary: str

    def __init__(
        self,
        blackboard_params_file: str,
        package_name: str
    ):
        """
        Initialize blackboard configuration from YAML parameters file.

        Args:
        ----
        blackboard_params_file (str): Path to the blackboard parameters YAML file.
        package_name (str): Name of the package to load parameters from.

        """
        self.params = load_yaml_params(
            blackboard_params_file,
            package_name=package_name
        )
        self.blackboard_params = self.params.get('blackboard_params', {})

        self.max_num_next_object = self.blackboard_params.get(
            'max_num_next_object', 2
        )
        self.use_drop_pose_from_rviz = self.blackboard_params.get(
            'use_drop_pose_from_rviz', False
        )
        self.object_frame_name = self.blackboard_params.get(
            'object_frame_name', 'detected_object1'
        )

        self.abort_motion = self.blackboard_params.get('abort_motion', False)
        self.max_planning_tries = self.blackboard_params.get(
            'max_planning_tries', 3)
        self.publish_grasp_poses = self.blackboard_params.get(
            'publish_grasp_poses', True)

        self.home_pose = self._create_pose_from_params(
            self.blackboard_params.get('home_pose', [
                -0.25, 0.45, 0.50, -0.677772, 0.734752, 0.020993, 0.017994
            ])
        )

        # Process target_poses from YAML config (convert arrays to Pose objects)
        target_poses_params = self.blackboard_params.get('target_poses', [])
        self.target_poses = []
        for pose_params in target_poses_params:
            if pose_params:  # Skip empty entries
                self.target_poses.append(self._create_pose_from_params(pose_params))

        # Process class_ids from YAML config
        self.class_ids = self.blackboard_params.get('class_ids', [])

        # Process mode from YAML config
        self.mode = self.blackboard_params.get('mode', 0)

        self.selected_object_id = self.blackboard_params.get(
            'selected_object_id', None)

        self.active_obj_id = self.blackboard_params.get('active_obj_id', None)

        goal_drop_pose_params = self.blackboard_params.get(
            'goal_drop_pose', None)
        if goal_drop_pose_params:
            self.goal_drop_pose = self._create_pose_from_params(
                goal_drop_pose_params)
        else:
            self.goal_drop_pose = None

        rviz_drop_pose_params = self.blackboard_params.get(
            'rviz_drop_pose', None)
        if rviz_drop_pose_params:
            self.rviz_drop_pose = self._create_pose_from_params(
                rviz_drop_pose_params)
        else:
            self.rviz_drop_pose = None

        self.object_info_cache = self.blackboard_params.get(
            'object_info_cache', None)

        next_object_ids = self.blackboard_params.get('next_object_id', [])
        self.next_object_id = collections.deque(next_object_ids)

        supported_objects = self.blackboard_params.get(
            'supported_objects', {})
        self.supported_objects_config = SupportedObjectsConfig.from_dict(
            supported_objects)

        # Status reporting variables
        self.workflow_status = self.blackboard_params.get(
            'workflow_status', MultiObjectPickAndPlace.Result.UNKNOWN)
        self.workflow_summary = self.blackboard_params.get('workflow_summary', '')

    def _create_pose_from_params(self, pose_params: List) -> Pose:
        """
        Create a ROS Pose from a list of parameters.

        Args
        ----
            pose_params (List): List of 7 numeric values representing [x, y, z, qx, qy, qz, qw].

        Raises
        ------
            ValueError: If pose_params does not contain exactly 7 elements.

        Returns
        -------
            Pose: ROS geometry_msgs/Pose message with position and orientation set.

        """
        if len(pose_params) != 7:
            raise ValueError(
                f'Expected 7 elements (xyz + quaternion) for Pose, '
                f'got {len(pose_params)}: {pose_params}'
            )
        pose = Pose()
        pose.position.x = float(pose_params[0])
        pose.position.y = float(pose_params[1])
        pose.position.z = float(pose_params[2])
        pose.orientation.x = float(pose_params[3])
        pose.orientation.y = float(pose_params[4])
        pose.orientation.z = float(pose_params[5])
        pose.orientation.w = float(pose_params[6])
        return pose
