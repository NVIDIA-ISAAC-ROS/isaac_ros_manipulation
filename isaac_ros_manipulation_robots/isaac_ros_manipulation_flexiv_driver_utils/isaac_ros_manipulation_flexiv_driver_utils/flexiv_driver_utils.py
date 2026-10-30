# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, List, Tuple

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_flexiv_driver_utils.config import (
    FlexivRizonDriverConfig
)
from isaac_ros_manipulation_flexiv_driver_utils.prefix_utils import (
    apply_calibrated_joint_origins,
    apply_joint_prefix,
)
from isaac_ros_manipulation_flexiv_driver_utils.robot_description import (
    get_robot_description_contents_for_real,
    get_robot_description_contents_for_sim,
    get_srdf_contents_for_real,
)
from isaac_ros_manipulation_robot_utils.robot_controller_base import (
    RobotControllerBase,
)

from launch.actions import (
    IncludeLaunchDescription, RegisterEventHandler, Shutdown, TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

import yaml


# Third-party Flexiv packages the real-robot launch leans on.
FLEXIV_DESCRIPTION_PKG = 'flexiv_description'
FLEXIV_MOVEIT_CONFIG_PKG = 'flexiv_moveit_config'
FLEXIV_GRIPPER_PKG = 'flexiv_gripper'
CUMOTION_DESCRIPTION_PKG = 'isaac_ros_cumotion_robot_description'
CUMOTION_MOVEIT_PKG = 'isaac_ros_cumotion_moveit'


def load_cumotion_config() -> Dict:
    """Load the cuMotion planning pipeline yaml from ``isaac_ros_cumotion_moveit``."""
    config_file_path = os.path.join(
        get_package_share_directory(CUMOTION_MOVEIT_PKG),
        'config', 'isaac_ros_cumotion_planning.yaml',
    )
    with open(config_file_path) as config_file:
        return yaml.safe_load(config_file)


def load_yaml_with_robot_sn(
    package_name: str, file_path: str, robot_sn: str,
) -> Dict:
    """
    Load a YAML file from a package share dir, substituting ``$(var robot_sn)``.

    Mirrors the loader the real-robot launch previously defined inline: treats
    an empty YAML document as a hard error so MoveIt never silently receives
    ``None`` where it expects a dict.
    """
    abs_path = os.path.join(
        get_package_share_directory(package_name), file_path)
    with open(abs_path, 'r') as f:
        content = f.read()
    if robot_sn:
        content = content.replace('$(var robot_sn)', robot_sn)
    parsed = yaml.safe_load(content)
    if parsed is None:
        raise ValueError(
            f"YAML file '{abs_path}' (from package '{package_name}') "
            f'parsed to an empty document.'
        )
    return parsed


class FlexivDriverUtils(RobotControllerBase):
    """
    Flexiv Rizon implementation of :class:`RobotControllerBase`.

    Single home for the three ABC methods (``get_robot_state_publisher``,
    ``get_moveit_group_node``, ``get_robot_control_nodes``) plus the
    real-robot orchestration helpers (static TFs, gripper include,
    RViz gating, event-handler sequencing). Each ABC method branches on
    ``driver_config.use_sim_time`` so the sim and real launch files both
    look like: build config -> instantiate FlexivDriverUtils -> call methods.

    Real-robot-only helpers (``get_real_*``) deliberately live on this class
    rather than in the launch file so the launch file stays declarative and
    custom-robot integrators can see all the moving parts in one place.
    """

    def __init__(self, driver_config: FlexivRizonDriverConfig):
        super().__init__(driver_config)
        # Deferred spawners wired by ``get_robot_control_nodes`` on the real
        # path; ``get_real_event_handlers`` reads them back by name.
        self._real_deferred_spawners: Dict[str, Node] = {}

    # ------------------------------------------------------------------
    # RobotControllerBase implementation
    # ------------------------------------------------------------------

    def get_robot_state_publisher(self) -> Node:
        """
        Return the ``robot_state_publisher`` for the Flexiv Rizon.

        Implements :meth:`RobotControllerBase.get_robot_state_publisher`.
        Sim path uses the Isaac-Sim-aware xacro shipped with
        ``isaac_ros_manipulation_flexiv_robot_description``; real path uses
        the third-party ``flexiv_description/urdf/rizon.urdf.xacro`` resolved
        via :func:`get_robot_description_contents_for_real`.

        Returns
        -------
            Node: Configured ``robot_state_publisher`` node.

        """
        driver_config = self.driver_config
        if driver_config.use_sim_time:
            return self._get_sim_robot_state_publisher()
        return self._get_real_robot_state_publisher()

    def get_moveit_group_node(self) -> Tuple[Node, Any]:
        """
        Return the MoveIt ``move_group`` node and the matching config bundle.

        Implements :meth:`RobotControllerBase.get_moveit_group_node`. The
        returned bundle's type depends on the code path:

        * sim: :class:`moveit_configs_utils.MoveItConfigsBuilder` instance,
          consumable by
          :func:`isaac_ros_manipulation_ros_python_utils.core.get_visualization_actions`.
        * real: plain ``dict`` with ``robot_description``,
          ``robot_description_semantic``, ``robot_description_kinematics``,
          and ``robot_description_planning`` entries for RViz param reuse.

        Returns
        -------
            Tuple[Node, Any]: The ``move_group`` node and its MoveIt config
            bundle.

        """
        driver_config = self.driver_config
        if driver_config.use_sim_time:
            return self._get_sim_moveit_group_node()
        return self._get_real_moveit_group_node()

    def get_robot_control_nodes(self) -> List[Node]:
        """
        Return the ``ros2_control`` node(s) and controller spawners for Flexiv.

        Implements :meth:`RobotControllerBase.get_robot_control_nodes`. The
        real path additionally caches the deferred spawners
        (arm + streaming controller) on ``self`` so
        :meth:`get_real_event_handlers` can sequence them after the joint
        state broadcaster comes up.

        Returns
        -------
            List[Node]: Non-deferred nodes the launch file should add
            directly to its ``LaunchDescription``. Deferred real spawners
            are *not* in this list; they are returned by
            :meth:`get_real_event_handlers`.

        """
        driver_config = self.driver_config
        if driver_config.use_sim_time:
            return self._get_sim_robot_control_nodes()
        return self._get_real_robot_control_nodes()

    # ------------------------------------------------------------------
    # Real-robot-only orchestration
    # ------------------------------------------------------------------

    def apply_real_cumotion_urdf_prefix(self) -> None:
        """
        Pre-process the cuMotion URDF/XRDF so joint names match Flexiv's prefixing.

        The real-robot Flexiv URDF prefixes every joint/link with
        ``<robot_sn>_`` (e.g. ``Rizon4s-062839_base_link``). cuMotion expects
        joint names to match the runtime TF tree, so copy the unprefixed
        package-shipped URDF/XRDF into ``/tmp/`` with the prefix applied.

        No-ops when ``driver_config.robot_sn`` is empty.
        """
        robot_sn = self.driver_config.robot_sn
        if not robot_sn:
            return
        cumotion_desc_dir = get_package_share_directory(
            CUMOTION_DESCRIPTION_PKG)
        prefix = f'{robot_sn}_'
        urdf_template = os.path.join(
            cumotion_desc_dir, 'urdf', 'flexiv_rizon4s_grav.urdf')

        calibrated_kinematics = self.driver_config.calibrated_kinematics_path
        if calibrated_kinematics and os.path.isfile(calibrated_kinematics):
            urdf_template = apply_calibrated_joint_origins(
                urdf_template,
                calibrated_kinematics,
                '/tmp/flexiv_rizon4s_grav_calibrated.urdf',
            )

        apply_joint_prefix(
            urdf_template,
            prefix,
            '/tmp/flexiv_rizon4s_grav_prefixed.urdf',
        )
        apply_joint_prefix(
            os.path.join(cumotion_desc_dir, 'xrdf', 'flexiv_rizon4s_grav.xrdf'),
            prefix,
            '/tmp/flexiv_rizon4s_grav_prefixed.xrdf',
        )

    def get_real_static_tfs(self) -> List[Node]:
        """
        Return the static TF publishers required for the Isaac Manipulation stack.

        The third-party Flexiv URDF (a) prefixes every link with
        ``<robot_sn>_`` and (b) omits ``gripper_frame``, ``grasp_frame``, and
        ``insertion_frame``. The Isaac Manipulation pipeline (cuMotion,
        pick-and-place orchestrator) expects an unprefixed ``base_link`` plus
        those three frames. Bridge the gap with four identity/offset static
        transforms that match the offsets defined in
        ``isaac_ros_manipulation_flexiv_robot_description/urdf/rizon_grav.urdf.xacro``.

        Returns
        -------
            List[Node]: Four ``tf2_ros`` ``static_transform_publisher`` nodes.

        """
        driver_config = self.driver_config
        prefix = driver_config.frame_prefix
        flange_frame = f'{prefix}flange'

        world_to_base_link_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_base_link_publisher',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', 'world',
                       '--child-frame-id', 'base_link'],
            output='log',
        )
        gripper_frame_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gripper_frame_publisher',
            arguments=['--x', '0', '--y', '0', '--z', '0',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', flange_frame,
                       '--child-frame-id', f'{prefix}gripper_frame'],
            output='log',
        )
        grasp_frame_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='grasp_frame_publisher',
            arguments=['--x', '0', '--y', '0', '--z', '0.20',
                       '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
                       '--frame-id', f'{prefix}gripper_frame',
                       '--child-frame-id', f'{prefix}grasp_frame'],
            output='log',
        )
        insertion_frame_tf = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='insertion_frame_publisher',
            arguments=['--x', '0', '--y', '0', '--z', '0.20',
                       '--qx', '0', '--qy', '0.99999968',
                       '--qz', '0', '--qw', '0.0007963',
                       '--frame-id', f'{prefix}gripper_frame',
                       '--child-frame-id', f'{prefix}insertion_frame'],
            output='log',
        )
        return [
            world_to_base_link_tf,
            gripper_frame_tf,
            grasp_frame_tf,
            insertion_frame_tf,
        ]

    def get_real_gripper_launch(self) -> IncludeLaunchDescription:
        """
        Return the Grav gripper launch include, gated on ``load_gripper``.

        Reuses ``flexiv_gripper/launch/flexiv_gripper.launch.py`` and forwards
        ``robot_sn`` / ``gripper_name`` / ``use_fake_hardware``.

        Returns
        -------
            IncludeLaunchDescription: The gripper launch include.

        """
        driver_config = self.driver_config
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare(FLEXIV_GRIPPER_PKG),
                    'launch', 'flexiv_gripper.launch.py',
                ])
            ),
            launch_arguments={
                'robot_sn': driver_config.robot_sn,
                'gripper_name': driver_config.gripper_name,
                'use_fake_hardware': driver_config.use_fake_hardware,
            }.items(),
            condition=IfCondition(driver_config.load_gripper),
        )

    def get_real_rviz_nodes(self, moveit_config: Dict) -> List[Node]:
        """
        Return the two RViz nodes (MoveIt view + workflow view) gated on launch args.

        Args
        ----
            moveit_config (Dict): The bundle returned alongside the
                ``move_group`` node by :meth:`get_moveit_group_node` on the
                real path; supplies ``robot_description``,
                ``robot_description_semantic``, and
                ``robot_description_kinematics`` params to RViz.

        Returns
        -------
            List[Node]: ``[rviz2_moveit, rviz2_workflow]``.

        """
        driver_config = self.driver_config
        moveit_rviz_config = PathJoinSubstitution(
            [FindPackageShare(FLEXIV_MOVEIT_CONFIG_PKG), 'rviz', 'moveit.rviz']
        )
        rviz_moveit_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_moveit',
            output='log',
            arguments=['-d', moveit_rviz_config],
            parameters=[
                moveit_config['robot_description'],
                moveit_config['robot_description_semantic'],
                moveit_config['robot_description_kinematics'],
                moveit_config['robot_description_planning'],
            ],
            condition=IfCondition(driver_config.start_rviz),
        )
        rviz_workflow_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', driver_config.rviz_config_file],
            parameters=[
                moveit_config['robot_description'],
                moveit_config['robot_description_semantic'],
                moveit_config['robot_description_kinematics'],
            ],
            condition=IfCondition(driver_config.enable_rviz_visualization),
        )
        return [rviz_moveit_node, rviz_workflow_node]

    def get_real_event_handlers(
        self,
        move_group_node: Node,
        rviz_nodes: List[Node],
        gripper_launch: IncludeLaunchDescription,
    ) -> List[RegisterEventHandler]:
        """
        Build the event handlers that sequence the real-robot bring-up.

        The Flexiv gripper requires the robot to be in IDLE mode for
        ``Tool::Switch()`` / ``Gripper::Init()``. Activating the arm
        controller switches the robot out of IDLE into
        ``NRT_JOINT_POSITION``, so the gripper must initialize first. The
        expected sequence is:

        1. ``ros2_control_node`` comes up and the robot becomes operational
           (IDLE).
        2. ``joint_state_broadcaster`` spawns.
        3. After the broadcaster: launch the gripper (``Init`` takes ~10s,
           robot still IDLE).
        4. 15s after the broadcaster: spawn the arm controller (switches
           mode).
        5. After the arm controller: spawn the streaming controller (inactive)
           and bring up ``move_group`` + RViz.

        Args
        ----
            move_group_node (Node): Deferred node returned by
                :meth:`get_moveit_group_node`.
            rviz_nodes (List[Node]): Deferred nodes returned by
                :meth:`get_real_rviz_nodes`.
            gripper_launch (IncludeLaunchDescription): Deferred include
                returned by :meth:`get_real_gripper_launch`.

        Returns
        -------
            List[RegisterEventHandler]: Seven event handlers wiring the
            sequence described above.

        Raises
        ------
            RuntimeError: If :meth:`get_robot_control_nodes` was not called
                first; the deferred spawners must be cached before event
                handlers can reference them.

        """
        spawners = self._real_deferred_spawners
        if not spawners:
            raise RuntimeError(
                'get_robot_control_nodes() must be called before '
                'get_real_event_handlers(); the deferred real-robot '
                'spawners have not been built yet.'
            )
        joint_state_broadcaster = spawners['joint_state_broadcaster']
        robot_controller = spawners['robot_controller']
        streaming_controller = spawners['streaming_controller']

        delay_gripper_after_broadcaster = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[gripper_launch],
            )
        )
        delay_robot_controller = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[TimerAction(
                    period=15.0,
                    actions=[robot_controller],
                )],
            )
        )
        delay_streaming_controller = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_controller,
                on_exit=[streaming_controller],
            )
        )
        delay_cartesian_motion_controller = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_controller,
                on_exit=[spawners['cartesian_motion_controller']],
            )
        )
        delay_move_group = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_controller,
                on_exit=[move_group_node],
            )
        )
        rviz_moveit_node, rviz_workflow_node = rviz_nodes
        delay_rviz = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_controller,
                on_exit=[rviz_moveit_node],
            )
        )
        delay_visualization_rviz = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=robot_controller,
                on_exit=[rviz_workflow_node],
            )
        )
        return [
            delay_gripper_after_broadcaster,
            delay_robot_controller,
            delay_streaming_controller,
            delay_cartesian_motion_controller,
            delay_move_group,
            delay_rviz,
            delay_visualization_rviz,
        ]

    # ------------------------------------------------------------------
    # Sim path implementations
    # ------------------------------------------------------------------

    def _get_sim_robot_state_publisher(self) -> Node:
        driver_config = self.driver_config
        robot_description_contents = get_robot_description_contents_for_sim(
            urdf_xacro_file=driver_config.urdf_path,
            rizon_type=driver_config.rizon_type,
            use_sim_time=driver_config.use_sim_time,
        )
        remappings = [
            ('/joint_states',
             driver_config.remapped_joint_states['/joint_states'])
        ]
        return Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'robot_description': robot_description_contents,
                 'publish_robot_description': True,
                 'use_sim_time': driver_config.use_sim_time}
            ],
            remappings=remappings,
            on_exit=Shutdown(),
        )

    def _get_sim_moveit_group_node(self) -> Tuple[Node, Any]:
        driver_config = self.driver_config
        robot_description_content = get_robot_description_contents_for_sim(
            urdf_xacro_file=driver_config.urdf_path,
            rizon_type=driver_config.rizon_type,
            use_sim_time=driver_config.use_sim_time,
        )
        moveit_config = (
            MoveItConfigsBuilder(
                'rizon_with_gripper',
                package_name='isaac_ros_manipulation_flexiv_robot_description')
            .robot_description_semantic(file_path=driver_config.srdf_path)
            .robot_description_kinematics(
                file_path=driver_config.kinematics_file_path)
            .joint_limits(file_path=driver_config.joint_limits_file_path)
            .trajectory_execution(
                file_path=driver_config.moveit_controllers_file_path)
            .planning_pipelines(pipelines=['ompl'])
            .to_moveit_configs()
        )
        cumotion_config = load_cumotion_config()
        moveit_config.planning_pipelines[
            'planning_pipelines'].insert(0, 'isaac_ros_cumotion')
        moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
        moveit_config.planning_pipelines[
            'default_planning_pipeline'] = 'isaac_ros_cumotion'
        moveit_config.robot_description = {
            'robot_description': robot_description_content}
        move_it_dict = moveit_config.to_dict()
        move_it_dict['planning_pipelines'] = {
            'pipeline_names': ['isaac_ros_cumotion'],
        }
        move_group_node = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                move_it_dict,
                {'use_sim_time': driver_config.use_sim_time}
            ],
            arguments=['--ros-args', '--log-level', 'info'],
            remappings=[('joint_states', '/rizon_parsed_joint_states')],
            on_exit=Shutdown(),
        )
        return move_group_node, moveit_config

    def _get_sim_robot_control_nodes(self) -> List[Node]:
        driver_config = self.driver_config
        robot_description = {
            'robot_description': get_robot_description_contents_for_sim(
                urdf_xacro_file=driver_config.urdf_path,
                rizon_type=driver_config.rizon_type,
                use_sim_time=driver_config.use_sim_time,
            )
        }
        ros2_control_node = Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                robot_description,
                ParameterFile(
                    driver_config.ros2_controllers_file_path,
                    allow_substs=True),
                {'use_sim_time': driver_config.use_sim_time}
            ],
            remappings=[
                (
                    '/controller_manager/robot_description',
                    driver_config.remapped_joint_states[
                        '/controller_manager/robot_description'],
                )
            ],
            arguments=['--ros-args', '--log-level', 'error'],
            output='screen',
            on_exit=Shutdown(),
        )
        scaled_joint_trajectory_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[
                ParameterFile(driver_config.ros2_controllers_file_path, allow_substs=True)
            ],
            arguments=[
                'scaled_joint_trajectory_controller',
                '-c', '/controller_manager'
            ],
        )
        joint_state_broadcaster_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[
                ParameterFile(driver_config.ros2_controllers_file_path, allow_substs=True)
            ],
            arguments=[
                'joint_state_broadcaster',
                '--controller-manager', '/controller_manager',
            ],
        )
        delay_scaled_controller_after_broadcaster = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[scaled_joint_trajectory_controller_spawner],
            )
        )

        return [
            ros2_control_node,
            joint_state_broadcaster_spawner,
            delay_scaled_controller_after_broadcaster,
        ]

    # ------------------------------------------------------------------
    # Real path implementations
    # ------------------------------------------------------------------

    def _get_real_robot_state_publisher(self) -> Node:
        driver_config = self.driver_config
        robot_description_contents = get_robot_description_contents_for_real(
            driver_config)
        return Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='both',
            parameters=[{'robot_description': robot_description_contents}],
        )

    def _get_real_moveit_group_node(self) -> Tuple[Node, Dict]:
        driver_config = self.driver_config
        robot_description = {
            'robot_description': get_robot_description_contents_for_real(
                driver_config)
        }
        robot_description_semantic = {
            'robot_description_semantic': get_srdf_contents_for_real(
                driver_config)
        }
        robot_description_kinematics = load_yaml_with_robot_sn(
            FLEXIV_MOVEIT_CONFIG_PKG,
            'config/kinematics.yaml',
            driver_config.robot_sn,
        )
        joint_limits_yaml = {
            'robot_description_planning': load_yaml_with_robot_sn(
                FLEXIV_MOVEIT_CONFIG_PKG,
                'config/joint_limits.yaml',
                driver_config.robot_sn,
            )
        }
        planning_pipelines = {
            'planning_pipelines': {
                'pipeline_names': ['isaac_ros_cumotion'],
            },
            'isaac_ros_cumotion': load_cumotion_config(),
        }
        moveit_simple_controllers_yaml = load_yaml_with_robot_sn(
            FLEXIV_MOVEIT_CONFIG_PKG,
            'config/moveit_controllers.yaml',
            driver_config.robot_sn,
        )
        moveit_controllers = {
            'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
            'moveit_controller_manager':
                'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        }
        trajectory_execution = {
            'moveit_manage_controllers': False,
            'trajectory_execution.allowed_execution_duration_scaling': 1.2,
            'trajectory_execution.allowed_goal_duration_margin': 0.5,
            'trajectory_execution.allowed_start_tolerance': 0.05,
        }
        planning_scene_monitor_parameters = {
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
        }
        move_group_node = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                {'publish_robot_description_semantic': True},
                robot_description_kinematics,
                joint_limits_yaml,
                planning_pipelines,
                trajectory_execution,
                moveit_controllers,
                planning_scene_monitor_parameters,
            ],
        )
        moveit_config: Dict[str, Any] = {
            'robot_description': robot_description,
            'robot_description_semantic': robot_description_semantic,
            'robot_description_kinematics': robot_description_kinematics,
            'robot_description_planning': joint_limits_yaml,
        }
        return move_group_node, moveit_config

    def _get_real_robot_control_nodes(self) -> List[Node]:
        driver_config = self.driver_config
        robot_description = {
            'robot_description': get_robot_description_contents_for_real(
                driver_config)
        }
        robot_controllers = PathJoinSubstitution([
            FindPackageShare('isaac_ros_manipulation_flexiv_robot_description'),
            'config', 'controllers.yaml',
        ])
        ros2_control_node = Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                robot_description,
                ParameterFile(robot_controllers, allow_substs=True),
                {'robot_sn': driver_config.robot_sn},
                {'rdk_control_mode': driver_config.rdk_control_mode},
            ],
            remappings=[('joint_states', 'flexiv_arm/joint_states')],
            output='both',
        )
        joint_state_publisher_node = Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{
                'source_list': [
                    'flexiv_arm/joint_states',
                    'flexiv_gripper_node/gripper_joint_states',
                ],
                'rate': 30,
            }],
        )
        # The spawner forwards this config to the controller node; without it the
        # controller starts with no 'joints' and fails to initialize.
        # The ros2_control_node remapping does not reach controller nodes, so the remap is passed
        # to the broadcaster here. Without it the broadcaster publishes /joint_states directly,
        # joint_state_publisher never receives arm data, and /joint_states interleaves the real
        # arm state with an all-zeros arm at 30 Hz.
        joint_state_broadcaster_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[ParameterFile(robot_controllers, allow_substs=True)],
            arguments=[
                'joint_state_broadcaster',
                '--controller-manager', '/controller_manager',
                '--controller-ros-args', '-r joint_states:=flexiv_arm/joint_states',
            ],
        )
        robot_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[ParameterFile(robot_controllers, allow_substs=True)],
            arguments=[
                'rizon_arm_controller',
                '--controller-manager', '/controller_manager',
            ],
        )
        # Streaming controller loads inactive so switch_controller can
        # activate it at runtime (e.g. during RL insertion).
        streaming_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[ParameterFile(robot_controllers, allow_substs=True)],
            arguments=[
                'streaming_position_controller',
                '--controller-manager', '/controller_manager',
                '--inactive',
            ],
        )
        # Loaded only when the driver is prepared to borrow
        # RT_CARTESIAN_MOTION_FORCE at runtime. Spawned ``--inactive`` so it
        # claims no command interfaces until the insertion orchestrator
        # explicitly switches to it for the task-space policy; the arm stays
        # under the joint trajectory controller for cuMotion up to that point.
        cartesian_motion_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            parameters=[ParameterFile(robot_controllers, allow_substs=True)],
            arguments=[
                'cartesian_motion_controller',
                '--controller-manager', '/controller_manager',
                '--inactive',
            ],
            condition=IfCondition(driver_config.runtime_cartesian_switching),
        )
        flexiv_robot_states_broadcaster_spawner = Node(
            package='controller_manager',
            executable='spawner',
            arguments=['flexiv_robot_states_broadcaster'],
            parameters=[{'robot_sn': driver_config.robot_sn}],
            condition=UnlessCondition(driver_config.use_fake_hardware),
        )
        gpio_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'gpio_controller',
                '--controller-manager', '/controller_manager',
            ],
            parameters=[{'robot_sn': driver_config.robot_sn}],
            condition=UnlessCondition(driver_config.use_fake_hardware),
        )

        # Cache deferred spawners for get_real_event_handlers() to wire.
        self._real_deferred_spawners = {
            'joint_state_broadcaster': joint_state_broadcaster_spawner,
            'robot_controller': robot_controller_spawner,
            'streaming_controller': streaming_controller_spawner,
            'cartesian_motion_controller': cartesian_motion_controller_spawner,
        }

        return [
            ros2_control_node,
            joint_state_publisher_node,
            joint_state_broadcaster_spawner,
            flexiv_robot_states_broadcaster_spawner,
            gpio_controller_spawner,
        ]


def get_isaac_sim_joint_parser_node(use_sim_time: bool) -> Node:
    """Return Isaac Sim joint parser node for the Flexiv Rizon."""
    return Node(
        package='isaac_ros_manipulation_flexiv_isaac_sim_utils',
        executable='isaac_sim_joint_parser_node.py',
        name='joint_parser',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        on_exit=Shutdown(),
    )


def get_grav_gripper_node(use_sim_time: bool) -> Node:
    """Return the Grav gripper action server for Isaac Sim."""
    return Node(
        package='isaac_ros_manipulation_flexiv_isaac_sim_utils',
        executable='isaac_sim_grav_gripper_driver.py',
        name='isaac_sim_grav_gripper_action_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        on_exit=Shutdown(),
    )
