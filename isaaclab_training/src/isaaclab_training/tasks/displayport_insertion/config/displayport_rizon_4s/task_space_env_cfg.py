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

"""Task-space DisplayPort insertion environment for the Flexiv Rizon 4S + Grav gripper.

Uses Operational Space Control (OSC) with 6D-rotation observations and an at-goal
reset curriculum. The end-effector pose is commanded as a relative Cartesian delta.
"""

import math
import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG
import isaaclab_training.mdp as mdp
import isaaclab_training.mdp.terminations as cable_terminations
from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
    DisplayportInsertionEnvCfg,
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
)

from .agents.rsl_rl_ppo_cfg import TASK_SPACE_NUM_STEPS_PER_ENV
from .joint_pos_env_cfg import (
    _PLUG_ROOT,
    _PLUG_ROT,
    _SOCKET_ROOT,
    _SOCKET_ROT,
    set_finger_joint_pos_grav,
)

# Rizon 4S arm joint names.
_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# OSC task-space gains. d_gain = 2 * sqrt(p_gain) * damping_ratio.
_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_DAMPING_RATIO_TRANS = 35.0 / (2.0 * math.sqrt(300.0))
_DAMPING_RATIO_ROT = 1.1 / (2.0 * math.sqrt(30.0))
_DAMPING_RATIO = (
    _DAMPING_RATIO_TRANS,
    _DAMPING_RATIO_TRANS,
    _DAMPING_RATIO_TRANS,
    _DAMPING_RATIO_ROT,
    _DAMPING_RATIO_ROT,
    _DAMPING_RATIO_ROT,
)

# Per-step Cartesian action scale applied to both position [m] and rotation [rad].
_ACTION_SCALE = 0.025

# DisplayPort blade engagement along the insertion axis at the seated pose [m].
_INSERTION_LENGTH = 0.011

# Physical seat of the plug relative to the socket keypoint, in the socket frame [m]
# (+X = out of the socket along the insertion axis, Z = wide lateral axis). Measured in sim:
# a free plug rests 1.84 mm above the keypoint and 0.45-0.80 mm towards -Z; the collision
# meshes overlap by only the 0.15 mm rest-offset allowance there (vs 1.77 mm at the keypoint).
# At-goal spawns are placed relative to this seat so they never start interpenetrating.
_SEAT_OFFSET = [0.00184, 0.0, -0.0006]

# Shallowest at-goal spawn along the insertion axis, measured from the socket keypoint [m].
_AT_GOAL_MAX_DEPTH = 0.015

# At-goal anneal length, in env steps: the first 333 of the 1,000 default 128-step iterations.
_AT_GOAL_ANNEAL_ENV_STEPS = 333 * 128

# Plug/socket solver position iterations. PhysX GPU runs the highest count in the scene for every
# body,
# so this sets the solver cost of the whole scene (128 -> 64: ~45 % less PhysX time per step).
_PLUG_SOCKET_SOLVER_POSITION_ITERATIONS = 64

# Plug mass properties exactly as PhysX derives them from the asset's collision shapes (including
# the
# disabled convex decompositions), measured with the PhysX rigid-body view: centre of mass [m] in
# the plug
# body frame, principal axes (w, x, y, z) and principal moments [kg m^2] for the 0.03 kg plug.
# Authoring
# them lets the disabled colliders be dropped without changing the plug's dynamics.
_PLUG_CENTER_OF_MASS = (0.0029111250769346952, 0.0008471199544146657, 0.015430336818099022)
_PLUG_PRINCIPAL_AXES_WXYZ = (0.9291425943374634, -0.013500147499144077, -0.3694750964641571,
                             1.6308516933349892e-05)
_PLUG_DIAGONAL_INERTIA = (3.6660012307437196e-06, 4.999659500369078e-06, 1.6410003934466692e-06)

# Replacement collision geometry (precomputed from the asset meshes, rigid-body frames; see
# collision/README.md):
# - socket Body8 (housing), ON: exact SDF for the top 12.5 mm (flange, top face, upper cavity walls
# -- everything the
#   plug reaches), one 64-vertex convex hull for the never-contacted lower part (5.3k instead of
# 43k SDF triangles).
#   Contact geometry unchanged; ~1 % faster, 6-20 s faster env creation.
# - plug casing as a 64-vertex convex hull (SDF kept only for the connector), OFF by default:
# +21-30 % simulation
#   throughput, but the hull deviates up to 0.16 mm from the curved casing the fingers grip, which
# shifts the plug
#   ~0.2 mm / ~0.7 deg in the gripper and jammed 2 of 8 tight-offset scripted insertions that the
# SDF casing seats.
#   Opt in with _USE_PLUG_CASING_HULL = True.
_USE_PLUG_CASING_HULL = False
_COLLISION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "collision")
_PLUG_COLLISION_PARTS = os.path.join(_COLLISION_DIR, "plug_connector_sdf_casing_hull.npz")
_SOCKET_COLLISION_PARTS = os.path.join(_COLLISION_DIR, "socket_body8_top_sdf_lower_hull.npz")
_SOCKET_BODY8_COLLIDER = "tn__2584N111_DisplayportCord_jP/Body8/Mesh"

# Gripper tool-center-point (TCP) offset from the flange body, in the flange's local
# frame [m]. The policy observes the TCP pose (where the plug is held), not the raw
# flange; the TCP shares the flange orientation, so only the position is offset.
_TCP_OFFSET = [0.0, 0.0, 0.2 - 0.0075]

##
# Observation configuration
##


@configclass
class TaskSpaceObservationsCfg:
    """Task-space observations with 6D rotation representation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: EEF pose + socket keypoint frame (18 dims)."""

        eef_pos = ObsTerm(
            func=mdp.eef_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "flange",
                "offset": _TCP_OFFSET
            },
        )
        eef_rot_6d = ObsTerm(
            func=mdp.eef_rot_6d_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "flange"
            },
        )
        socket_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("dp_socket"),
                "offset": SOCKET_INSERTION_OFFSET
            },
        )
        socket_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: joint state + both keypoint frames (32 dims)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS)},
        )
        socket_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("dp_socket"),
                "offset": SOCKET_INSERTION_OFFSET
            },
        )
        socket_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        plug_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("dp_plug"),
                "offset": PLUG_INSERTION_OFFSET
            },
        )
        plug_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Event configuration
##


@configclass
class TaskSpaceEventCfg:
    """Events for the task-space DisplayPort insertion environment."""

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    socket_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_socket", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
                "z": [-0.02, 0.02],
                "roll": [-math.radians(2.0), math.radians(2.0)],
                "pitch": [-math.radians(2.0), math.radians(2.0)],
                "yaw": [-math.radians(2.0), math.radians(2.0)],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    # At-goal reset curriculum: a fraction ``at_goal_prob`` of envs spawn the plug
    # already inserted at a random depth with goal orientation; the rest start from a
    # uniformly randomized approach pose. ``at_goal_prob`` anneals linearly from 0.8 to
    # 0.0 over the first 64k env steps (500 iterations at 128 steps) so the
    # policy learns the full approach. At-goal depths are measured from the physical seat
    # (``at_goal_seat_offset``) and span the same absolute 1.84-15 mm range as before.
    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.0,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": _AT_GOAL_ANNEAL_ENV_STEPS / TASK_SPACE_NUM_STEPS_PER_ENV,
            "num_steps_per_env": TASK_SPACE_NUM_STEPS_PER_ENV,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            "at_goal_depth_range": [0.0, _AT_GOAL_MAX_DEPTH - _SEAT_OFFSET[0]],
            "at_goal_seat_offset": _SEAT_OFFSET,
            "approach_depth_range": [0.02, 0.06],
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
            },
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [-0.0, 0.0]
            },
            "target_object_name": "dp_plug",
            "grasp_offset": [0.0, 0.0, 0.0],
        },
    )


##
# Terminations
##


@configclass
class TaskSpaceTerminationsCfg:
    """Termination terms for the task-space DisplayPort insertion."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    plug_dropped = DoneTerm(
        func=cable_terminations.reset_when_plug_dropped,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "distance_threshold": 0.15,
            "end_effector_body_name": "link7",
            "grasp_offset": [0.0, 0.0, 0.0],
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )

    plug_orientation_exceeded = DoneTerm(
        func=cable_terminations.reset_when_plug_orientation_exceeded,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "roll_threshold_deg": 15.0,
            "pitch_threshold_deg": 15.0,
            "yaw_threshold_deg": 180.0,
            "end_effector_body_name": "link7",
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )


##
# Main environment configuration
##


@configclass
class Rizon4sTaskSpaceDisplayportInsertionEnvCfg(DisplayportInsertionEnvCfg):
    """Task-space DisplayPort insertion with OSC control, 6D observations, and curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # EEF / grasp settings (shared with the joint-space variant).
        self.end_effector_body_name = "flange"
        self.num_arm_joints = 7
        self.grasp_offset = [0.0025, 0.0, -0.1875]
        self.grasp_rot_offset = [0.0, 0.0, 0.0, 1.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav

        self.plug_orientation_roll_threshold_deg = 15.0
        self.plug_orientation_pitch_threshold_deg = 15.0
        self.plug_orientation_yaw_threshold_deg = 180.0

        # ----- Observations: task-space with 6D rotation -----
        self.observations = TaskSpaceObservationsCfg()

        # ----- Actions: Operational Space Controller -----
        # The Deploy OSC action wrapper behaves identically to the stock OSC action at
        # train time and additionally exposes the scaled pose-delta contract used by the
        # LEAPP export (see scripts/debug/export_taskspace_policy_leapp.py).
        self.actions.arm_action = mdp.DeployOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="flange",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=False,
                motion_stiffness_task=_STIFFNESS,
                motion_damping_ratio_task=_DAMPING_RATIO,
                nullspace_control="none",
            ),
            position_scale=_ACTION_SCALE,
            orientation_scale=_ACTION_SCALE,
        )

        # ----- Events -----
        self.events = TaskSpaceEventCfg()

        # ----- Terminations -----
        self.terminations = TaskSpaceTerminationsCfg()

        self.terminations.plug_orientation_exceeded.params["roll_threshold_deg"] = (
            self.plug_orientation_roll_threshold_deg)
        self.terminations.plug_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.plug_orientation_pitch_threshold_deg)
        self.terminations.plug_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.plug_orientation_yaw_threshold_deg)

        # ----- Scene: robot -----
        self.scene.robot = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005,
                                                                 rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": math.radians(32.44),
                    "joint2": math.radians(-16.71),
                    "joint3": math.radians(-5.69),
                    "joint4": math.radians(128.38),
                    "joint5": math.radians(6.74),
                    "joint6": math.radians(55.95),
                    "joint7": math.radians(111.54),
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # OSC requires the arm joints to be pure torque-controlled. The stock Flexiv cfg
        # gives the arm a stiff implicit position PD that would hold each joint at its
        # reset target and override the OSC-computed efforts, so zero the arm PD gains
        # (damping is provided by the OSC via ``motion_damping_ratio_task``).
        for _arm_actuator in ("shoulder", "elbow", "wrist"):
            self.scene.robot.actuators[_arm_actuator].stiffness = 0.0
            self.scene.robot.actuators[_arm_actuator].damping = 0.0

        # Grav gripper actuators.
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        )

        # ----- Plug / socket physics -----
        for asset in (self.scene.dp_plug, self.scene.dp_socket):
            asset.spawn.rigid_props.solver_position_iteration_count = (
                _PLUG_SOCKET_SOLVER_POSITION_ITERATIONS
            )
            # Disabled colliders take no part in contact; dropping them saves PhysX parsing/cooking
            # per env.
            asset.spawn.remove_disabled_colliders = True
        # The plug's mass properties are pinned to what the disabled shapes produced. The socket is
        # kinematic,
        # so its mass properties (which shift slightly without its disabled SDF) do not enter the
        # dynamics.
        self.scene.dp_plug.spawn.center_of_mass = _PLUG_CENTER_OF_MASS
        self.scene.dp_plug.spawn.principal_axes_wxyz = _PLUG_PRINCIPAL_AXES_WXYZ
        self.scene.dp_plug.spawn.diagonal_inertia = _PLUG_DIAGONAL_INERTIA
        # Simplified collision geometry where it does not mate (mass properties above stay pinned).
        if _USE_PLUG_CASING_HULL:
            self.scene.dp_plug.spawn.collision_parts_file = _PLUG_COLLISION_PARTS
            self.scene.dp_plug.spawn.collision_parts_template = "collision_mesh"
            self.scene.dp_plug.spawn.remove_colliders = ("collision_mesh", )
        self.scene.dp_socket.spawn.collision_parts_file = _SOCKET_COLLISION_PARTS
        self.scene.dp_socket.spawn.collision_parts_template = _SOCKET_BODY8_COLLIDER
        self.scene.dp_socket.spawn.remove_colliders = (_SOCKET_BODY8_COLLIDER, )

        # ----- Workspace poses -----
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(pos=_SOCKET_ROOT,
                                                                         rot=_SOCKET_ROT)
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(pos=_PLUG_ROOT,
                                                                       rot=_PLUG_ROT)

        # Gripper widths for the DisplayPort plug.
        self.hand_grasp_width = 0.3
        self.hand_hold_width = -0.05
        self.hand_close_width = -0.155

        # Wire grasp-event params.
        self.events.set_robot_to_grasp_pose.params[
            "end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params[
            "gripper_joint_setter_func"] = self.gripper_joint_setter_func
        # Reset IK: DLS reaches ~3 um in ~5 iterations; the float32 world-coordinate floor is
        # up to ~15 um for envs far from the origin, so 1e-6 is never met and all iterations ran.
        # 1e-5 exits after 5-6 iterations (<= 11 um); 50 caps the rare floor-limited reset batch.
        self.events.set_robot_to_grasp_pose.params["pos_threshold"] = 1e-5
        self.events.set_robot_to_grasp_pose.params["rot_threshold"] = 1e-5
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 50

        # Wire termination params.
        self.terminations.plug_dropped.params[
            "end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params[
            "end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params[
            "grasp_rot_offset"] = self.grasp_rot_offset

        # Use 1:1 linear:exponential keypoint-tracking reward weighting.
        self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
            self.rewards.plug_socket_keypoint_tracking.weight)

    def play_mode(self):
        """Apply playback overrides on top of the shared Isaac Lab defaults.

        ``super().play_mode()`` caps ``num_envs`` and disables observation corruption; only the
        wider env spacing is specific to this task.
        """
        super().play_mode()
        self.scene.env_spacing = 2.5
