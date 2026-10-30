# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""ROS inference metadata for the Newton DisplayPort OSC policy."""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.configclass import configclass
from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose, compute_socket_root)

from .task_space_newton_env_cfg import (_OSC_ORIENTATION_SCALE, _OSC_POSITION_SCALE,
                                        Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg)

_DEPLOY_GEOMETRY_POS = (0.475, 0.125, 0.06)
_DEPLOY_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_DEPLOY_PLUG_CLEARANCE_Z = 0.068

_DEPLOY_SOCKET_ROOT = compute_socket_root(_DEPLOY_GEOMETRY_POS, _DEPLOY_SOCKET_ROT)
_DEPLOY_PLUG_ROOT, _DEPLOY_PLUG_ROT = compute_plug_pose(
    _DEPLOY_GEOMETRY_POS,
    _DEPLOY_SOCKET_ROT,
    z_clearance=_DEPLOY_PLUG_CLEARANCE_Z,
)


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg(
        Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg):
    """Newton OSC training configuration with its ROS and LEAPP contract."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep internal observation-term aliases in ``task_space_obs_order``;
        # expose only the canonical Isaac ROS Deploy port names here.
        self.obs_order = ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
        self.policy_action_space = "task"
        self.task_space_policy_abi = "displayport_newton_flange_v1"
        self.task_space_backend = "newton_mjwarp"
        self.task_space_observation_reference = "flange"
        self.task_space_observation_body_name = "flange"
        self.task_space_observation_body_offset = [0.0, 0.0, 0.0]
        self.task_space_action_reference = "robot_root"
        self.task_space_action_body_name = "flange"
        self.task_space_action_body_offset = [0.0, 0.0, 0.0]
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
        ]
        self.action_space = 6
        self.observation_space = 18
        self.action_scale = [*_OSC_POSITION_SCALE, *([_OSC_ORIENTATION_SCALE] * 3)]
        # The critic observes 13 robot joints and their velocities plus true
        # socket and plug position/quaternion pairs.
        self.state_space = 40
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(32.44),
            "joint2": math.radians(-16.71),
            "joint3": math.radians(-5.69),
            "joint4": math.radians(128.38),
            "joint5": math.radians(6.74),
            "joint6": math.radians(55.95),
            "joint7": math.radians(111.54),
        }

        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_DEPLOY_SOCKET_ROOT,
            rot=_DEPLOY_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_DEPLOY_PLUG_ROOT,
            rot=_DEPLOY_PLUG_ROT,
        )

        self.fixed_asset_init_pos_center = list(_DEPLOY_GEOMETRY_POS)
        pose_range = self.events.randomize_socket_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [pose_range[axis][1] for axis in ("x", "y", "z")]
        self.fixed_asset_init_orn_deg = [0.0, 0.0, 0.0]
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range[axis][1]) for axis in ("roll", "pitch", "yaw")
        ]

        socket_pos_noise = self.observations.policy.socket_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [socket_pos_noise] * 3
