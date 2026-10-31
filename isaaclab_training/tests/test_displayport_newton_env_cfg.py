# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Newton DisplayPort insertion production configuration."""

from dataclasses import fields
import math
from types import SimpleNamespace

import gymnasium as gym
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonSDFCollisionCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_training.mdp import DeployOperationalSpaceControllerActionCfg
from isaaclab_training.mdp import events as deploy_events
from isaaclab_training.mdp.actions import _pose_rel_action_extra
from isaaclab_training.mdp.events import (
    _body_link_jacobian_for_ik,
    _curriculum_goal_keypoint_positions,
    reset_plug_at_goal_curriculum,
    set_robot_to_object_grasp_pose,
)
from isaaclab_training.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_training.mdp.observations import eef_pos_w, rigid_object_pos_w
import isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s  # noqa: F401
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s import (
    task_space_newton_env_cfg as newton_cfg,
)
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s import (
    task_space_newton_ros_inference_env_cfg as newton_ros_cfg,
)
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s import (
    task_space_ros_inference_env_cfg as physx_ros_cfg,
)
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s \
    .agents.rsl_rl_ppo_cfg import (
        Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg,
        Rizon4sGravDisplayportInsertionRNNPPORunnerCfg,
        Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg,
    )
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s \
    .joint_pos_env_cfg import set_finger_joint_pos_grav
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s \
    .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s \
    .task_space_newton_env_cfg import (
        _is_valid_diagonal_inertia,
        _is_valid_principal_axes,
        _NEWTON_MEASURED_PEAK_CONSTRAINT_ROWS_PER_WORLD,
        _NEWTON_MEASURED_PEAK_CONTACTS_PER_WORLD,
        _NEWTON_NCONMAX_PER_WORLD,
        _NEWTON_NJMAX_PER_WORLD,
        _spawn_plug_with_validated_sdf_meshes,
        _spawn_rizon_with_validated_flange_inertia,
        _spawn_socket_with_validated_sdf_meshes,
        _uniform_sphere_inertia,
        _use_explicit_effort_control_arm_actuators,
        DisplayportNewtonPhysicsCfg,
        Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg,
    )
from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
    _DISPLAY_ASSETS_DEFAULT_DIR,
    DISPLAY_ASSETS_DIR,
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
)
import pytest
import torch
import warp as wp

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_ENV_ENTRY_POINT = (
    "isaaclab_training.tasks.displayport_insertion.insertion_env:DisplayportInsertionEnv"
)
_CFG_PACKAGE = "isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s"
_AGENT_MODULE = f"{_CFG_PACKAGE}.agents.rsl_rl_ppo_cfg"


def _observation_term_names(group: object) -> list[str]:
    """Return observation terms in their concatenation order."""
    return [
        field.name for field in fields(group)
        if isinstance(getattr(group, field.name), ObservationTermCfg)
    ]


def test_displayport_curriculum_composes_physical_seat_in_socket_frame() -> None:
    """Only at-goal rows receive the socket-local physical-seat offset."""
    keypoint_origin = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    socket_quat = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    ])
    insertion_axis = torch.tensor([1.0, 0.0, 0.0])
    depth = torch.tensor([0.0, 0.02, 0.02])
    at_goal_mask = torch.tensor([True, True, False])
    seat_offset = torch.tensor([0.00184, 0.0, -0.0006])

    result = _curriculum_goal_keypoint_positions(
        keypoint_origin,
        socket_quat,
        insertion_axis,
        depth,
        at_goal_mask,
        seat_offset,
    )

    torch.testing.assert_close(result[0],
                               torch.tensor([1.00184, 2.0, 2.9994]),
                               atol=1.0e-7,
                               rtol=0.0)
    torch.testing.assert_close(result[1],
                               torch.tensor([4.0, 5.02184, 5.9994]),
                               atol=1.0e-7,
                               rtol=0.0)
    torch.testing.assert_close(result[2], torch.tensor([7.0, 8.02, 9.0]), atol=1.0e-7, rtol=0.0)


@pytest.mark.parametrize(
    "cfg_cls",
    [
        Rizon4sTaskSpaceDisplayportInsertionEnvCfg,
        Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg,
    ],
)
def test_displayport_effective_reset_curricula_keep_physical_seat_bounds(cfg_cls: type) -> None:
    """Static regression for the reset curriculum geometry contract."""
    cfg = cfg_cls()
    curriculum = cfg.events.reset_plug_curriculum.params

    assert curriculum["at_goal_depth_range"] == pytest.approx([0.0, 0.01316])
    assert curriculum["at_goal_seat_offset"] == pytest.approx([0.00184, 0.0, -0.0006])


@pytest.mark.parametrize(
    ("is_fixed_base", "num_base_dofs", "expected_body_idx"),
    [
        pytest.param(True, 0, 2, id="fixed-base"),
        pytest.param(False, 6, 3, id="floating-base"),
    ],
)
def test_displayport_grasp_reset_selects_backend_neutral_jacobians(
    is_fixed_base: bool,
    num_base_dofs: int,
    expected_body_idx: int,
):
    """Reset IK must select the correct body and actuated columns from public data."""
    jacobians = torch.arange(3 * 5 * 6 * 10, dtype=torch.float32).reshape(3, 5, 6, 10)
    asset = SimpleNamespace(
        is_fixed_base=is_fixed_base,
        num_base_dofs=num_base_dofs,
        data=SimpleNamespace(body_link_jacobian_w=SimpleNamespace(torch=jacobians)),
    )
    env_ids = torch.tensor([2, 0])

    selected = _body_link_jacobian_for_ik(asset, env_ids, body_idx=3)

    torch.testing.assert_close(selected, jacobians[env_ids, expected_body_idx, :, num_base_dofs:])


def test_displayport_grasp_reset_holds_randomized_target_during_ik(
        monkeypatch: pytest.MonkeyPatch):
    """A reset must sample one grasp offset and hold it fixed across all IK iterations."""
    num_envs = 2
    num_arm_joints = 7
    identity_quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    joint_limits = torch.tensor([[[-math.pi, math.pi]] * num_arm_joints] * num_envs)

    robot_data = SimpleNamespace(
        joint_pos=wp.from_torch(torch.zeros(num_envs, num_arm_joints)),
        joint_vel=wp.from_torch(torch.zeros(num_envs, num_arm_joints)),
        joint_pos_limits=wp.from_torch(joint_limits),
        body_pos_w=wp.from_torch(torch.zeros(num_envs, 1, 3)),
        body_quat_w=wp.from_torch(identity_quaternions.unsqueeze(1)),
    )
    robot = SimpleNamespace(
        data=robot_data,
        set_joint_position_target_index=lambda **_kwargs: None,
        set_joint_velocity_target_index=lambda **_kwargs: None,
        write_joint_position_to_sim_index=lambda **_kwargs: None,
        write_joint_velocity_to_sim_index=lambda **_kwargs: None,
    )
    plug = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=wp.from_torch(torch.zeros(num_envs, 3)),
            root_link_quat_w=wp.from_torch(identity_quaternions),
        ),
        write_root_pose_to_sim=lambda *_args, **_kwargs: None,
        write_root_velocity_to_sim=lambda *_args, **_kwargs: None,
    )
    env = SimpleNamespace(device="cpu", scene={"dp_plug": plug})
    env_ids = torch.arange(num_envs)

    term = object.__new__(set_robot_to_object_grasp_pose)
    term.robot_asset = robot
    term.target_object_name = "dp_plug"
    term.grasp_offsets_buffer = torch.zeros(num_envs, 3)
    term.grasp_offset_tensor = torch.zeros(3)
    term.grasp_rot_offset_tensor = identity_quaternions
    term.eef_idx = 0
    term.num_arm_joints = num_arm_joints
    term.all_joints = list(range(num_arm_joints))
    term.finger_joints = []
    term.joint_name_to_idx = {}
    term.hand_hold_width = 0.0
    term.hand_close_width = 0.0
    term.gripper_joint_setter_func = lambda *_args, **_kwargs: None

    sample_count = 0
    ik_targets: list[torch.Tensor] = []

    def sample_uniform_once(
        _lower: torch.Tensor,
        _upper: torch.Tensor,
        size: tuple[int, int],
        device: str,
    ) -> torch.Tensor:
        nonlocal sample_count
        sample_count += 1
        return torch.full(size, float(sample_count), device=device)

    def record_pose_target(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        target = kwargs["ctrl_target_fingertip_midpoint_pos"]
        ik_targets.append(target.clone())
        return torch.ones_like(target), torch.zeros_like(target)

    monkeypatch.setattr(deploy_events.math_utils, "sample_uniform", sample_uniform_once)
    monkeypatch.setattr(deploy_events.fc, "get_pose_error", record_pose_target)
    monkeypatch.setattr(
        deploy_events.fc,
        "_get_delta_dof_pos",
        lambda **_kwargs: torch.zeros(num_envs, num_arm_joints),
    )
    monkeypatch.setattr(
        deploy_events,
        "_body_link_jacobian_for_ik",
        lambda *_args, **_kwargs: torch.zeros(num_envs, 6, num_arm_joints),
    )

    term(
        env,
        env_ids,
        max_iterations=3,
        pos_randomization_range={
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01)
        },
    )

    assert sample_count == 1
    assert len(ik_targets) == 3
    for target in ik_targets[1:]:
        torch.testing.assert_close(target, ik_targets[0])


def test_displayport_grasp_reset_supports_legacy_four_argument_gripper_callback():
    """Legacy gripper callbacks remain usable while receiving one deprecation warning."""
    calls = []

    def legacy_callback(joint_pos, reset_indices, finger_joints, finger_joint_position):
        calls.append((joint_pos, reset_indices, finger_joints, finger_joint_position))

    term = object.__new__(set_robot_to_object_grasp_pose)
    term.gripper_joint_setter_func = legacy_callback
    term._gripper_joint_setter_mode = None
    term.joint_name_to_idx = {"finger_joint": 7}
    joint_pos = torch.zeros(1, 8)

    with pytest.warns(DeprecationWarning,
                      match="Four-argument gripper_joint_setter_func") as warnings_record:
        term._set_gripper_joint_position(joint_pos, [0], [7], -0.1)
        term._set_gripper_joint_position(joint_pos, [0], [7], -0.2)

    assert len(warnings_record) == 1
    assert len(calls) == 2
    assert calls[0][0] is joint_pos
    assert calls[1][0] is joint_pos
    assert [call[1:] for call in calls] == [([0], [7], -0.1), ([0], [7], -0.2)]


def test_displayport_grasp_reset_passes_joint_name_mapping_to_updated_gripper_callback():
    """Updated gripper callbacks receive the backend-independent joint-name mapping."""
    received_mappings = []

    def mapped_callback(
        _joint_pos,
        _reset_indices,
        _finger_joints,
        _finger_joint_position,
        *,
        joint_name_to_idx,
    ):
        received_mappings.append(joint_name_to_idx)

    term = object.__new__(set_robot_to_object_grasp_pose)
    term.gripper_joint_setter_func = mapped_callback
    term._gripper_joint_setter_mode = None
    term.joint_name_to_idx = {"finger_joint": 7}

    term._set_gripper_joint_position(torch.zeros(1, 8), [0], [7], -0.1)

    assert received_mappings == [term.joint_name_to_idx]


def test_displayport_grav_reset_preserves_legacy_positional_joint_order():
    """Direct four-argument calls retain the original positional mimic behavior."""
    finger_joints = [8, 2, 5, 1, 9, 4]
    joint_pos = torch.full((2, 10), 42.0)

    with pytest.warns(DeprecationWarning, match="without joint_name_to_idx"):
        set_finger_joint_pos_grav(joint_pos, [0], finger_joints, -0.1)

    expected = {8: -0.1, 2: -0.1, 5: -0.1, 1: -0.1, 9: 0.1, 4: 0.1}
    for joint_idx, value in expected.items():
        assert joint_pos[0, joint_idx] == pytest.approx(value)
    torch.testing.assert_close(joint_pos[1], torch.full((10, ), 42.0))


def test_displayport_grav_reset_sets_mimic_joints_by_name():
    """The reset must preserve Grav mimic signs when Newton reorders the joints."""
    joint_names = [
        "finger_joint",
        "left_outer_finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_outer_finger_joint",
    ]
    finger_joints = list(range(7, 13))
    joint_name_to_idx = dict(zip(joint_names, finger_joints, strict=True))
    joint_pos = torch.full((2, 13), 42.0)

    set_finger_joint_pos_grav(
        joint_pos,
        [0],
        finger_joints,
        -0.1,
        joint_name_to_idx,
    )

    expected_gearing = {
        "finger_joint": 1.0,
        "left_inner_knuckle_joint": 1.0,
        "right_inner_knuckle_joint": 1.0,
        "right_outer_knuckle_joint": 1.0,
        "left_outer_finger_joint": -1.0,
        "right_outer_finger_joint": -1.0,
    }
    for joint_name, gearing in expected_gearing.items():
        assert joint_pos[0, joint_name_to_idx[joint_name]] == pytest.approx(-0.1 * gearing)
    torch.testing.assert_close(joint_pos[1], torch.full((13, ), 42.0))


def test_rigid_object_offset_is_rotated_and_cached_as_a_batch_view():
    """Object offsets must compose in the local frame without per-step batching."""

    class Scene(dict):
        pass

    positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    scene = Scene(dp_socket=SimpleNamespace(data=SimpleNamespace(
        root_pos_w=wp.from_torch(positions), root_quat_w=wp.from_torch(quaternions))))
    scene.env_origins = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    env = SimpleNamespace(device="cpu", num_envs=2, scene=scene)
    cfg = ObservationTermCfg(
        func=rigid_object_pos_w,
        params={
            "asset_cfg": SceneEntityCfg("dp_socket"),
            "offset": [1.0, 0.0, 0.0]
        },
    )
    term = rigid_object_pos_w(cfg, env)
    offset_batch_data_ptr = term._offset_batch.data_ptr()

    observed = term(env)
    observed_again = term(env)

    expected = torch.tensor([[1.5, 1.5, 2.5], [3.0, 5.0, 5.0]])
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_again, expected)
    assert term._offset_batch.data_ptr() == offset_batch_data_ptr
    assert term._offset_batch.stride()[0] == 0


def test_zero_eef_offset_skips_orientation_lookup():
    """A static zero offset must return body position without entering rotation code."""

    class Scene(dict):
        pass

    body_positions = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=wp.from_torch(body_positions)),
        find_bodies=lambda _name: ([0], ["flange"]),
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    env = SimpleNamespace(device="cpu", num_envs=2, scene=scene)
    cfg = ObservationTermCfg(
        func=eef_pos_w,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_name": "flange",
            "offset": [0.0, 0.0, 0.0]
        },
    )

    observed = eef_pos_w(cfg, env)(env)

    torch.testing.assert_close(observed, torch.tensor([[0.5, 1.5, 2.5], [3.0, 4.0, 5.0]]))


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        pytest.param((1.0, 1.0, 1.0), True, id="valid"),
        pytest.param((1.0, 1.0, 3.0), False, id="triangle-inequality"),
        pytest.param((0.0, 0.0, 0.0), False, id="zero"),
        pytest.param((float("nan"), 1.0, 1.0), False, id="non-finite"),
        pytest.param(None, False, id="missing"),
    ],
)
def test_newton_flange_inertia_validation(diagonal, expected):
    """Newton must accept only finite, positive, physically realizable inertia."""
    assert _is_valid_diagonal_inertia(diagonal) is expected


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        pytest.param((1.0, 0.0, 0.0, 0.0), True, id="identity"),
        pytest.param((0.5, 0.5, 0.5, 0.5), True, id="normalized"),
        pytest.param((0.0, 0.0, 0.0, 0.0), False, id="zero"),
        pytest.param((2.0, 0.0, 0.0, 0.0), False, id="not-normalized"),
        pytest.param((float("nan"), 0.0, 0.0, 0.0), False, id="non-finite"),
        pytest.param(None, False, id="missing"),
    ],
)
def test_newton_flange_principal_axes_validation(quaternion, expected):
    """Newton must reject missing, non-finite, or non-normalized principal axes."""
    assert _is_valid_principal_axes(quaternion) is expected


def test_newton_flange_inertia_matches_documented_fallback():
    """The authored nominal inertia must reproduce Newton's former deterministic fallback."""
    assert _uniform_sphere_inertia(1.0e-4) == pytest.approx((3.3164051824e-10, ) * 3)
    with pytest.raises(ValueError, match="positive and finite"):
        _uniform_sphere_inertia(0.0)


def test_newton_flange_inertia_is_authored_before_multi_env_clone(monkeypatch: pytest.MonkeyPatch):
    """The deterministic flange inertia must be copied to every matching environment."""
    from isaaclab.sim.utils import prims as prim_utils
    from isaaclab.sim.utils import queries as query_utils
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for env_index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_index}")

    monkeypatch.setattr(prim_utils, "get_current_stage", lambda: stage)
    monkeypatch.setattr(query_utils, "get_current_stage", lambda: stage)

    def spawn_zero_inertia_robot(prim_path, _cfg, *_args, **_kwargs):
        robot = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        flange = UsdGeom.Xform.Define(stage, f"{prim_path}/flange").GetPrim()
        mass_api = UsdPhysics.MassAPI.Apply(flange)
        mass_api.CreateMassAttr(1.0e-4)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(0.0, 0.0, 0.0, 0.0))
        return robot

    monkeypatch.setattr(newton_cfg, "spawn_from_usd", spawn_zero_inertia_robot)

    spawned = _spawn_rizon_with_validated_flange_inertia(
        "/World/envs/env_.*/Robot",
        SimpleNamespace(usd_path="test_rizon.usd"),
    )

    assert str(spawned.GetPath()) == "/World/envs/env_0/Robot"
    expected = _uniform_sphere_inertia(1.0e-4)
    for env_index in range(2):
        flange = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot/flange")
        mass_api = UsdPhysics.MassAPI(flange)
        assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(expected)
        principal_axes = mass_api.GetPrincipalAxesAttr().Get()
        assert principal_axes.GetReal() == pytest.approx(1.0)
        assert tuple(principal_axes.GetImaginary()) == pytest.approx((0.0, 0.0, 0.0))


def test_newton_flange_rejects_valid_inertia_with_invalid_principal_axes(
        monkeypatch: pytest.MonkeyPatch):
    """A calibrated asset must not preserve an invalid principal-axis quaternion."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()

    def spawn_invalid_axes_robot(prim_path, _cfg, *_args, **_kwargs):
        robot = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        flange = UsdGeom.Xform.Define(stage, f"{prim_path}/flange").GetPrim()
        mass_api = UsdPhysics.MassAPI.Apply(flange)
        mass_api.CreateMassAttr(1.0)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(0.0, 0.0, 0.0, 0.0))
        return robot

    monkeypatch.setattr(newton_cfg, "spawn_from_usd", spawn_invalid_axes_robot)

    with pytest.raises(ValueError, match="normalized principal axes"):
        _spawn_rizon_with_validated_flange_inertia.__wrapped__(
            "/World/Robot",
            SimpleNamespace(usd_path="invalid_axes.usd"),
        )


def test_newton_ideal_pd_rejects_missing_limits():
    """A future robot asset must not silently create an unbounded OSC actuator."""
    cfg = Rizon4sTaskSpaceDisplayportInsertionEnvCfg()
    cfg.scene.robot.actuators["shoulder"].actuator_effort_limit = None
    cfg.scene.robot.actuators["shoulder"].joint_effort_limit = None
    with pytest.raises(ValueError, match="must define effort limits"):
        _use_explicit_effort_control_arm_actuators(cfg)


def test_displayport_newton_tasks_route_to_dedicated_configs():
    """Newton registrations must not replace or redirect existing PhysX tasks."""
    expected = {
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton": (
            f"{_CFG_PACKAGE}.task_space_newton_env_cfg:"
            f"Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference": (
            f"{_CFG_PACKAGE}.task_space_newton_ros_inference_env_cfg:"
            "Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton": (
            f"{_CFG_PACKAGE}.task_space_newton_env_cfg:"
            f"Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference": (
            f"{_CFG_PACKAGE}.task_space_newton_ros_inference_env_cfg:"
            "Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg",
        ),
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace": (
            f"{_CFG_PACKAGE}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg",
        ),
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference": (
            f"{_CFG_PACKAGE}.task_space_ros_inference_env_cfg:"
            f"Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg",
            f"{_AGENT_MODULE}:Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg",
        ),
    }

    for task_id, (env_cfg_entry_point, runner_cfg_entry_point) in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == _ENV_ENTRY_POINT
        assert spec.disable_env_checker
        assert spec.kwargs["env_cfg_entry_point"] == env_cfg_entry_point
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == runner_cfg_entry_point


@pytest.mark.parametrize(
    ("task_id", "expected_num_envs"),
    [
        pytest.param(
            "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton", 256, id="training"),
        pytest.param(
            "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference",
            256,
            id="ros-inference",
        ),
    ],
)
def test_registered_displayport_newton_tasks_resolve_compatible_safe_defaults(
        task_id: str, expected_num_envs: int):
    """Every Newton variant must resolve to Newton with its supported shard size."""
    cfg = resolve_presets(load_cfg_from_registry(task_id, "env_cfg_entry_point"))
    runner = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.actions.arm_action, DeployOperationalSpaceControllerActionCfg)
    assert cfg.scene.num_envs == expected_num_envs
    assert cfg.newton_triangle_pairs_per_env == 12288
    assert cfg.sim.physics.collision_cfg.max_triangle_pairs == 12288 * 256
    assert runner.seed == 123
    assert runner.clip_actions == pytest.approx(1.0)
    cfg.validate()


def test_displayport_newton_timing_solver_and_point_sdf_assets():
    """Newton defaults must retain the functional timing and point-SDF contracts."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()

    assert cfg.scene.num_envs == 256
    assert cfg.sim.dt == pytest.approx(0.01)
    assert cfg.decimation == 3
    assert cfg.sim.render_interval == cfg.decimation
    assert isinstance(cfg.sim.physics, DisplayportNewtonPhysicsCfg)

    physics = cfg.sim.physics.default
    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 20
    assert physics.collision_decimation == 10
    assert physics.default_shape_cfg.gap == pytest.approx(0.005)
    assert physics.num_substeps / cfg.sim.dt == pytest.approx(2000.0)
    assert physics.num_substeps / cfg.sim.dt / physics.collision_decimation == pytest.approx(200.0)
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == pytest.approx(100.0 / 3.0)

    solver = physics.solver_cfg
    assert isinstance(solver, MJWarpSolverCfg)
    assert solver.solver == "newton"
    assert solver.integrator == "implicitfast"
    assert solver.use_mujoco_contacts is False
    assert solver.njmax == _NEWTON_NJMAX_PER_WORLD == 2048
    assert solver.nconmax == _NEWTON_NCONMAX_PER_WORLD == 1024
    assert solver.njmax >= 2 * _NEWTON_MEASURED_PEAK_CONSTRAINT_ROWS_PER_WORLD
    assert solver.nconmax >= 2 * _NEWTON_MEASURED_PEAK_CONTACTS_PER_WORLD

    collision = physics.collision_cfg
    assert collision is not None
    assert collision.reduce_contacts is True
    assert cfg.newton_triangle_pairs_per_env == 12288
    assert collision.max_triangle_pairs == 12288 * 256

    expected_default_asset_dir = f"{ISAAC_NUCLEUS_DIR}/Props/Factory/display_port_cable_assets"
    assert expected_default_asset_dir == _DISPLAY_ASSETS_DEFAULT_DIR
    assert "staging" not in _DISPLAY_ASSETS_DEFAULT_DIR
    assert cfg.scene.dp_plug.spawn.usd_path == f"{DISPLAY_ASSETS_DIR}/displayport_plug.usd"
    assert cfg.scene.dp_socket.spawn.usd_path == (
        f"{DISPLAY_ASSETS_DIR}/displayport_socket_no_protrusions.usd"
    )
    for asset_cfg in (cfg.scene.dp_plug, cfg.scene.dp_socket):
        assert asset_cfg.spawn.remove_disabled_colliders is False
        assert asset_cfg.spawn.collision_parts_file is None
        assert asset_cfg.spawn.collision_parts_template is None
        assert asset_cfg.spawn.remove_colliders == ()
    plug_sdf_paths = ("/collision_mesh", )
    socket_sdf_paths = tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh"
                             for body_id in (5, 6, 8, 12, 13))
    collision_fragment_sets = (
        (cfg.scene.dp_plug.spawn.collision_props, 0.00001, -0.00005, plug_sdf_paths),
        (cfg.scene.dp_socket.spawn.collision_props, 0.0001, -0.0001, socket_sdf_paths),
    )
    for mapping, contact_offset, rest_offset, sdf_paths in collision_fragment_sets:
        assert set(mapping) == {"/.*", *sdf_paths}

        physx_fragments = mapping["/.*"]
        assert not any(isinstance(fragment, NewtonCollisionCfg) for fragment in physx_fragments)
        assert not any(isinstance(fragment, NewtonSDFCollisionCfg) for fragment in physx_fragments)
        physx = next(fragment for fragment in physx_fragments
                     if isinstance(fragment, PhysxCollisionCfg))
        assert physx.contact_offset == pytest.approx(contact_offset)
        assert physx.rest_offset == pytest.approx(rest_offset)

        for prim_path in sdf_paths:
            fragments = mapping[prim_path]
            assert not any(isinstance(fragment, PhysxCollisionCfg) for fragment in fragments)
            collision = next(fragment for fragment in fragments
                             if isinstance(fragment, NewtonCollisionCfg))
            sdf = next(fragment for fragment in fragments
                       if isinstance(fragment, NewtonSDFCollisionCfg))
            assert collision.contact_margin == pytest.approx(0.0)
            assert collision.contact_gap == pytest.approx(0.005)
            assert sdf.sdf_max_resolution == 256
            assert sdf.sdf_narrow_band_inner == pytest.approx(-0.005)
            assert sdf.sdf_narrow_band_outer == pytest.approx(0.005)
            assert sdf.sdf_texture_format == "uint16"
            assert sdf.sdf_padding == pytest.approx(0.005)
            assert sdf.hydroelastic_enabled is False
            assert sdf.hydroelastic_stiffness == pytest.approx(1.0e8)


def test_displayport_newton_osc_abi_robot_and_gravity_settings():
    """OSC must command flange-relative pose deltas without double gravity compensation."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    action = cfg.actions.arm_action

    assert isinstance(action, DeployOperationalSpaceControllerActionCfg)
    assert action.asset_name == "robot"
    assert action.joint_names == _ARM_JOINTS
    assert action.body_name == "flange"
    assert action.body_offset is not None
    assert tuple(action.body_offset.pos) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(action.body_offset.rot) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert tuple(action.position_scale) == pytest.approx((0.025, 0.025, 0.010))
    assert action.orientation_scale == pytest.approx(0.025)
    assert action.clip is None
    assert action.nullspace_joint_pos_target == "none"

    controller = action.controller_cfg
    assert isinstance(controller, OperationalSpaceControllerCfg)
    assert list(controller.target_types) == ["pose_rel"]
    assert controller.impedance_mode == "fixed"
    assert controller.inertial_dynamics_decoupling is True
    assert controller.partial_inertial_dynamics_decoupling is False
    assert controller.gravity_compensation is False
    assert tuple(controller.motion_stiffness_task) == pytest.approx(
        (300.0, 300.0, 300.0, 30.0, 30.0, 30.0))
    assert tuple(controller.motion_damping_ratio_task) == pytest.approx((1.0, ) * 6)
    assert controller.nullspace_control == "none"

    assert cfg.scene.robot.spawn.rigid_props.gravcomp == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is False

    expected_limits = {
        "shoulder": (123.0, 2.094),
        "elbow": (64.0, 2.443),
        "wrist": (39.0, 4.887),
    }
    for actuator_name, (effort_limit, velocity_limit) in expected_limits.items():
        actuator = cfg.scene.robot.actuators[actuator_name]
        assert isinstance(actuator, IdealPDActuatorCfg)
        assert actuator.actuator_effort_limit == pytest.approx(effort_limit)
        assert actuator.joint_effort_limit == pytest.approx(effort_limit)
        assert actuator.actuator_velocity_limit == pytest.approx(velocity_limit)
        assert actuator.joint_velocity_limit == pytest.approx(velocity_limit)
        assert actuator.stiffness == pytest.approx(0.0)
        assert actuator.damping == pytest.approx(0.0)

    drive = cfg.scene.robot.actuators["gripper_drive"]
    assert drive.joint_names_expr == ["finger_joint"]
    assert drive.effort_limit_sim == pytest.approx(200.0)
    assert drive.velocity_limit_sim == pytest.approx(2.0)
    assert drive.stiffness == pytest.approx(2000.0)
    assert drive.damping == pytest.approx(10.0)
    assert drive.friction == pytest.approx(0.0)
    assert drive.armature == pytest.approx(0.1)

    passive = cfg.scene.robot.actuators["gripper_passive"]
    assert passive.joint_names_expr == [".*_knuckle_joint", ".*_outer_finger_joint"]
    assert passive.effort_limit_sim == pytest.approx(20.0)
    assert passive.velocity_limit_sim == pytest.approx(1.0)
    assert passive.stiffness == pytest.approx(2000.0)
    assert passive.damping == pytest.approx(10.0)
    assert passive.friction == pytest.approx(0.0)
    assert passive.armature == pytest.approx(0.05)
    assert cfg.hand_hold_width == pytest.approx(-0.1)
    assert cfg.hand_close_width == pytest.approx(-0.1)


def test_displayport_newton_observation_abi_noise_and_deployment_metadata():
    """Actor order, flange frame, reset-held noise, and dimensions form the checkpoint ABI."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    actor = cfg.observations.policy
    critic = cfg.observations.critic
    expected_actor_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

    assert cfg.task_space_obs_order == expected_actor_order
    assert _observation_term_names(actor) == expected_actor_order
    assert actor.concatenate_terms is True
    assert actor.enable_corruption is True
    assert actor.socket_pos.params["asset_cfg"].name == "dp_socket"
    assert tuple(actor.socket_pos.params["offset"]) == pytest.approx(
        tuple(SOCKET_INSERTION_OFFSET))
    assert isinstance(actor.socket_pos.noise, ResetSampledConstantNoiseModelCfg)
    assert isinstance(actor.socket_pos.noise.noise_cfg, UniformNoiseCfg)
    assert actor.socket_pos.noise.noise_cfg.n_min == pytest.approx(-0.01)
    assert actor.socket_pos.noise.noise_cfg.n_max == pytest.approx(0.01)
    assert actor.socket_pos.noise.noise_cfg.operation == "add"
    assert actor.tool_pos.params["asset_cfg"].name == "robot"
    assert actor.tool_pos.params["body_name"] == "flange"
    assert tuple(actor.tool_pos.params["offset"]) == pytest.approx((0.0, 0.0, 0.0))
    assert actor.tool_rot_6d.params["body_name"] == "flange"
    assert actor.socket_rot_6d.params["asset_cfg"].name == "dp_socket"

    assert _observation_term_names(critic) == [
        "joint_pos",
        "joint_vel",
        "socket_pos",
        "socket_quat",
        "plug_pos",
        "plug_quat",
    ]
    assert critic.joint_pos.params["asset_cfg"].joint_names == [".*"]
    assert critic.joint_vel.params["asset_cfg"].joint_names == [".*"]

    ros_cfg = newton_ros_cfg.Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg()
    assert ros_cfg.obs_order == ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
    assert ros_cfg.policy_action_space == "task"
    assert ros_cfg.task_space_policy_abi == "displayport_newton_flange_v1"
    assert ros_cfg.task_space_backend == "newton_mjwarp"
    assert ros_cfg.task_space_observation_reference == "flange"
    assert ros_cfg.task_space_observation_body_name == "flange"
    assert ros_cfg.task_space_observation_body_offset == pytest.approx([0.0, 0.0, 0.0])
    assert ros_cfg.task_space_action_reference == "robot_root"
    assert ros_cfg.task_space_action_body_name == "flange"
    assert ros_cfg.task_space_action_body_offset == pytest.approx([0.0, 0.0, 0.0])
    assert ros_cfg.arm_joint_names == _ARM_JOINTS
    assert ros_cfg.action_space == 6
    assert ros_cfg.action_scale == pytest.approx([0.025, 0.025, 0.010, 0.025, 0.025, 0.025])
    assert ros_cfg.observation_space == 18
    assert ros_cfg.state_space == 40
    assert ros_cfg.fixed_asset_init_pos_range == pytest.approx([0.01, 0.01, 0.02])
    assert ros_cfg.fixed_asset_init_orn_deg_range == pytest.approx([2.0, 2.0, 2.0])
    assert ros_cfg.fixed_asset_pos_obs_noise_level == pytest.approx([0.01, 0.01, 0.01])
    assert ros_cfg.events.set_robot_to_grasp_pose.params["pos_threshold"] == pytest.approx(1.0e-5)
    assert ros_cfg.events.set_robot_to_grasp_pose.params["rot_threshold"] == pytest.approx(1.0e-5)
    assert ros_cfg.events.set_robot_to_grasp_pose.params["max_iterations"] == 50


def test_physx_task_space_deployment_frame_metadata_remains_explicit():
    """PhysX and Newton bundles must expose distinguishable observation frames."""
    cfg = physx_ros_cfg.Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg()
    assert cfg.task_space_policy_abi == "displayport_physx_tcp_v1"
    assert cfg.task_space_backend == "physx"
    assert cfg.task_space_observation_reference == "tcp"
    assert cfg.task_space_observation_body_name == "flange"
    assert cfg.task_space_observation_body_offset == pytest.approx([0.0, 0.0, 0.1925])
    assert cfg.task_space_action_reference == "robot_root"
    assert cfg.task_space_action_body_name == "flange"
    assert cfg.task_space_action_body_offset == pytest.approx([0.0, 0.0, 0.0])
    assert cfg.action_scale == pytest.approx([0.025] * 6)


def test_displayport_newton_domain_randomization_curriculum_and_rewards():
    """Checkpoint material, reset, friction, curriculum, and reward contracts must remain exact."""
    cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    events = cfg.events

    material_coefficients = {
        "plug_physics_material": 3.0,
        "socket_physics_material": 0.001,
        "robot_physics_material": 1.0,
    }
    for event_name, coefficient in material_coefficients.items():
        event = getattr(events, event_name)
        assert event.mode == "startup"
        assert event.params["static_friction_range"] == pytest.approx((coefficient, coefficient))
        assert event.params["dynamic_friction_range"] == pytest.approx((coefficient, coefficient))
        assert event.params["restitution_range"] == pytest.approx((0.0, 0.0))
        assert event.params["num_buckets"] == 16

    arm_friction = events.randomize_arm_joint_friction
    assert arm_friction.mode == "reset"
    assert arm_friction.params["asset_cfg"].name == "robot"
    assert arm_friction.params["asset_cfg"].joint_names == _ARM_JOINTS
    assert arm_friction.params["friction_distribution_params"] == pytest.approx((0.0, 0.15))
    assert arm_friction.params["operation"] == "add"
    assert arm_friction.params["distribution"] == "uniform"
    assert events.randomize_arm_pd_gains is None

    pose_range = events.randomize_socket_pose.params["pose_range"]
    assert pose_range["x"] == pytest.approx([-0.01, 0.01])
    assert pose_range["y"] == pytest.approx([-0.01, 0.01])
    assert pose_range["z"] == pytest.approx([-0.02, 0.02])
    for axis in ("roll", "pitch", "yaw"):
        assert pose_range[axis] == pytest.approx([-math.radians(2.0), math.radians(2.0)])

    curriculum = events.reset_plug_curriculum.params
    assert curriculum["at_goal_prob"] == pytest.approx(0.8)
    assert curriculum["at_goal_prob_final"] == pytest.approx(0.0)
    assert curriculum["anneal_start_iter"] == pytest.approx(0.0)
    assert curriculum["anneal_end_iter"] == pytest.approx(500.0)
    assert curriculum["num_steps_per_env"] == 128
    assert curriculum["insertion_axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert curriculum["insertion_length"] == pytest.approx(0.011)
    assert curriculum["at_goal_depth_range"] == pytest.approx([0.0, 0.01316])
    assert curriculum["at_goal_seat_offset"] == pytest.approx([0.00184, 0.0, -0.0006])
    assert curriculum["at_goal_depth_range"][1] + curriculum["at_goal_seat_offset"][
        0] == pytest.approx(0.015)
    assert curriculum["approach_depth_range"] == pytest.approx([0.02, 0.06])
    assert curriculum["socket_insertion_offset"] == pytest.approx(SOCKET_INSERTION_OFFSET)
    assert curriculum["plug_insertion_offset"] == pytest.approx(PLUG_INSERTION_OFFSET)
    assert curriculum["goal_rot"] == pytest.approx(PLUG_GOAL_ROT)
    assert curriculum["normal_pose_range"]["x"] == pytest.approx([-0.02, 0.02])
    assert curriculum["normal_pose_range"]["y"] == pytest.approx([-0.02, 0.02])
    assert curriculum["normal_pose_range"]["z"] == pytest.approx([0.0, 0.0])
    grasp_reset = events.set_robot_to_grasp_pose.params
    assert grasp_reset["pos_threshold"] == pytest.approx(1.0e-5)
    assert grasp_reset["rot_threshold"] == pytest.approx(1.0e-5)
    assert grasp_reset["max_iterations"] == 50

    rewards = cfg.rewards
    assert rewards.plug_socket_keypoint_tracking.weight == pytest.approx(-1.5)
    assert rewards.plug_socket_keypoint_tracking_exp.weight == pytest.approx(1.5)
    assert rewards.action_rate.weight == pytest.approx(-5.0e-6)
    assert rewards.plug_socket_keypoint_tracking_exp.params["kp_exp_coeffs"] == [
        (50, 0.0001),
        (300, 0.0001),
        (600, 0.0001),
        (2000, 0.0001),
    ]
    assert rewards.plug_socket_keypoint_tracking_exp.params["kp_use_sum_of_exps"] is False


@pytest.mark.parametrize(
    ("iteration", "expected_probability"),
    [
        pytest.param(0, 0.8, id="start"),
        pytest.param(1000, 0.4, id="midpoint"),
        pytest.param(2000, 0.0, id="end"),
        pytest.param(3000, 0.0, id="after-end"),
    ],
)
def test_displayport_newton_curriculum_anneals_at_goal_probability(iteration: int,
                                                                   expected_probability: float):
    """The optimized reset curriculum must preserve its 256k environment-step horizon."""
    term = object.__new__(reset_plug_at_goal_curriculum)
    term.at_goal_prob = 0.8
    term.at_goal_prob_final = 0.0
    term.anneal_start_iter = 0.0
    term.anneal_end_iter = 2000.0
    term.num_steps_per_env = 128
    env = SimpleNamespace(common_step_counter=iteration * term.num_steps_per_env)

    assert term._current_at_goal_prob(env) == pytest.approx(expected_probability)


def test_displayport_newton_runner_and_play_preserve_physx_defaults():
    """Newton uses its checkpoint horizon and safe shard size without mutating PhysX defaults."""
    newton_runner = Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg()
    physx_runner = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()
    task_space_runner = Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg()
    assert newton_runner.seed == 123
    assert newton_runner.max_iterations == 1500
    assert newton_runner.experiment_name == "displayport_insertion_rizon4s_newton_osc"
    assert physx_runner.max_iterations == 1500
    assert physx_runner.experiment_name == "displayport_insertion_rizon4s"
    assert task_space_runner.max_iterations == 1000
    assert task_space_runner.num_steps_per_env == 128
    assert newton_runner.num_steps_per_env == 128
    assert physx_runner.num_steps_per_env == 512
    assert newton_runner.clip_actions == physx_runner.clip_actions == pytest.approx(1.0)
    assert newton_runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert physx_runner.obs_groups == {"actor": ["policy"], "critic": ["critic"]}

    train_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    play_cfg = Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg()
    play_cfg.play_mode()
    assert train_cfg.scene.num_envs == 256
    assert train_cfg.sim.use_newton_actuators is True
    assert train_cfg.scene.robot.spawn.func is _spawn_rizon_with_validated_flange_inertia
    assert train_cfg.scene.dp_plug.spawn.func is _spawn_plug_with_validated_sdf_meshes
    assert train_cfg.scene.dp_socket.spawn.func is _spawn_socket_with_validated_sdf_meshes
    assert train_cfg.observations.policy.enable_corruption is True
    assert train_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == pytest.approx(0.8)
    assert play_cfg.observations.policy.enable_corruption is False
    assert play_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == pytest.approx(0.0)
    assert play_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] == pytest.approx(0.0)
    assert (play_cfg.events.randomize_socket_pose.params["pose_range"] == (
        train_cfg.events.randomize_socket_pose.params["pose_range"]))
    assert play_cfg.scene.num_envs == 50

    physx_cfg = Rizon4sTaskSpaceDisplayportInsertionEnvCfg()
    for actuator_name in ("shoulder", "elbow", "wrist"):
        assert isinstance(physx_cfg.scene.robot.actuators[actuator_name], ImplicitActuatorCfg)
    assert physx_cfg.scene.num_envs == 4096
    assert isinstance(physx_cfg.sim.physics, PhysxCfg)
    assert physx_cfg.sim.dt == pytest.approx(1.0 / 240.0)
    assert physx_cfg.decimation == 8
    assert 1.0 / (physx_cfg.sim.dt * physx_cfg.decimation) == pytest.approx(30.0)
    actor = physx_cfg.observations.policy
    assert _observation_term_names(actor) == [
        "eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"
    ]
    assert actor.enable_corruption is False
    assert actor.socket_kp_pos.noise is None
    assert tuple(actor.eef_pos.params["offset"]) == pytest.approx((0.0, 0.0, 0.1925))
    assert physx_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling is False
    nominal_usd_path = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.usd_path
    assert physx_cfg.scene.robot.spawn.usd_path == nominal_usd_path
    assert train_cfg.scene.robot.spawn.usd_path == nominal_usd_path
    assert physx_cfg.events.plug_physics_material.params["static_friction_range"] == pytest.approx(
        (0.001, 0.001))
    assert physx_cfg.events.robot_physics_material.params[
        "static_friction_range"] == pytest.approx((0.75, 0.75))


def test_osc_leapp_metadata_expands_per_axis_action_scales():
    """OSC export metadata must serialize scalar and per-axis scales consistently."""
    extra = _pose_rel_action_extra(
        position_scale=[0.025, 0.025, 0.01],
        orientation_scale=0.025,
        target_types=["pose_rel"],
    )

    assert extra["scale"] == pytest.approx([0.025, 0.025, 0.01, 0.025, 0.025, 0.025])
    assert extra["position_scale"] == pytest.approx([0.025, 0.025, 0.01])
    assert extra["orientation_scale"] == pytest.approx(0.025)
