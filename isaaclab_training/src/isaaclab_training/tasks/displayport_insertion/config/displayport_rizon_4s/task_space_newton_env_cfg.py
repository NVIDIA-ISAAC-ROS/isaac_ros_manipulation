# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Newton point-SDF OSC configuration for Rizon 4S DisplayPort insertion.

It is separate from :mod:`task_space_env_cfg` because its checkpoint ABI observes
the flange origin in a different tensor order than the PhysX task-space policy.
"""

import logging
import math

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_newton.physics import (MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg,
                                     NewtonShapeCfg)
from isaaclab_newton.sim.schemas import (MujocoJointDrivePropertiesCfg,
                                         MujocoRigidBodyPropertiesCfg, NewtonCollisionCfg,
                                         NewtonSDFCollisionCfg)
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from isaaclab_tasks.utils import PresetCfg
import isaaclab_training.mdp as deploy_mdp
from isaaclab_training.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
    ObservationsCfg, SOCKET_INSERTION_OFFSET)

from .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg, TaskSpaceEventCfg

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_OSC_POSITION_SCALE = (0.025, 0.025, 0.010)
_OSC_ORIENTATION_SCALE = 0.025
_OSC_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_OSC_DAMPING_RATIO = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_NEWTON_NUM_ENVS = 256
# Each replicated environment is an independent Newton world, so nconmax and njmax are
# per-world limits and must not be multiplied by scene.num_envs. The measured peaks below cover
# scripted insertion, seated, carry, curriculum-reset, and policy-rollout scenarios. Keep at
# least 2x headroom because MJWarp silently drops contacts or constraint rows past these limits.
_NEWTON_MEASURED_PEAK_CONTACTS_PER_WORLD = 382
_NEWTON_MEASURED_PEAK_CONSTRAINT_ROWS_PER_WORLD = 612
_NEWTON_NCONMAX_PER_WORLD = 1024
_NEWTON_NJMAX_PER_WORLD = 2048
# Triangle-pair capacity is the exception: Newton owns one scene-wide candidate-pair buffer.
# Measurements peaked at 6.4k pairs per environment during ordinary scenarios and 8.1k under
# full-force presses.
_NEWTON_TRIANGLE_PAIRS_PER_ENV = 12288
_FLANGE_FALLBACK_DENSITY = 1000.0

_LOGGER = logging.getLogger(__name__)


def _use_explicit_effort_control_arm_actuators(
    env_cfg: Rizon4sTaskSpaceDisplayportInsertionEnvCfg,
) -> None:
    """Use zero-gain explicit arm actuators to clamp OSC joint efforts."""
    for actuator_name in ("shoulder", "elbow", "wrist"):
        source_cfg = env_cfg.scene.robot.actuators[actuator_name]
        actuator_effort_limit = source_cfg.actuator_effort_limit
        if actuator_effort_limit is None:
            actuator_effort_limit = source_cfg.joint_effort_limit
        actuator_velocity_limit = source_cfg.actuator_velocity_limit
        if actuator_velocity_limit is None:
            actuator_velocity_limit = source_cfg.joint_velocity_limit
        if actuator_effort_limit is None or source_cfg.joint_effort_limit is None:
            raise ValueError(
                f"Newton OSC arm actuator {actuator_name!r} must define effort limits.")
        if actuator_velocity_limit is None or source_cfg.joint_velocity_limit is None:
            raise ValueError(
                f"Newton OSC arm actuator {actuator_name!r} must define velocity limits.")
        env_cfg.scene.robot.actuators[actuator_name] = IdealPDActuatorCfg(
            joint_names_expr=list(source_cfg.joint_names_expr),
            actuator_effort_limit=actuator_effort_limit,
            actuator_velocity_limit=actuator_velocity_limit,
            joint_effort_limit=source_cfg.joint_effort_limit,
            joint_velocity_limit=source_cfg.joint_velocity_limit,
            stiffness=0.0,
            damping=0.0,
            armature=source_cfg.armature,
            friction=source_cfg.friction,
            dynamic_friction=source_cfg.dynamic_friction,
            viscous_friction=source_cfg.viscous_friction,
        )


def _uniform_sphere_inertia(mass: float,
                            density: float = _FLANGE_FALLBACK_DENSITY
                            ) -> tuple[float, float, float]:
    """Return the diagonal inertia for a uniform sphere with the given mass."""
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Flange mass must be positive and finite, got {mass!r}.")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError(f"Fallback density must be positive and finite, got {density!r}.")
    radius = (3.0 * mass / (4.0 * math.pi * density))**(1.0 / 3.0)
    moment = 0.4 * mass * radius**2
    return (moment, moment, moment)


def _is_valid_diagonal_inertia(diagonal: object) -> bool:
    """Return whether a diagonal inertia is finite, positive, and physically realizable."""
    if diagonal is None:
        return False
    values = tuple(float(value) for value in diagonal)
    if len(values) != 3 or not all(math.isfinite(value) and value > 0.0 for value in values):
        return False
    return all(values[index] <= values[(index + 1) % 3] + values[(index + 2) % 3]
               for index in range(3))


def _is_valid_principal_axes(quaternion: object) -> bool:
    """Return whether a principal-axis quaternion is finite and normalized."""
    if quaternion is None:
        return False
    if hasattr(quaternion, "GetReal") and hasattr(quaternion, "GetImaginary"):
        values = (float(quaternion.GetReal()), *(float(value)
                                                 for value in quaternion.GetImaginary()))
    else:
        values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return False
    return math.isclose(sum(value * value for value in values),
                        1.0,
                        rel_tol=1.0e-5,
                        abs_tol=1.0e-5)


@clone
def _spawn_rizon_with_validated_flange_inertia(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn Rizon and author the deterministic Newton fallback inertia when needed."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    from pxr import Gf, UsdPhysics

    flange = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("flange"))
    if not flange.IsValid():
        raise ValueError(f"Rizon USD {cfg.usd_path!r} does not contain the required flange prim.")
    mass_api = UsdPhysics.MassAPI(flange)
    if not mass_api:
        raise ValueError(f"Rizon flange {flange.GetPath()} must have UsdPhysics.MassAPI.")

    diagonal = mass_api.GetDiagonalInertiaAttr().Get()
    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
    if _is_valid_diagonal_inertia(diagonal):
        if not _is_valid_principal_axes(principal_axes):
            raise ValueError(
                f"Rizon flange {flange.GetPath()} must have normalized principal axes.")
        return prim

    mass = mass_api.GetMassAttr().Get()
    diagonal = _uniform_sphere_inertia(float(mass) if mass is not None else float("nan"))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    _LOGGER.info("Authored deterministic flange inertia %s kg*m^2 for Newton asset %s.", diagonal,
                 cfg.usd_path)
    return prim


def _validate_sdf_meshes(root_prim, relative_paths: tuple[str, ...]) -> None:
    """Fail when a release asset no longer contains an expected colliding SDF mesh."""
    from pxr import UsdGeom, UsdPhysics

    missing = []
    for relative_path in relative_paths:
        mesh_path = f"{root_prim.GetPath()}{relative_path}"
        mesh_prim = root_prim.GetStage().GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(
                UsdGeom.Mesh) or not UsdPhysics.CollisionAPI(mesh_prim):
            missing.append(mesh_path)
    if missing:
        raise ValueError(
            f"DisplayPort asset is missing required colliding SDF meshes: {missing!r}.")


@clone
def _spawn_plug_with_validated_sdf_meshes(prim_path: str,
                                          cfg,
                                          translation=None,
                                          orientation=None,
                                          **kwargs):
    """Spawn the plug and validate the point-SDF mesh contract."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    _validate_sdf_meshes(prim, ("/collision_mesh", ))
    return prim


@clone
def _spawn_socket_with_validated_sdf_meshes(prim_path: str,
                                            cfg,
                                            translation=None,
                                            orientation=None,
                                            **kwargs):
    """Spawn the socket and validate the point-SDF mesh contract."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    _validate_sdf_meshes(
        prim,
        tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh"
              for body_id in (5, 6, 8, 12, 13)),
    )
    return prim


def _newton_sdf_properties(
    contact_offset: float,
    rest_offset: float,
    sdf_prim_paths: tuple[str, ...],
) -> dict[str, list[PhysxCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]]:
    """Create point-SDF properties only for meshes authored as SDF colliders."""
    properties: dict[str, list[PhysxCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]] = {
        "/.*": [PhysxCollisionCfg(contact_offset=contact_offset, rest_offset=rest_offset)]
    }
    for prim_path in sdf_prim_paths:
        properties[prim_path] = [
            NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.005),
            NewtonSDFCollisionCfg(
                sdf_max_resolution=256,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
                sdf_texture_format="uint16",
                sdf_padding=0.005,
                hydroelastic_enabled=False,
                hydroelastic_stiffness=1.0e8,
            ),
        ]
    return properties


@configclass
class DisplayportNewtonPhysicsCfg(PresetCfg):
    """Newton point-SDF physics profile for DisplayPort insertion."""

    newton_sdf: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=_NEWTON_NJMAX_PER_WORLD,
            nconmax=_NEWTON_NCONMAX_PER_WORLD,
            iterations=100,
            ls_iterations=50,
            update_data_interval=10,
            impratio=10.0,
            cone="elliptic",
            ccd_iterations=35,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            # Scene-wide candidate-pair capacity. DisplayportInsertionEnv sets it to
            # newton_triangle_pairs_per_env x num_envs once the final env count is known.
            # If this overflows, Newton warns and may omit candidate contacts.
            # max_triangle_pairs=_NEWTON_MAX_TRIANGLE_PAIRS,
            max_triangle_pairs=_NEWTON_TRIANGLE_PAIRS_PER_ENV * _NEWTON_NUM_ENVS,
        ),
        num_substeps=20,
        collision_decimation=10,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_sdf


@configclass
class NewtonTaskSpaceObservationsCfg:
    """Checkpoint-compatible actor and critic observations for Newton OSC."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations in the 18-dimensional checkpoint ABI order."""

        socket_pos = ObsTerm(
            func=deploy_mdp.rigid_object_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("dp_socket"),
                "offset": SOCKET_INSERTION_OFFSET
            },
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")),
        )
        tool_pos = ObsTerm(
            func=deploy_mdp.eef_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "flange",
                "offset": [0.0, 0.0, 0.0]
            },
        )
        tool_rot_6d = ObsTerm(
            func=deploy_mdp.eef_rot_6d_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "flange"
            },
        )
        socket_rot_6d = ObsTerm(
            func=deploy_mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # Keep every robot joint in the privileged critic to preserve the
    # checkpoint's 40-dimensional critic input.
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class NewtonTaskSpaceEventCfg(TaskSpaceEventCfg):
    """Newton-specific material and arm-friction randomization events."""

    randomize_arm_joint_friction = EventTerm(
        func=env_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS),
            "friction_distribution_params": (0.0, 0.15),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # OSC is torque controlled, so position-controller gain randomization is
    # intentionally disabled while joint-friction randomization remains active.
    randomize_arm_pd_gains: EventTerm | None = None


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """Newton point-SDF OSC training configuration.

    The actor contract is ``socket_pos, flange_pos, flange_rot_6d,
    socket_rot_6d``. This differs from the TCP-first PhysX task-space contract,
    so the configurations must not share checkpoints despite both being 18-D.

    Override ``scene.robot.spawn.usd_path`` with a USD calibrated for the
    robot that will execute the policy.
    """

    newton_triangle_pairs_per_env: int | None = _NEWTON_TRIANGLE_PAIRS_PER_ENV
    """Narrow-phase triangle-pair capacity per environment; scene-wide buffer = this x num_envs."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 100 Hz outer simulation ticks, 20 solver substeps, collision every 10
        # substeps, and one policy action every three outer ticks give 2 kHz,
        # 200 Hz, and 33.3 Hz respectively.
        self.sim.dt = 0.01
        self.sim.physics = DisplayportNewtonPhysicsCfg()
        # Use Newton-native actuator execution. Zero joint gains preserve direct
        # OSC effort control while effort saturation and solver limits remain active.
        self.sim.use_newton_actuators = True
        # The scene-wide triangle-pair capacity follows the final env count
        # (newton_triangle_pairs_per_env); nconmax / njmax are per world.
        self.scene.num_envs = _NEWTON_NUM_ENVS
        # Newton trains 1,500 iterations at 256 envs/GPU; anneal over the first third.
        self.events.reset_plug_curriculum.params["anneal_end_iter"] = 500.0
        self.decimation = 3
        self.sim.render_interval = self.decimation

        # Preserve source PhysX offsets on every collider, but apply Newton's
        # point-SDF schema only to meshes authored for SDF collision.
        self.scene.robot.spawn.func = _spawn_rizon_with_validated_flange_inertia
        # The PhysX profile replaces selected colliders with simplified hulls and removes the
        # corresponding source meshes. Newton instead needs the original point-SDF meshes listed
        # below, so undo every inherited collider-pruning/replacement directive before spawning.
        for asset_cfg in (self.scene.dp_plug, self.scene.dp_socket):
            asset_cfg.spawn.remove_disabled_colliders = False
            asset_cfg.spawn.collision_parts_file = None
            asset_cfg.spawn.collision_parts_template = None
            asset_cfg.spawn.remove_colliders = ()
        self.scene.dp_plug.spawn.func = _spawn_plug_with_validated_sdf_meshes
        self.scene.dp_socket.spawn.func = _spawn_socket_with_validated_sdf_meshes
        self.scene.dp_plug.spawn.collision_props = _newton_sdf_properties(
            0.00001,
            -0.00005,
            ("/collision_mesh", ),
        )
        self.scene.dp_socket.spawn.collision_props = _newton_sdf_properties(
            0.0001,
            -0.0001,
            tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh"
                  for body_id in (5, 6, 8, 12, 13)),
        )

        self.observations = NewtonTaskSpaceObservationsCfg()
        self.task_space_obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

        # RSL-RL clips raw actor outputs to +/-1 before OSC applies these scales.
        # Keep the action-term clip unset so the checkpoint contract has one
        # effective clipping stage and cannot acquire a latent +/-0.5 limit.
        self.actions.arm_action = deploy_mdp.DeployOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="flange",
            body_offset=deploy_mdp.DeployOperationalSpaceControllerActionCfg.OffsetCfg(),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=_OSC_STIFFNESS,
                motion_damping_ratio_task=_OSC_DAMPING_RATIO,
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            clip=None,
            position_scale=_OSC_POSITION_SCALE,
            orientation_scale=_OSC_ORIENTATION_SCALE,
        )

        # Retain the fully wired base events and replace only Newton-specific
        # randomization terms. Replacing the group would duplicate grasp wiring.
        newton_events = NewtonTaskSpaceEventCfg()
        self.events.randomize_arm_joint_friction = newton_events.randomize_arm_joint_friction
        self.events.randomize_arm_pd_gains = None
        self.events.plug_physics_material.params["static_friction_range"] = (3.0, 3.0)
        self.events.plug_physics_material.params["dynamic_friction_range"] = (3.0, 3.0)
        self.events.robot_physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.robot_physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        # Newton cancels robot-body gravity directly. OSC gravity compensation
        # stays disabled to avoid applying gravity twice.
        self.scene.robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        self.scene.robot.spawn.joint_drive_props = MujocoJointDrivePropertiesCfg(
            actuatorgravcomp=False)
        _use_explicit_effort_control_arm_actuators(self)

        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.1,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.05,
        )
        self.hand_hold_width = -0.1
        self.hand_close_width = -0.1

    def play_mode(self) -> None:
        """Apply deterministic playback overrides to the Newton training profile."""
        super().play_mode()
        self.scene.env_spacing = 2.5
        self.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
        self.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0
