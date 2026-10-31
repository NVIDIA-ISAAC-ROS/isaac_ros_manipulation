# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Plan and validate DisplayPort training-optimization experiments.

This module deliberately has no simulator imports. It provides deterministic,
unit-testable planning and post-run checks around the GPU-dependent validation
work. Run it with ``python -m isaaclab_training.cli.displayport_training_validation``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
from enum import StrEnum
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys
from typing import Any


class Backend(StrEnum):
    """Physics backends supported by the DisplayPort task."""

    PHYSX = "physx"
    NEWTON = "newton"


class RobotAsset(StrEnum):
    """Robot-asset variants in the optimization matrix."""

    NOMINAL = "nominal"
    CALIBRATED = "calibrated"


@dataclass(frozen=True)
class ExperimentProfile:
    """One environment, solver, and PPO-minibatch configuration."""

    name: str
    num_envs: int
    solver_iterations: int
    num_mini_batches: int = 16


@dataclass(frozen=True)
class ExperimentSpec:
    """One controlled DisplayPort training experiment."""

    backend: Backend
    profile: ExperimentProfile
    robot_asset: RobotAsset
    curriculum_enabled: bool
    num_steps_per_env: int = 128
    world_size: int = 1
    seed: int = 123
    max_iterations: int | None = None

    @property
    def name(self) -> str:
        """Return a collision-resistant run name encoding every matrix dimension."""
        curriculum = "cur" if self.curriculum_enabled else "nocur"
        iterations = f"_i{self.max_iterations}" if self.max_iterations is not None else ""
        return (
            f"dp_opt_{self.backend}_{self.profile.name}_{self.robot_asset}_{curriculum}"
            f"{iterations}_seed{self.seed}"
        )


@dataclass(frozen=True)
class ReferenceSchedule:
    """Reference schedule whose sample budgets must be preserved."""

    num_envs: int = 1024
    num_steps_per_env: int = 512
    curriculum_anneal_end_iteration: int = 500
    save_interval: int = 50
    physx_max_iterations: int = 1500
    newton_max_iterations: int = 1000
    world_size: int = 1

    def max_iterations(self, backend: Backend) -> int:
        """Return the reference training length for ``backend``."""
        if backend is Backend.PHYSX:
            return self.physx_max_iterations
        return self.newton_max_iterations


@dataclass(frozen=True)
class ValidatedBackendLimits:
    """Largest environment count validated on one GPU for each backend."""

    physx_envs_per_gpu: int = 1024
    newton_envs_per_gpu: int = 256

    def for_backend(self, backend: Backend) -> int:
        """Return the evidence-backed per-GPU limit for ``backend``."""
        if backend is Backend.PHYSX:
            return self.physx_envs_per_gpu
        return self.newton_envs_per_gpu


@dataclass(frozen=True)
class ConstraintViolation:
    """A reason an experiment must not be submitted yet."""

    code: str
    message: str


@dataclass(frozen=True)
class MetricRequirement:
    """A required numeric or Boolean measurement from a real validation run."""

    name: str
    maximum: float | None = None
    minimum: float | None = None
    exact: float | bool | None = None
    at_least_metric: str | None = None


@dataclass(frozen=True)
class TrainingHealthThresholds:
    """Optional scalar thresholds for a training log."""

    tail_window: int = 20
    minimum_iterations: int = 1
    minimum_final_terminal_success: float | None = None
    minimum_tail_terminal_success: float | None = None
    minimum_tail_median_fps: float | None = None
    minimum_fps_ratio_to_baseline: float | None = None


@dataclass(frozen=True)
class HealthFinding:
    """One blocking training-health issue."""

    code: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class TrainingHealthReport:
    """Structured result of training-log analysis."""

    metrics: dict[str, list[float]]
    findings: tuple[HealthFinding, ...]
    source: str = "<memory>"
    tail_window: int = 20

    @property
    def passed(self) -> bool:
        """Whether the log satisfies all requested health gates."""
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "source": self.source,
            "passed": self.passed,
            "metrics": {
                name: [value if math.isfinite(value) else None for value in values]
                for name, values in self.metrics.items()
            },
            "summary": summarize_metric_series(self.metrics, self.tail_window),
            "findings": [asdict(finding) for finding in self.findings],
        }


REFERENCE_SCHEDULE = ReferenceSchedule()

EXPERIMENT_PROFILES = (
    ExperimentProfile("e4096_s64_mb16", num_envs=4096, solver_iterations=64),
    ExperimentProfile("e4096_s128_mb16", num_envs=4096, solver_iterations=128),
    ExperimentProfile("e1024_s64_mb16", num_envs=1024, solver_iterations=64),
    ExperimentProfile("e4096_s64_mb64", num_envs=4096, solver_iterations=64, num_mini_batches=64),
)

DIAGNOSTIC_PROFILES = (
    ExperimentProfile("e256_s64_mb16", num_envs=256, solver_iterations=64),
    ExperimentProfile("e256_s100_mb16", num_envs=256, solver_iterations=100),
    ExperimentProfile("e256_s128_mb16", num_envs=256, solver_iterations=128),
)

_ALL_PROFILES = EXPERIMENT_PROFILES + DIAGNOSTIC_PROFILES

_TASK_IDS = {
    Backend.PHYSX: "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference",
    Backend.NEWTON: "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference",
}

_METRIC_ALIASES = {
    "iteration": (),
    "success_rate": ("Metrics/success_rate", "success_rate"),
    "terminal_success_rate": ("Metrics/terminal_success_rate", "terminal_success_rate"),
    "total_fps": ("Perf/total_fps", "total_fps", "Total timesteps per second", "Steps per second"),
    "value_loss": ("Loss/value_function", "Mean value_function loss", "value_loss"),
    "mean_reward": ("Train/mean_reward", "Mean reward", "mean_reward"),
}

_NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf)"
_LEARNING_ITERATION_PATTERN = re.compile(
    rf"\bLearning iteration\s*:?\s*({_NUMBER_PATTERN})(?:\s*/\s*\d+)?",
    re.IGNORECASE,
)
_COMPUTATION_FPS_PATTERN = re.compile(
    rf"\bComputation\s*:\s*({_NUMBER_PATTERN})\s*(?:steps/s|step/s)",
    re.IGNORECASE,
)

_FATAL_LOG_PATTERNS = (
    ("python_traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("cuda_error", re.compile(r"CUDA (?:error|kernel error)", re.IGNORECASE)),
    ("cuda_illegal_access", re.compile(r"illegal memory access", re.IGNORECASE)),
    ("cuda_device_assert", re.compile(r"device-side assert", re.IGNORECASE)),
    ("out_of_memory", re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)),
    ("process_crash", re.compile(r"segmentation fault|core dumped|fatal signal", re.IGNORECASE)),
    ("non_finite", re.compile(r"(?<![\w.])(?:nan|[+-]?inf)(?![\w.])", re.IGNORECASE)),
)

_CAPACITY_ERROR_PATTERN = re.compile(
    r"overflow|dropped[- ]contact|contact[^\n]*(?:capacity|buffer)[^\n]*(?:exceed|full)|"
    r"triangle[-_ ]pair[^\n]*(?:exceed|overflow|full)",
    re.IGNORECASE,
)
_ZERO_CAPACITY_PATTERN = re.compile(r"(?:overflow|dropped[- ]contact)[^\n]*[:=]\s*0(?:\.0+)?\b",
                                    re.IGNORECASE)

VALIDATION_EVIDENCE_REQUIREMENTS = (
    MetricRequirement("scene.load_exit_code", exact=0.0),
    MetricRequirement("scene.non_finite_count", exact=0.0),
    MetricRequirement("reset_ik.p99_position_error_m", maximum=15.0e-6),
    MetricRequirement("reset_ik.max_position_error_m", maximum=81.0e-6),
    MetricRequirement("reset_ik.cap_hit_fraction", maximum=0.005),
    MetricRequirement("curriculum.max_spawn_overlap_m", maximum=0.00022),
    MetricRequirement("curriculum.max_contact_force_n", maximum=10.0),
    MetricRequirement("curriculum.max_tilt_rad", maximum=math.radians(2.0)),
    MetricRequirement("curriculum.termination_count", exact=0.0),
    MetricRequirement("seated_hold.seated_count", exact=8.0),
    MetricRequirement("seated_hold.max_drift_m", maximum=0.00025),
    MetricRequirement("seated_hold.max_overlap_m", maximum=0.00022),
    MetricRequirement("scripted_insertion.candidate_seated_count",
                      at_least_metric="scripted_insertion.baseline_seated_count"),
    MetricRequirement("scripted_insertion.max_overlap_m", maximum=0.0003),
    MetricRequirement("scripted_insertion.max_seat_depth_delta_m", maximum=0.00025),
    MetricRequirement("scripted_insertion.max_jam_depth_delta_m", maximum=0.00025),
    MetricRequirement("scripted_insertion.tunneling_count", exact=0.0),
    MetricRequirement("scripted_insertion.contact_regression_count", exact=0.0),
    MetricRequirement("offset_sweep.candidate_seated_count",
                      at_least_metric="offset_sweep.baseline_seated_count"),
    MetricRequirement("offset_sweep.max_overlap_m", maximum=0.0003),
    MetricRequirement("carry.lost_count", exact=0.0),
    MetricRequirement("carry.max_in_hand_drift_m", maximum=0.00025),
    MetricRequirement("properties.mass_matches", exact=True),
    MetricRequirement("properties.center_of_mass_matches", exact=True),
    MetricRequirement("properties.inertia_matches", exact=True),
    MetricRequirement("properties.active_colliders_match", exact=True),
    MetricRequirement("capacity.overflow_count", exact=0.0),
    MetricRequirement("capacity.dropped_contact_count", exact=0.0),
    MetricRequirement("capacity.reducer_insert_failure_count", exact=0.0),
    MetricRequirement("capacity.mjwarp_structural_overflow_flags", exact=0.0),
    MetricRequirement("capacity.max_collision_fill_fraction", maximum=0.8),
    MetricRequirement("capacity.max_reducer_fill_fraction", maximum=0.8),
    MetricRequirement("capacity.max_contact_fill_fraction", maximum=0.8),
    MetricRequirement("capacity.max_constraint_fill_fraction", maximum=0.8),
    MetricRequirement("capacity.peak_gpu_memory_fraction", maximum=0.8),
    MetricRequirement("render.assets_visible", exact=True),
    MetricRequirement("training.non_finite_count", exact=0.0),
    MetricRequirement("training.backend_error_count", exact=0.0),
    MetricRequirement("deployment.observation_order_matches", exact=True),
    MetricRequirement("deployment.max_action_parity_error", maximum=1.0e-5),
)


def sample_count(num_envs: int,
                 num_steps_per_env: int,
                 iterations: int,
                 world_size: int = 1) -> int:
    """Return the number of environment samples represented by a schedule."""
    values = {
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "iterations": iterations,
        "world_size": world_size,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return num_envs * num_steps_per_env * iterations * world_size


def sample_equivalent_iterations(
    *,
    reference_num_envs: int,
    reference_num_steps_per_env: int,
    reference_iterations: int,
    candidate_num_envs: int,
    candidate_num_steps_per_env: int,
    reference_world_size: int = 1,
    candidate_world_size: int = 1,
) -> int:
    """Return candidate iterations preserving the reference sample budget.

    The result rounds up so a candidate never receives fewer samples than the
    reference schedule.
    """
    reference_samples = sample_count(
        reference_num_envs,
        reference_num_steps_per_env,
        reference_iterations,
        reference_world_size,
    )
    samples_per_candidate_iteration = sample_count(
        candidate_num_envs,
        candidate_num_steps_per_env,
        iterations=1,
        world_size=candidate_world_size,
    )
    return math.ceil(reference_samples / samples_per_candidate_iteration)


def schedule_for_experiment(
    spec: ExperimentSpec,
    reference: ReferenceSchedule = REFERENCE_SCHEDULE,
) -> dict[str, int]:
    """Return sample-equivalent curriculum, checkpoint, and training schedules."""

    def convert(iterations: int) -> int:
        return sample_equivalent_iterations(
            reference_num_envs=reference.num_envs,
            reference_num_steps_per_env=reference.num_steps_per_env,
            reference_iterations=iterations,
            reference_world_size=reference.world_size,
            candidate_num_envs=spec.profile.num_envs,
            candidate_num_steps_per_env=spec.num_steps_per_env,
            candidate_world_size=spec.world_size,
        )

    return {
        "curriculum_anneal_end_iteration":
        convert(reference.curriculum_anneal_end_iteration),
        "save_interval":
        convert(reference.save_interval),
        "max_iterations": (spec.max_iterations if spec.max_iterations is not None else convert(
            reference.max_iterations(spec.backend))),
    }


def generate_experiment_matrix(
    *,
    backends: Sequence[Backend] = tuple(Backend),
    profiles: Sequence[ExperimentProfile] = EXPERIMENT_PROFILES,
    robot_assets: Sequence[RobotAsset] = tuple(RobotAsset),
    curriculum_modes: Sequence[bool] = (True, False),
    num_steps_per_env: int = 128,
    world_size: int = 1,
    seed: int = 123,
    max_iterations: int | None = None,
) -> list[ExperimentSpec]:
    """Build the full Cartesian product of controlled experiment dimensions."""
    return [
        ExperimentSpec(
            backend=backend,
            profile=profile,
            robot_asset=robot_asset,
            curriculum_enabled=curriculum_enabled,
            num_steps_per_env=num_steps_per_env,
            world_size=world_size,
            seed=seed,
            max_iterations=max_iterations,
        ) for backend in backends for profile in profiles for robot_asset in robot_assets
        for curriculum_enabled in curriculum_modes
    ]


def validate_experiment(
    spec: ExperimentSpec,
    *,
    limits: ValidatedBackendLimits = ValidatedBackendLimits(),
    calibrated_usd: Path | None = None,
    require_calibrated_usd_exists: bool = True,
) -> tuple[ConstraintViolation, ...]:
    """Return blocking safety violations for an experiment."""
    violations: list[ConstraintViolation] = []
    profile = spec.profile

    positive_values = [
        ("num_envs", profile.num_envs),
        ("solver_iterations", profile.solver_iterations),
        ("num_mini_batches", profile.num_mini_batches),
        ("num_steps_per_env", spec.num_steps_per_env),
        ("world_size", spec.world_size),
    ]
    if spec.max_iterations is not None:
        positive_values.append(("max_iterations", spec.max_iterations))
    for name, value in positive_values:
        if value <= 0:
            violations.append(
                ConstraintViolation("non_positive_value", f"{name} must be positive, got {value}"))

    if spec.world_size != 1:
        violations.append(
            ConstraintViolation(
                "distributed_command_not_supported",
                "The generated command is single-GPU; submit distributed runs through a "
                "dedicated workflow.",
            ))

    if profile.num_envs < profile.num_mini_batches:
        violations.append(
            ConstraintViolation(
                "too_many_minibatches",
                "Recurrent RSL-RL requires num_mini_batches no greater than num_envs.",
            ))

    if profile.num_envs % profile.num_mini_batches:
        violations.append(
            ConstraintViolation(
                "non_integral_minibatch",
                f"{profile.num_envs} environments per rank are not divisible by "
                f"{profile.num_mini_batches} recurrent minibatches.",
            ))

    validated_limit = limits.for_backend(spec.backend)
    if profile.num_envs > validated_limit:
        violations.append(
            ConstraintViolation(
                "environment_count_not_validated",
                f"{profile.num_envs} environments/GPU exceeds the evidence-backed "
                f"{spec.backend} limit "
                f"of {validated_limit}; run capacity and memory diagnostics before submission.",
            ))

    if spec.robot_asset is RobotAsset.CALIBRATED:
        if calibrated_usd is None:
            violations.append(
                ConstraintViolation(
                    "calibrated_usd_missing",
                    "A calibrated experiment requires --calibrated_usd.",
                ))
        elif not calibrated_usd.is_absolute():
            violations.append(
                ConstraintViolation(
                    "calibrated_usd_not_absolute",
                    f"The calibrated USD path must be absolute: {calibrated_usd}",
                ))
        elif require_calibrated_usd_exists and not calibrated_usd.is_file():
            violations.append(
                ConstraintViolation(
                    "calibrated_usd_not_found",
                    f"The calibrated USD does not exist: {calibrated_usd}",
                ))

    return tuple(violations)


def training_command(
    spec: ExperimentSpec,
    *,
    calibrated_usd: Path | None = None,
    reference: ReferenceSchedule = REFERENCE_SCHEDULE,
) -> list[str]:
    """Build a local training command with backend-specific solver overrides."""
    if spec.world_size != 1:
        raise ValueError("training_command only generates single-GPU commands")
    schedule = schedule_for_experiment(spec, reference)
    profile = spec.profile
    command = [
        "uv",
        "run",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        _TASK_IDS[spec.backend],
        "--num_envs",
        str(profile.num_envs),
        "--max_iterations",
        str(schedule["max_iterations"]),
        "--seed",
        str(spec.seed),
        "--run_name",
        spec.name,
        "--visualizer",
        "none",
    ]
    if spec.backend is Backend.NEWTON:
        command.extend([
            "physics=newton_sdf",
            f"env.sim.physics.solver_cfg.iterations={profile.solver_iterations}",
        ])
    else:
        command.extend([
            f"env.scene.dp_plug.spawn.rigid_props."
            f"solver_position_iteration_count={profile.solver_iterations}",
            f"env.scene.dp_socket.spawn.rigid_props."
            f"solver_position_iteration_count={profile.solver_iterations}",
        ])

    command.extend([
        f"agent.num_steps_per_env={spec.num_steps_per_env}",
        f"agent.save_interval={schedule['save_interval']}",
        f"agent.algorithm.num_mini_batches={profile.num_mini_batches}",
    ])
    if spec.curriculum_enabled:
        command.extend([
            "env.events.reset_plug_curriculum.params.at_goal_prob=0.8",
            "env.events.reset_plug_curriculum.params.at_goal_prob_final=0.0",
            f"env.events.reset_plug_curriculum.params."
            f"anneal_end_iter={schedule['curriculum_anneal_end_iteration']}",
            f"env.events.reset_plug_curriculum.params.num_steps_per_env={spec.num_steps_per_env}",
        ])
    else:
        command.extend([
            "env.events.reset_plug_curriculum.params.at_goal_prob=0.0",
            "env.events.reset_plug_curriculum.params.at_goal_prob_final=0.0",
        ])

    if spec.robot_asset is RobotAsset.CALIBRATED:
        if calibrated_usd is None:
            raise ValueError("calibrated_usd is required for a calibrated experiment")
        command.append(f"env.scene.robot.spawn.usd_path={calibrated_usd}")
    return command


def extract_metric_series(text: str) -> dict[str, list[float]]:
    """Extract known scalar series from RSL-RL console or exported text logs."""
    series = {name: [] for name in _METRIC_ALIASES}
    for line in text.splitlines():
        iteration_match = _LEARNING_ITERATION_PATTERN.search(line)
        if iteration_match:
            series["iteration"].append(float(iteration_match.group(1)))
        computation_fps_match = _COMPUTATION_FPS_PATTERN.search(line)
        if computation_fps_match:
            series["total_fps"].append(float(computation_fps_match.group(1)))
        for canonical_name, aliases in _METRIC_ALIASES.items():
            for alias in aliases:
                label_pattern = re.escape(
                    alias) if "/" in alias else rf"(?<![\w/]){re.escape(alias)}(?![\w/])"
                match = re.search(
                    rf"{label_pattern}\s*[:=]\s*({_NUMBER_PATTERN})",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    series[canonical_name].append(float(match.group(1)))
                    break
    return {name: values for name, values in series.items() if values}


def summarize_metric_series(metrics: Mapping[str, Sequence[float]],
                            tail_window: int = 20) -> dict[str, Any]:
    """Summarize finite metric series over their final window."""
    if tail_window <= 0:
        raise ValueError(f"tail_window must be positive, got {tail_window}")
    result: dict[str, Any] = {}
    for name, values in metrics.items():
        if not values:
            continue
        tail = list(values[-tail_window:])
        finite_tail = [value for value in tail if math.isfinite(value)]
        result[name] = {
            "count": len(values),
            "final": values[-1] if math.isfinite(values[-1]) else None,
            "tail_count": len(tail),
            "tail_median": statistics.median(finite_tail) if finite_tail else None,
            "minimum": min(finite_tail) if finite_tail else None,
            "maximum": max(finite_tail) if finite_tail else None,
        }
    return result


def analyze_training_log(
    text: str,
    *,
    source: str = "<memory>",
    thresholds: TrainingHealthThresholds = TrainingHealthThresholds(),
    baseline_text: str | None = None,
) -> TrainingHealthReport:
    """Check training logs for failures, capacity errors, and scalar thresholds."""
    if thresholds.tail_window <= 0:
        raise ValueError(f"tail_window must be positive, got {thresholds.tail_window}")
    if thresholds.minimum_iterations <= 0:
        raise ValueError(
            f"minimum_iterations must be positive, got {thresholds.minimum_iterations}")
    for name, value in (
        ("minimum_final_terminal_success", thresholds.minimum_final_terminal_success),
        ("minimum_tail_terminal_success", thresholds.minimum_tail_terminal_success),
        ("minimum_tail_median_fps", thresholds.minimum_tail_median_fps),
        ("minimum_fps_ratio_to_baseline", thresholds.minimum_fps_ratio_to_baseline),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    for name, value in (
        ("minimum_final_terminal_success", thresholds.minimum_final_terminal_success),
        ("minimum_tail_terminal_success", thresholds.minimum_tail_terminal_success),
    ):
        if value is not None and value > 1.0:
            raise ValueError(f"{name} must not exceed 1.0, got {value}")

    findings: list[HealthFinding] = []
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for code, pattern in _FATAL_LOG_PATTERNS:
            if pattern.search(line):
                findings.append(HealthFinding(code, line.strip(), line_number))
        if _CAPACITY_ERROR_PATTERN.search(line) and not _ZERO_CAPACITY_PATTERN.search(line):
            findings.append(HealthFinding("contact_capacity", line.strip(), line_number))

    metrics = extract_metric_series(text)
    for name, values in metrics.items():
        for index, value in enumerate(values):
            if not math.isfinite(value):
                findings.append(HealthFinding("non_finite_metric", f"{name}[{index}] is {value}"))

    iteration_count = len(metrics.get("iteration", []))
    if iteration_count < thresholds.minimum_iterations:
        findings.append(
            HealthFinding(
                "too_few_iterations",
                f"Found {iteration_count} iteration records; require at least "
                f"{thresholds.minimum_iterations}.",
            ))

    terminal_success = metrics.get("terminal_success_rate", [])
    if thresholds.minimum_final_terminal_success is not None:
        if not terminal_success:
            findings.append(
                HealthFinding("missing_terminal_success", "No terminal-success metric was found."))
        elif terminal_success[-1] < thresholds.minimum_final_terminal_success:
            findings.append(
                HealthFinding(
                    "low_final_terminal_success",
                    f"Final terminal success {terminal_success[-1]:.6g} is below "
                    f"{thresholds.minimum_final_terminal_success:.6g}.",
                ))

    if thresholds.minimum_tail_terminal_success is not None:
        if not terminal_success:
            findings.append(
                HealthFinding("missing_terminal_success", "No terminal-success metric was found."))
        else:
            tail = terminal_success[-thresholds.tail_window:]
            median = statistics.median(tail)
            if median < thresholds.minimum_tail_terminal_success:
                findings.append(
                    HealthFinding(
                        "low_tail_terminal_success",
                        f"Tail terminal-success median {median:.6g} is below "
                        f"{thresholds.minimum_tail_terminal_success:.6g}.",
                    ))

    fps = metrics.get("total_fps", [])
    if thresholds.minimum_tail_median_fps is not None:
        if not fps:
            findings.append(HealthFinding("missing_fps", "No throughput metric was found."))
        else:
            median_fps = statistics.median(fps[-thresholds.tail_window:])
            if median_fps < thresholds.minimum_tail_median_fps:
                findings.append(
                    HealthFinding(
                        "low_throughput",
                        f"Tail FPS median {median_fps:.6g} is below "
                        f"{thresholds.minimum_tail_median_fps:.6g}.",
                    ))

    if thresholds.minimum_fps_ratio_to_baseline is not None:
        if baseline_text is None:
            findings.append(
                HealthFinding("missing_baseline", "FPS-ratio validation requires a baseline log."))
        else:
            baseline_fps = extract_metric_series(baseline_text).get("total_fps", [])
            if not fps or not baseline_fps:
                findings.append(
                    HealthFinding("missing_fps",
                                  "Both candidate and baseline logs require throughput metrics."))
            elif any(not math.isfinite(value) or value <= 0.0 for value in fps):
                findings.append(
                    HealthFinding("invalid_fps",
                                  "Candidate throughput metrics must be finite and positive."))
            elif any(not math.isfinite(value) or value <= 0.0 for value in baseline_fps):
                findings.append(
                    HealthFinding("invalid_baseline_fps",
                                  "Baseline throughput metrics must be finite and positive."))
            else:
                candidate_median = statistics.median(fps[-thresholds.tail_window:])
                baseline_median = statistics.median(baseline_fps[-thresholds.tail_window:])
                ratio = candidate_median / baseline_median if baseline_median else 0.0
                if ratio < thresholds.minimum_fps_ratio_to_baseline:
                    findings.append(
                        HealthFinding(
                            "fps_regression",
                            f"Candidate/baseline tail FPS ratio {ratio:.6g} is below "
                            f"{thresholds.minimum_fps_ratio_to_baseline:.6g}.",
                        ))

    unique_findings = tuple(dict.fromkeys(findings))
    return TrainingHealthReport(
        metrics=metrics,
        findings=unique_findings,
        source=source,
        tail_window=thresholds.tail_window,
    )


def validation_evidence_template() -> dict[str, Any]:
    """Return an intentionally failing template for measurements from real GPU probes."""
    metric_names = {requirement.name for requirement in VALIDATION_EVIDENCE_REQUIREMENTS}
    metric_names.update(requirement.at_least_metric
                        for requirement in VALIDATION_EVIDENCE_REQUIREMENTS
                        if requirement.at_least_metric is not None)
    return {
        "schema_version": 1,
        "metadata": {
            "baseline_git_commit": None,
            "candidate_git_commit": None,
            "backend": None,
            "candidate_label": None,
            "baseline_label": None,
            "seed": None,
            "gpu": None,
            "baseline_command": None,
            "candidate_command": None,
        },
        "metrics": {
            name: None
            for name in sorted(metric_names)
        },
        "artifacts": {
            "logs": [],
            "videos": [],
            "tables": []
        },
    }


def _validate_artifact_reference(
    artifact_type: str,
    reference: str,
    artifact_root: Path,
) -> ConstraintViolation | None:
    """Validate one local artifact reference relative to ``artifact_root``."""
    reference_path = Path(reference)
    if reference_path.is_absolute() or "://" in reference:
        return ConstraintViolation(
            "artifact_path",
            f"artifacts.{artifact_type} reference {reference!r} must be a relative local path.",
        )

    try:
        candidate = (artifact_root / reference_path).resolve()
        candidate.relative_to(artifact_root)
    except (OSError, RuntimeError, ValueError):
        return ConstraintViolation(
            "artifact_path",
            f"artifacts.{artifact_type} reference {reference!r} escapes the artifact root.",
        )

    if not candidate.exists():
        return ConstraintViolation(
            "artifact_missing",
            f"artifacts.{artifact_type} reference {reference!r} does not exist.",
        )
    if not candidate.is_file():
        return ConstraintViolation(
            "artifact_not_file",
            f"artifacts.{artifact_type} reference {reference!r} is not a regular file.",
        )
    try:
        is_empty = candidate.stat().st_size == 0
    except OSError as error:
        return ConstraintViolation(
            "artifact_unreadable",
            f"artifacts.{artifact_type} reference {reference!r} cannot be inspected: {error}.",
        )
    if is_empty:
        return ConstraintViolation(
            "artifact_empty",
            f"artifacts.{artifact_type} reference {reference!r} is empty.",
        )
    return None


def validate_evidence_document(
    document: object,
    *,
    artifact_root: Path | None = None,
) -> tuple[ConstraintViolation, ...]:
    """Validate measured GPU evidence against the release-gate contract.

    When ``artifact_root`` is omitted, artifact references are checked as schema
    values only. Supplying a root additionally requires every reference to name
    a non-empty regular file contained beneath that directory.
    """
    if not isinstance(document, Mapping):
        return (ConstraintViolation("document_type", "Evidence document must be an object."), )

    violations: list[ConstraintViolation] = []
    if isinstance(document.get("schema_version"), bool) or document.get("schema_version") != 1:
        violations.append(
            ConstraintViolation("schema_version", "Evidence schema_version must be 1."))
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        violations.append(
            ConstraintViolation("metadata_missing", "Evidence metadata must be an object."))
    else:
        string_fields = (
            "baseline_git_commit",
            "candidate_git_commit",
            "backend",
            "candidate_label",
            "baseline_label",
            "gpu",
            "baseline_command",
            "candidate_command",
        )
        for name in string_fields:
            if not isinstance(metadata.get(name), str) or not metadata[name]:
                violations.append(
                    ConstraintViolation("metadata_missing", f"metadata.{name} is required."))
        for name in ("baseline_git_commit", "candidate_git_commit"):
            git_commit = metadata.get(name)
            if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}",
                                                               git_commit) is None:
                violations.append(
                    ConstraintViolation("metadata_value",
                                        f"metadata.{name} must be a 40-digit SHA."))
        backend = metadata.get("backend")
        if not isinstance(backend, str) or backend not in tuple(item.value for item in Backend):
            violations.append(
                ConstraintViolation("metadata_value",
                                    "metadata.backend must be either 'physx' or 'newton'."))
        if not isinstance(metadata.get("seed"), int) or isinstance(metadata.get("seed"), bool):
            violations.append(
                ConstraintViolation("metadata_value", "metadata.seed must be an integer."))

    resolved_artifact_root = artifact_root.resolve() if artifact_root is not None else None
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        violations.append(
            ConstraintViolation("artifacts_missing", "Evidence artifacts must be an object."))
    else:
        for artifact_type in ("logs", "videos", "tables"):
            entries = artifacts.get(artifact_type)
            if (not isinstance(entries, list) or not entries
                    or any(not isinstance(entry, str) or not entry.strip() for entry in entries)):
                violations.append(
                    ConstraintViolation(
                        "artifacts_missing",
                        f"artifacts.{artifact_type} must contain at least one non-empty "
                        f"artifact reference.",
                    ))
                continue
            if resolved_artifact_root is not None:
                for entry in entries:
                    violation = _validate_artifact_reference(
                        artifact_type,
                        entry,
                        resolved_artifact_root,
                    )
                    if violation is not None:
                        violations.append(violation)

    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        return (*violations,
                ConstraintViolation("metrics_missing", "Evidence metrics must be an object."))

    for requirement in VALIDATION_EVIDENCE_REQUIREMENTS:
        value = metrics.get(requirement.name)
        if value is None:
            violations.append(
                ConstraintViolation("metric_missing", f"{requirement.name} is required."))
            continue
        expects_boolean = isinstance(requirement.exact, bool)
        if expects_boolean and not isinstance(value, bool):
            violations.append(
                ConstraintViolation(
                    "metric_type",
                    f"{requirement.name} must be Boolean, got {type(value).__name__}."))
            continue
        if not expects_boolean and (isinstance(value, bool) or not isinstance(value, int | float)):
            violations.append(
                ConstraintViolation(
                    "metric_type",
                    f"{requirement.name} must be numeric, got {type(value).__name__}."))
            continue
        if not expects_boolean and not math.isfinite(float(value)):
            violations.append(
                ConstraintViolation("metric_non_finite", f"{requirement.name} must be finite."))
            continue
        if not expects_boolean and float(value) < 0.0:
            violations.append(
                ConstraintViolation("metric_minimum",
                                    f"{requirement.name} must be nonnegative, got {value!r}."))
            continue
        if requirement.exact is not None and value != requirement.exact:
            violations.append(
                ConstraintViolation(
                    "metric_exact",
                    f"{requirement.name}={value!r}, expected {requirement.exact!r}.",
                ))
        if requirement.maximum is not None and float(value) > requirement.maximum:
            violations.append(
                ConstraintViolation(
                    "metric_maximum",
                    f"{requirement.name}={float(value):.9g}, maximum {requirement.maximum:.9g}.",
                ))
        if requirement.minimum is not None and float(value) < requirement.minimum:
            violations.append(
                ConstraintViolation(
                    "metric_minimum",
                    f"{requirement.name}={float(value):.9g}, minimum {requirement.minimum:.9g}.",
                ))
        if requirement.at_least_metric is not None:
            baseline = metrics.get(requirement.at_least_metric)
            if (isinstance(baseline, bool) or not isinstance(baseline, int | float)
                    or not math.isfinite(float(baseline)) or float(baseline) < 0.0):
                violations.append(
                    ConstraintViolation(
                        "reference_metric_missing",
                        f"{requirement.at_least_metric} is required to evaluate "
                        f"{requirement.name}.",
                    ))
            elif float(value) < float(baseline):
                violations.append(
                    ConstraintViolation(
                        "metric_regression",
                        f"{requirement.name}={float(value):.9g} is below "
                        f"{requirement.at_least_metric}={float(baseline):.9g}.",
                    ))
    return tuple(violations)


def _spec_to_dict(
    spec: ExperimentSpec,
    *,
    violations: Sequence[ConstraintViolation],
    calibrated_usd: Path | None,
) -> dict[str, Any]:
    schedule = schedule_for_experiment(spec)
    result = {
        "name":
        spec.name,
        "backend":
        spec.backend,
        "profile":
        asdict(spec.profile),
        "robot_asset":
        spec.robot_asset,
        "curriculum_enabled":
        spec.curriculum_enabled,
        "num_steps_per_env":
        spec.num_steps_per_env,
        "world_size":
        spec.world_size,
        "seed":
        spec.seed,
        "schedule":
        schedule,
        "rollout_samples":
        spec.profile.num_envs * spec.num_steps_per_env * spec.world_size,
        "samples_per_minibatch": (spec.profile.num_envs * spec.num_steps_per_env *
                                  spec.world_size // spec.profile.num_mini_batches),
        "envs_per_recurrent_minibatch":
        spec.profile.num_envs // spec.profile.num_mini_batches,
        "runnable":
        not violations,
        "violations": [asdict(violation) for violation in violations],
    }
    if not violations:
        result["command"] = training_command(spec, calibrated_usd=calibrated_usd)
    return result


def _select_enum(value: str, enum_type: type[Backend] | type[RobotAsset]) -> list[Any]:
    if value == "all" or value == "both":
        return list(enum_type)
    return [enum_type(value)]


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _add_matrix_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("matrix",
                                   help="Generate the controlled optimization experiment matrix.")
    parser.add_argument("--backend", choices=("all", *Backend), default="all")
    parser.add_argument("--asset", choices=("both", *RobotAsset), default="both")
    parser.add_argument("--curriculum", choices=("both", "on", "off"), default="both")
    parser.add_argument("--profile",
                        action="append",
                        choices=[profile.name for profile in _ALL_PROFILES])
    parser.add_argument("--calibrated_usd", type=Path)
    parser.add_argument("--max_iterations", type=_positive_int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--format", choices=("json", "commands", "table"), default="table")
    parser.add_argument("--validated_physx_envs_per_gpu", type=_positive_int, default=1024)
    parser.add_argument("--validated_newton_envs_per_gpu", type=_positive_int, default=256)
    parser.add_argument(
        "--allow_missing_calibrated_usd",
        action="store_true",
        help="Plan with a not-yet-staged path; such output is not an executable release gate.",
    )


def _add_sample_budget_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("sample_budget",
                                   help="Convert an iteration schedule at equal samples.")
    parser.add_argument("--reference_envs", type=_positive_int, required=True)
    parser.add_argument("--reference_steps", type=_positive_int, required=True)
    parser.add_argument("--reference_iterations", type=_positive_int, required=True)
    parser.add_argument("--candidate_envs", type=_positive_int, required=True)
    parser.add_argument("--candidate_steps", type=_positive_int, required=True)
    parser.add_argument("--reference_world_size", type=_positive_int, default=1)
    parser.add_argument("--candidate_world_size", type=_positive_int, default=1)


def _add_analyze_log_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("analyze_log", help="Analyze one RSL-RL training console log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tail_window", type=_positive_int, default=20)
    parser.add_argument("--minimum_iterations", type=_positive_int, default=1)
    parser.add_argument("--minimum_final_terminal_success", type=float)
    parser.add_argument("--minimum_tail_terminal_success", type=float)
    parser.add_argument("--minimum_tail_median_fps", type=float)
    parser.add_argument("--minimum_fps_ratio_to_baseline", type=float)


def _run_matrix(args: argparse.Namespace) -> int:
    backends = _select_enum(args.backend, Backend)
    robot_assets = _select_enum(args.asset, RobotAsset)
    curriculum_modes = (True, False) if args.curriculum == "both" else (args.curriculum == "on", )
    profiles_by_name = {profile.name: profile for profile in _ALL_PROFILES}
    profiles = [profiles_by_name[name]
                for name in args.profile] if args.profile else list(EXPERIMENT_PROFILES)
    limits = ValidatedBackendLimits(
        physx_envs_per_gpu=args.validated_physx_envs_per_gpu,
        newton_envs_per_gpu=args.validated_newton_envs_per_gpu,
    )
    rows = []
    for spec in generate_experiment_matrix(
            backends=backends,
            profiles=profiles,
            robot_assets=robot_assets,
            curriculum_modes=curriculum_modes,
            seed=args.seed,
            max_iterations=args.max_iterations,
    ):
        violations = validate_experiment(
            spec,
            limits=limits,
            calibrated_usd=args.calibrated_usd,
            require_calibrated_usd_exists=not args.allow_missing_calibrated_usd,
        )
        rows.append(_spec_to_dict(spec, violations=violations, calibrated_usd=args.calibrated_usd))

    if args.format == "json":
        print(json.dumps(rows, indent=2, default=str))
    elif args.format == "commands":
        for row in rows:
            if row["runnable"]:
                print(shlex.join(row["command"]))
            else:
                reasons = "; ".join(item["message"] for item in row["violations"])
                print(f"# BLOCKED {row['name']}: {reasons}")
    else:
        print("name\trunnable\tanneal_end\tmax_iterations\tsamples/minibatch\treason")
        for row in rows:
            reason = "; ".join(item["code"] for item in row["violations"])
            print(f"{row['name']}\t{row['runnable']}\t"
                  f"{row['schedule']['curriculum_anneal_end_iteration']}\t"
                  f"{row['schedule']['max_iterations']}\t{row['samples_per_minibatch']}\t{reason}")
    return 0 if all(row["runnable"] for row in rows) else 2


def _run_sample_budget(args: argparse.Namespace) -> int:
    iterations = sample_equivalent_iterations(
        reference_num_envs=args.reference_envs,
        reference_num_steps_per_env=args.reference_steps,
        reference_iterations=args.reference_iterations,
        candidate_num_envs=args.candidate_envs,
        candidate_num_steps_per_env=args.candidate_steps,
        reference_world_size=args.reference_world_size,
        candidate_world_size=args.candidate_world_size,
    )
    result = {
        "candidate_iterations":
        iterations,
        "reference_samples":
        sample_count(
            args.reference_envs,
            args.reference_steps,
            args.reference_iterations,
            args.reference_world_size,
        ),
        "candidate_samples":
        sample_count(
            args.candidate_envs,
            args.candidate_steps,
            iterations,
            args.candidate_world_size,
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


def _run_analyze_log(args: argparse.Namespace) -> int:
    thresholds = TrainingHealthThresholds(
        tail_window=args.tail_window,
        minimum_iterations=args.minimum_iterations,
        minimum_final_terminal_success=args.minimum_final_terminal_success,
        minimum_tail_terminal_success=args.minimum_tail_terminal_success,
        minimum_tail_median_fps=args.minimum_tail_median_fps,
        minimum_fps_ratio_to_baseline=args.minimum_fps_ratio_to_baseline,
    )
    baseline_text = args.baseline.read_text(encoding="utf-8") if args.baseline else None
    report = analyze_training_log(
        args.log.read_text(encoding="utf-8"),
        source=str(args.log),
        thresholds=thresholds,
        baseline_text=baseline_text,
    )
    print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    return 0 if report.passed else 1


def _run_validate_evidence(args: argparse.Namespace) -> int:
    evidence_path = args.evidence.resolve()
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    violations = validate_evidence_document(document, artifact_root=evidence_path.parent)
    result = {
        "source": str(evidence_path),
        "passed": not violations,
        "violations": [asdict(violation) for violation in violations],
    }
    print(json.dumps(result, indent=2))
    return 0 if not violations else 1


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_matrix_parser(subparsers)
    _add_sample_budget_parser(subparsers)
    _add_analyze_log_parser(subparsers)
    subparsers.add_parser("evidence_template", help="Print the required real-GPU evidence schema.")
    evidence_parser = subparsers.add_parser("validate_evidence",
                                            help="Validate measurements from real GPU probes.")
    evidence_parser.add_argument("evidence", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the DisplayPort optimization validation utility."""
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.command == "matrix":
        return _run_matrix(args)
    if args.command == "sample_budget":
        return _run_sample_budget(args)
    if args.command == "analyze_log":
        return _run_analyze_log(args)
    if args.command == "evidence_template":
        print(json.dumps(validation_evidence_template(), indent=2))
        return 0
    if args.command == "validate_evidence":
        return _run_validate_evidence(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
