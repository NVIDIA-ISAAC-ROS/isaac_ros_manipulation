# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the DisplayPort training-optimization validation utility."""

import json
from pathlib import Path

from isaaclab_training.cli.displayport_training_validation import analyze_training_log
from isaaclab_training.cli.displayport_training_validation import Backend
from isaaclab_training.cli.displayport_training_validation import EXPERIMENT_PROFILES
from isaaclab_training.cli.displayport_training_validation import ExperimentProfile
from isaaclab_training.cli.displayport_training_validation import ExperimentSpec
from isaaclab_training.cli.displayport_training_validation import generate_experiment_matrix
from isaaclab_training.cli.displayport_training_validation import main
from isaaclab_training.cli.displayport_training_validation import RobotAsset
from isaaclab_training.cli.displayport_training_validation import sample_count
from isaaclab_training.cli.displayport_training_validation import sample_equivalent_iterations
from isaaclab_training.cli.displayport_training_validation import schedule_for_experiment
from isaaclab_training.cli.displayport_training_validation import training_command
from isaaclab_training.cli.displayport_training_validation import TrainingHealthThresholds
from isaaclab_training.cli.displayport_training_validation import validate_evidence_document
from isaaclab_training.cli.displayport_training_validation import validate_experiment
from isaaclab_training.cli.displayport_training_validation import ValidatedBackendLimits
from isaaclab_training.cli.displayport_training_validation import VALIDATION_EVIDENCE_REQUIREMENTS
from isaaclab_training.cli.displayport_training_validation import validation_evidence_template
import pytest


def _spec(
    backend: Backend = Backend.PHYSX,
    *,
    num_envs: int = 4096,
    solver_iterations: int = 64,
    num_mini_batches: int = 16,
    asset: RobotAsset = RobotAsset.NOMINAL,
    curriculum_enabled: bool = True,
) -> ExperimentSpec:
    return ExperimentSpec(
        backend=backend,
        profile=ExperimentProfile(
            "test_profile",
            num_envs=num_envs,
            solver_iterations=solver_iterations,
            num_mini_batches=num_mini_batches,
        ),
        robot_asset=asset,
        curriculum_enabled=curriculum_enabled,
    )


def _passing_evidence() -> dict:
    document = validation_evidence_template()
    document["metadata"] = {
        "baseline_git_commit": "0" * 40,
        "candidate_git_commit": "1" * 40,
        "backend": "newton",
        "candidate_label": "candidate",
        "baseline_label": "baseline",
        "seed": 123,
        "gpu": "L40S",
        "baseline_command": "uv run isaaclab train baseline",
        "candidate_command": "uv run isaaclab train candidate",
    }
    values = {
        "scene.load_exit_code": 0,
        "scene.non_finite_count": 0,
        "reset_ik.p99_position_error_m": 10.0e-6,
        "reset_ik.max_position_error_m": 80.0e-6,
        "reset_ik.cap_hit_fraction": 0.001,
        "curriculum.max_spawn_overlap_m": 0.0002,
        "curriculum.max_contact_force_n": 5.0,
        "curriculum.max_tilt_rad": 0.02,
        "curriculum.termination_count": 0,
        "seated_hold.seated_count": 8,
        "seated_hold.max_drift_m": 0.0002,
        "seated_hold.max_overlap_m": 0.0002,
        "scripted_insertion.baseline_seated_count": 7,
        "scripted_insertion.candidate_seated_count": 7,
        "scripted_insertion.max_overlap_m": 0.0002,
        "scripted_insertion.max_seat_depth_delta_m": 0.0001,
        "scripted_insertion.max_jam_depth_delta_m": 0.0001,
        "scripted_insertion.tunneling_count": 0,
        "scripted_insertion.contact_regression_count": 0,
        "offset_sweep.baseline_seated_count": 8,
        "offset_sweep.candidate_seated_count": 8,
        "offset_sweep.max_overlap_m": 0.0002,
        "carry.lost_count": 0,
        "carry.max_in_hand_drift_m": 0.0002,
        "properties.mass_matches": True,
        "properties.center_of_mass_matches": True,
        "properties.inertia_matches": True,
        "properties.active_colliders_match": True,
        "capacity.overflow_count": 0,
        "capacity.dropped_contact_count": 0,
        "capacity.reducer_insert_failure_count": 0,
        "capacity.mjwarp_structural_overflow_flags": 0,
        "capacity.max_collision_fill_fraction": 0.7,
        "capacity.max_reducer_fill_fraction": 0.7,
        "capacity.max_contact_fill_fraction": 0.7,
        "capacity.max_constraint_fill_fraction": 0.7,
        "capacity.peak_gpu_memory_fraction": 0.7,
        "render.assets_visible": True,
        "training.non_finite_count": 0,
        "training.backend_error_count": 0,
        "deployment.observation_order_matches": True,
        "deployment.max_action_parity_error": 1.0e-6,
    }
    document["metrics"].update(values)
    document["artifacts"] = {
        "logs": ["artifacts/training.log"],
        "videos": ["artifacts/scripted_insertion.mp4"],
        "tables": ["artifacts/physics_metrics.json"],
    }
    return document


def test_sample_budget_preserves_reference_exactly():
    reference_samples = sample_count(1024, 512, 500)

    candidate_iterations = sample_equivalent_iterations(
        reference_num_envs=1024,
        reference_num_steps_per_env=512,
        reference_iterations=500,
        candidate_num_envs=4096,
        candidate_num_steps_per_env=128,
    )

    assert candidate_iterations == 500
    assert sample_count(4096, 128, candidate_iterations) == reference_samples


def test_sample_budget_rounds_up_instead_of_shortchanging_candidate():
    candidate_iterations = sample_equivalent_iterations(
        reference_num_envs=10,
        reference_num_steps_per_env=10,
        reference_iterations=3,
        candidate_num_envs=8,
        candidate_num_steps_per_env=8,
    )

    assert candidate_iterations == 5
    assert sample_count(8, 8, candidate_iterations) >= sample_count(10, 10, 3)


@pytest.mark.parametrize("field", ("num_envs", "num_steps_per_env", "iterations", "world_size"))
def test_sample_count_rejects_non_positive_inputs(field: str):
    values = {"num_envs": 1, "num_steps_per_env": 1, "iterations": 1, "world_size": 1}
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        sample_count(**values)


@pytest.mark.parametrize(
    ("backend", "num_envs", "expected_anneal", "expected_max_iterations", "expected_save"),
    [
        (Backend.PHYSX, 4096, 500, 1500, 50),
        (Backend.PHYSX, 1024, 2000, 6000, 200),
        (Backend.NEWTON, 4096, 500, 1000, 50),
        (Backend.NEWTON, 1024, 2000, 4000, 200),
    ],
)
def test_schedule_converts_every_iteration_based_setting(
    backend: Backend,
    num_envs: int,
    expected_anneal: int,
    expected_max_iterations: int,
    expected_save: int,
):
    schedule = schedule_for_experiment(_spec(backend, num_envs=num_envs))

    assert schedule == {
        "curriculum_anneal_end_iteration": expected_anneal,
        "save_interval": expected_save,
        "max_iterations": expected_max_iterations,
    }


def test_full_matrix_contains_every_requested_dimension_once():
    matrix = generate_experiment_matrix()

    assert len(matrix) == 2 * len(EXPERIMENT_PROFILES) * 2 * 2
    assert len({spec.name for spec in matrix}) == len(matrix)
    assert {spec.backend for spec in matrix} == set(Backend)
    assert {spec.robot_asset for spec in matrix} == set(RobotAsset)
    assert {spec.curriculum_enabled for spec in matrix} == {True, False}


def test_default_backend_limits_block_unqualified_large_jobs():
    physx_violations = validate_experiment(_spec(Backend.PHYSX, num_envs=4096))
    newton_violations = validate_experiment(_spec(Backend.NEWTON, num_envs=1024))

    assert [item.code for item in physx_violations] == ["environment_count_not_validated"]
    assert [item.code for item in newton_violations] == ["environment_count_not_validated"]


def test_evidence_backed_limits_unlock_only_the_selected_backend():
    limits = ValidatedBackendLimits(physx_envs_per_gpu=4096, newton_envs_per_gpu=256)

    assert validate_experiment(_spec(Backend.PHYSX, num_envs=4096), limits=limits) == ()
    assert validate_experiment(_spec(Backend.NEWTON, num_envs=4096), limits=limits)


def test_calibrated_experiment_requires_existing_absolute_usd(tmp_path: Path):
    calibrated = _spec(asset=RobotAsset.CALIBRATED, num_envs=1024)

    assert {item.code for item in validate_experiment(calibrated)} == {"calibrated_usd_missing"}
    assert {
        item.code
        for item in validate_experiment(calibrated, calibrated_usd=Path("relative.usd"))
    } == {"calibrated_usd_not_absolute"}
    missing = tmp_path / "missing.usd"
    assert {item.code
            for item in validate_experiment(calibrated, calibrated_usd=missing)
            } == {"calibrated_usd_not_found"}
    existing = tmp_path / "calibrated.usd"
    existing.touch()
    assert validate_experiment(calibrated, calibrated_usd=existing) == ()


def test_minibatch_contract_rejects_non_integral_batches():
    spec = ExperimentSpec(
        backend=Backend.PHYSX,
        profile=ExperimentProfile("bad", num_envs=3, solver_iterations=64, num_mini_batches=5),
        robot_asset=RobotAsset.NOMINAL,
        curriculum_enabled=True,
        num_steps_per_env=7,
    )

    assert {item.code
            for item in validate_experiment(spec)} == {
                "too_many_minibatches",
                "non_integral_minibatch",
            }


def test_recurrent_minibatches_require_environment_divisibility_per_rank():
    spec = _spec(num_envs=5, num_mini_batches=2)

    assert {item.code for item in validate_experiment(spec)} == {"non_integral_minibatch"}


def test_single_gpu_command_builder_rejects_distributed_schedule():
    spec = ExperimentSpec(
        backend=Backend.PHYSX,
        profile=ExperimentProfile("distributed", num_envs=1024, solver_iterations=64),
        robot_asset=RobotAsset.NOMINAL,
        curriculum_enabled=True,
        world_size=2,
    )

    assert {item.code
            for item in validate_experiment(spec)} == {"distributed_command_not_supported"}
    with pytest.raises(ValueError, match="single-GPU"):
        training_command(spec)


def test_physx_command_uses_physx_only_solver_fields():
    command = training_command(_spec(Backend.PHYSX, num_envs=1024))
    joined = " ".join(command)

    assert "dp_plug.spawn.rigid_props.solver_position_iteration_count=64" in joined
    assert "dp_socket.spawn.rigid_props.solver_position_iteration_count=64" in joined
    assert "physics=newton_sdf" not in command
    assert "solver_cfg.iterations" not in joined


def test_newton_command_uses_typed_newton_selection_and_newton_solver_field():
    command = training_command(_spec(Backend.NEWTON, num_envs=1024))
    joined = " ".join(command)

    assert "physics=newton_sdf" in command
    assert "env.sim.physics.solver_cfg.iterations=64" in command
    assert "solver_position_iteration_count" not in joined


def test_curriculum_commands_distinguish_annealing_from_disabled_reset_mode():
    enabled = training_command(_spec(num_envs=1024, curriculum_enabled=True))
    disabled = training_command(_spec(num_envs=1024, curriculum_enabled=False))

    assert "env.events.reset_plug_curriculum.params.anneal_end_iter=2000" in enabled
    assert "env.events.reset_plug_curriculum.params.num_steps_per_env=128" in enabled
    assert "env.events.reset_plug_curriculum.params.at_goal_prob=0.0" in disabled
    assert not any("anneal_end_iter" in value for value in disabled)


def test_training_log_health_accepts_finite_learning_log():
    log = """
Learning iteration: 1
Perf/total_fps: 3900
Metrics/success_rate: 0.40
Metrics/terminal_success_rate: 0.35
Loss/value_function: 0.4
Learning iteration: 2
Perf/total_fps: 4100
Metrics/success_rate: 0.70
Metrics/terminal_success_rate: 0.65
Loss/value_function: 0.2
"""
    report = analyze_training_log(
        log,
        thresholds=TrainingHealthThresholds(
            minimum_iterations=2,
            minimum_final_terminal_success=0.6,
            minimum_tail_terminal_success=0.5,
            minimum_tail_median_fps=3900,
        ),
    )

    assert report.passed
    assert report.metrics["total_fps"] == [3900.0, 4100.0]
    assert report.to_dict()["summary"]["terminal_success_rate"]["tail_median"] == pytest.approx(
        0.5)


def test_training_log_health_parses_real_rsl_rl_console_labels_without_metric_overlap():
    log = """
Learning iteration 7/1000
Steps per second: 3819
Metrics/success_rate: 0.40
Metrics/terminal_success_rate: 0.65
Learning iteration: 8
Steps per second: 4000
"""

    report = analyze_training_log(log)

    assert report.passed
    assert report.metrics["iteration"] == [7.0, 8.0]
    assert report.metrics["total_fps"] == [3819.0, 4000.0]
    assert report.metrics["success_rate"] == [0.4]
    assert report.metrics["terminal_success_rate"] == [0.65]


def test_training_log_health_serializes_non_finite_metrics_as_json_null(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    log_path = tmp_path / "non_finite.log"
    log_path.write_text(
        "Learning iteration 1/10\nSteps per second: 1000\nLoss/value_function: nan\n")

    result = main(["analyze_log", str(log_path)])
    output = json.loads(capsys.readouterr().out)

    assert result == 1
    assert output["metrics"]["value_loss"] == [None]
    assert {finding["code"]
            for finding in output["findings"]} >= {"non_finite", "non_finite_metric"}


@pytest.mark.parametrize(
    ("bad_line", "expected_code"),
    [
        ("Traceback (most recent call last):", "python_traceback"),
        ("CUDA error: invalid argument", "cuda_error"),
        ("PhysX contact buffer overflow", "contact_capacity"),
        ("Newton dropped contacts: 4", "contact_capacity"),
        ("Loss/value_function: nan", "non_finite"),
    ],
)
def test_training_log_health_rejects_runtime_failures(bad_line: str, expected_code: str):
    log = f"Learning iteration: 1\nPerf/total_fps: 1000\n{bad_line}\n"

    report = analyze_training_log(log)

    assert not report.passed
    assert expected_code in {finding.code for finding in report.findings}


def test_training_log_health_does_not_reject_zero_capacity_telemetry():
    log = "Learning iteration: 1\nPerf/total_fps: 1000\ncontact overflow count: 0\n"

    report = analyze_training_log(log)

    assert report.passed


def test_training_log_health_compares_tail_throughput_against_baseline():
    candidate = "Learning iteration: 1\nPerf/total_fps: 1800\n"
    baseline = "Learning iteration: 1\nPerf/total_fps: 2000\n"

    passed = analyze_training_log(
        candidate,
        thresholds=TrainingHealthThresholds(minimum_fps_ratio_to_baseline=0.85),
        baseline_text=baseline,
    )
    failed = analyze_training_log(
        candidate,
        thresholds=TrainingHealthThresholds(minimum_fps_ratio_to_baseline=0.95),
        baseline_text=baseline,
    )

    assert passed.passed
    assert {finding.code for finding in failed.findings} == {"fps_regression"}


def test_training_log_health_rejects_non_finite_baseline_throughput():
    candidate = "Learning iteration 1/10\nSteps per second: 1800\n"
    baseline = "Learning iteration 1/10\nSteps per second: nan\n"

    report = analyze_training_log(
        candidate,
        thresholds=TrainingHealthThresholds(minimum_fps_ratio_to_baseline=0.9),
        baseline_text=baseline,
    )

    assert not report.passed
    assert {finding.code for finding in report.findings} == {"invalid_baseline_fps"}


@pytest.mark.parametrize(
    "thresholds",
    [
        TrainingHealthThresholds(tail_window=0),
        TrainingHealthThresholds(minimum_iterations=0),
        TrainingHealthThresholds(minimum_final_terminal_success=1.1),
        TrainingHealthThresholds(minimum_tail_median_fps=float("nan")),
    ],
)
def test_training_log_health_rejects_invalid_thresholds(thresholds: TrainingHealthThresholds):
    with pytest.raises(ValueError):
        analyze_training_log("", thresholds=thresholds)


def test_evidence_template_is_complete_but_cannot_accidentally_pass():
    template = validation_evidence_template()
    expected_metrics = {requirement.name for requirement in VALIDATION_EVIDENCE_REQUIREMENTS}
    expected_metrics.update(requirement.at_least_metric
                            for requirement in VALIDATION_EVIDENCE_REQUIREMENTS
                            if requirement.at_least_metric is not None)

    assert set(template["metrics"]) == expected_metrics
    assert validate_evidence_document(template)


@pytest.mark.parametrize(
    ("document", "expected_code"),
    [
        ([], "document_type"),
        ({
            "schema_version": True
        }, "schema_version"),
    ],
)
def test_evidence_validation_rejects_malformed_top_level_documents(document: object,
                                                                   expected_code: str):
    violations = validate_evidence_document(document)

    assert expected_code in {item.code for item in violations}


def test_measured_gpu_evidence_passes_all_contracts():
    assert validate_evidence_document(_passing_evidence()) == ()


def test_evidence_validation_checks_artifacts_under_explicit_root(tmp_path: Path):
    document = _passing_evidence()
    document["artifacts"] = {
        "logs": ["training.log"],
        "videos": ["captures/scripted_insertion.mp4"],
        "tables": ["physics_metrics.json"],
    }
    (tmp_path / "captures").mkdir()
    for reference in ("training.log", "captures/scripted_insertion.mp4", "physics_metrics.json"):
        artifact = tmp_path / reference
        artifact.write_bytes(b"evidence")

    assert validate_evidence_document(document, artifact_root=tmp_path) == ()


@pytest.mark.parametrize(
    ("reference", "expected_code"),
    [
        ("missing.log", "artifact_missing"),
        ("empty.log", "artifact_empty"),
        ("directory", "artifact_not_file"),
        ("../outside.log", "artifact_path"),
    ],
)
def test_evidence_validation_rejects_invalid_artifact_files(
    tmp_path: Path,
    reference: str,
    expected_code: str,
):
    artifact_root = tmp_path / "evidence"
    artifact_root.mkdir()
    (artifact_root / "empty.log").touch()
    (artifact_root / "directory").mkdir()
    (tmp_path / "outside.log").write_bytes(b"outside")
    document = _passing_evidence()
    document["artifacts"]["logs"] = [reference]

    violations = validate_evidence_document(document, artifact_root=artifact_root)

    assert expected_code in {item.code for item in violations}


def test_validate_evidence_cli_checks_paths_relative_to_evidence_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    document = _passing_evidence()
    document["artifacts"] = {
        "logs": ["training.log"],
        "videos": ["scripted_insertion.mp4"],
        "tables": ["physics_metrics.json"],
    }
    for reference in ("training.log", "scripted_insertion.mp4", "physics_metrics.json"):
        (tmp_path / reference).write_bytes(b"evidence")
    evidence_path = tmp_path / "displayport_validation.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")

    result = main(["validate_evidence", str(evidence_path)])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["passed"] is True
    (tmp_path / "training.log").unlink()

    result = main(["validate_evidence", str(evidence_path)])
    output = json.loads(capsys.readouterr().out)

    assert result == 1
    assert "artifact_missing" in {item["code"] for item in output["violations"]}


def test_evidence_validation_rejects_wrong_types_negative_values_and_missing_artifacts():
    document = _passing_evidence()
    document["metrics"]["scene.load_exit_code"] = False
    document["metrics"]["properties.mass_matches"] = 1
    document["metrics"]["curriculum.max_contact_force_n"] = -1.0
    document["artifacts"]["videos"] = []

    violations = validate_evidence_document(document)
    codes = [item.code for item in violations]

    assert codes.count("metric_type") == 2
    assert codes.count("metric_minimum") == 1
    assert codes.count("artifacts_missing") == 1


def test_evidence_validation_requires_typed_baseline_and_candidate_provenance():
    document = _passing_evidence()
    document["metadata"]["baseline_git_commit"] = 123
    document["metadata"]["candidate_label"] = 456

    violations = validate_evidence_document(document)
    codes = [item.code for item in violations]

    assert codes.count("metadata_missing") == 2
    assert codes.count("metadata_value") == 1

    document = _passing_evidence()
    document["metadata"]["backend"] = []
    assert "metadata_value" in {item.code for item in validate_evidence_document(document)}


def test_evidence_validation_detects_physics_and_deployment_regressions():
    document = _passing_evidence()
    document["metrics"]["curriculum.max_contact_force_n"] = 10.1
    document["metrics"]["scripted_insertion.candidate_seated_count"] = 6
    document["metrics"]["deployment.max_action_parity_error"] = 2.0e-5

    violations = validate_evidence_document(document)

    assert [item.code for item in violations].count("metric_maximum") == 2
    assert [item.code for item in violations].count("metric_regression") == 1


def test_matrix_cli_returns_blocking_status_and_machine_readable_json(
        capsys: pytest.CaptureFixture[str]):
    result = main([
        "matrix",
        "--backend",
        "newton",
        "--asset",
        "nominal",
        "--curriculum",
        "on",
        "--profile",
        "e4096_s64_mb16",
        "--format",
        "json",
    ])
    output = json.loads(capsys.readouterr().out)

    assert result == 2
    assert output[0]["runnable"] is False
    assert output[0]["violations"][0]["code"] == "environment_count_not_validated"


def test_matrix_cli_generates_safe_bounded_newton_diagnostic(capsys: pytest.CaptureFixture[str]):
    result = main([
        "matrix",
        "--backend",
        "newton",
        "--asset",
        "nominal",
        "--curriculum",
        "on",
        "--profile",
        "e256_s100_mb16",
        "--max_iterations",
        "10",
        "--format",
        "json",
    ])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output[0]["runnable"] is True
    assert output[0]["schedule"]["max_iterations"] == 10
    assert "_i10_" in output[0]["name"]
    assert "--max_iterations" in output[0]["command"]
    assert "10" in output[0]["command"]


def test_sample_budget_cli_prints_exact_conversion(capsys: pytest.CaptureFixture[str]):
    result = main([
        "sample_budget",
        "--reference_envs",
        "1024",
        "--reference_steps",
        "512",
        "--reference_iterations",
        "500",
        "--candidate_envs",
        "1024",
        "--candidate_steps",
        "128",
    ])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["candidate_iterations"] == 2000
    assert output["candidate_samples"] == output["reference_samples"]
