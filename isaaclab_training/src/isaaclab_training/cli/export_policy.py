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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Export RSL-RL policies with LEAPP using packaged Isaac Lab APIs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import contextlib
import importlib.metadata as metadata
import os
import random
import sys
from typing import Any

from isaaclab_training.export.contracts import get_export_contract

RSL_RL_MIN_VERSION = "5.0.1"


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    from isaaclab.app import AppLauncher
    from isaaclab_tasks.utils import setup_preset_cli
    # Preset argument discovery queries Gym's registry, including for ``--help``.
    import isaaclab_training.tasks  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Export an Isaac Lab policy with RSL-RL and LEAPP.")
    parser.add_argument("--task", type=str, default=None, help="Name of the registered task.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path, checkpoint directory, 'pretrained', "
            "or omitted for automatic local discovery."
        ),
    )
    parser.add_argument("--export_task_name",
                        type=str,
                        default=None,
                        help="Name of the exported graph.")
    parser.add_argument(
        "--export_method",
        type=str,
        default=None,
        choices=["onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace", "pt2"],
        help="Export backend. Defaults to onnx-dynamo.",
    )
    parser.add_argument("--export_save_path",
                        type=str,
                        default=None,
                        help="Directory for exported artifacts.")
    parser.add_argument("--validation_steps",
                        type=int,
                        default=5,
                        help="Number of export validation steps.")
    parser.add_argument("--disable_graph_visualization", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment folder used to locate checkpoints.",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="Optional named export contract, e.g. 'displayport_task_space'.",
    )
    parser.add_argument(
        "--task_space_contract",
        action="store_true",
        default=False,
        help=(
            "Compatibility shortcut for '--contract displayport_task_space'. Prefer --contract "
            "for new tasks."
         ),
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--limit_cpu_threads",
                        type=int,
                        default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)

    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.headless = True
    if args_cli.task_space_contract:
        args_cli.contract = "displayport_task_space"
    return args_cli, hydra_args


def disable_torchscript_for_export() -> None:
    """Disable TorchScript so LEAPP traces Python math helpers rather than folded constants."""
    import torch

    torch.jit._state.disable()


def create_graph_configs(env_cfg: Any):
    """Create LEAPP graph metadata from an Isaac Lab environment configuration."""
    from leapp import GraphConfigs

    policy_frequency = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    return GraphConfigs(frequency=policy_frequency)


def get_actor_memory_module(policy):
    """Return the actor-side RNN module for supported RSL-RL recurrent policies."""
    if hasattr(policy, "rnn"):
        return policy.rnn
    return None


def is_actor_recurrent_policy(policy) -> bool:
    """Return whether the actor policy has a supported recurrent state container."""
    return bool(
        getattr(policy, "is_recurrent", False) and get_actor_memory_module(policy) is not None)


def get_actor_hidden_state(policy):
    """Return the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    if hasattr(policy, "get_hidden_state"):
        return policy.get_hidden_state()
    memory = get_actor_memory_module(policy)
    return None if memory is None else getattr(memory, "hidden_state", None)


def set_actor_hidden_state(policy, actor_hidden) -> None:
    """Assign the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    memory = get_actor_memory_module(policy)
    if memory is not None:
        memory.hidden_state = actor_hidden


def ensure_actor_hidden_state_initialized(policy, batch_size: int, device, dtype):
    """Initialize and return the actor hidden state when needed."""
    import torch

    actor_state = get_actor_hidden_state(policy)
    if actor_state is not None:
        return actor_state

    memory = get_actor_memory_module(policy)
    if memory is None or not hasattr(memory, "rnn"):
        return None

    zeros = torch.zeros(memory.rnn.num_layers,
                        batch_size,
                        memory.rnn.hidden_size,
                        device=device,
                        dtype=dtype)
    actor_state = (zeros.clone(),
                   zeros.clone()) if isinstance(memory.rnn, torch.nn.LSTM) else zeros
    set_actor_hidden_state(policy, actor_state)
    return actor_state


def state_dict_from_actor_hidden(actor_hidden):
    """Convert actor hidden state into LEAPP named tensors."""
    if actor_hidden is None:
        return {}
    if isinstance(actor_hidden, tuple):
        return {f"actor_state_{idx}": tensor for idx, tensor in enumerate(actor_hidden)}
    return {"actor_state": actor_hidden}


def actor_hidden_from_registered(registered_state, original_hidden):
    """Restore registered LEAPP state to the structure expected by the actor."""
    if isinstance(original_hidden, tuple):
        if isinstance(registered_state, tuple):
            return registered_state
        return (registered_state, )
    return registered_state


def _update_agent_cfg_from_export_args(agent_cfg, args_cli: argparse.Namespace):
    """Apply export-relevant CLI overrides to the RSL-RL agent config."""
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    return agent_cfg


def export_rsl_rl_agent(args_cli: argparse.Namespace, env_cfg, agent_cfg) -> bool:
    """Export an RSL-RL agent with an optional task-specific I/O contract."""
    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.leapp import patch_env_for_export
    from isaaclab.utils.leapp.utils import ensure_env_spec_id
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import (get_pretrained_checkpoint_backend_names,
                                                         get_published_pretrained_checkpoint)
    from isaaclab_tasks.utils import get_checkpoint_path
    import leapp
    from packaging import version
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    import torch

    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_MIN_VERSION):
        print(
            f"[WARNING] LEAPP RSL-RL export is validated with rsl-rl-lib {RSL_RL_MIN_VERSION} "
            f"or newer. "
            f"Installed version is '{installed_version}'.")

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")

    agent_cfg = _update_agent_cfg_from_export_args(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.checkpoint == "pretrained":
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        resume_path = get_published_pretrained_checkpoint("rsl_rl", checkpoint_task_name,
                                                          *backend_names)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint is currently available for this task.")
            return False
    elif args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        resume_path = get_checkpoint_path(os.path.dirname(args_cli.checkpoint),
                                          os.path.basename(args_cli.checkpoint),
                                          agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run,
                                          agent_cfg.load_checkpoint)

    if not resume_path:
        print(
            f"[INFO] No checkpoint found for task: {checkpoint_task_name} in directory: "
            f"{log_root_path}"
        )
        return False

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False
    try:
        contract = get_export_contract(args_cli.contract)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = ensure_env_spec_id(env)
        graph_name = (
            args_cli.export_task_name
            if args_cli.export_task_name is not None else task_name
        )
        if contract is not None and args_cli.export_task_name is None:
            graph_name = contract.graph_name
        if contract is not None:
            policy_node_name = graph_name

        export_method = "onnx-dynamo" if args_cli.export_method is None else args_cli.export_method

        if isinstance(env.unwrapped, ManagerBasedRLEnv):
            obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
            if isinstance(obs_groups_cfg, Mapping):
                required_obs_groups = set(obs_groups_cfg.get("actor", ["policy"]))
            else:
                required_obs_groups = {"policy"}
            if contract is None or not contract.skip_automatic_env_patch:
                patch_env_for_export(env,
                                     export_method=export_method,
                                     required_obs_groups=required_obs_groups)
        elif args_cli.export_method is not None:
            raise ValueError("--export_method is only supported for manager-based environments.")

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env,
                                    agent_cfg.to_dict(),
                                    log_dir=None,
                                    device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env,
                                        agent_cfg.to_dict(),
                                        log_dir=None,
                                        device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.checkpoint == "pretrained":
            save_path = os.path.join(".pretrained_checkpoints", "rsl_rl", checkpoint_task_name)
        else:
            save_path = log_dir

        leapp.start(graph_name,
                    save_path=save_path,
                    max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True
        obs = env.reset()[0]

        for _ in range(max(args_cli.validation_steps, 2)):
            with torch.inference_mode():
                if contract is not None:
                    obs_for_policy = contract.prepare_observations(
                        graph_name,
                        obs,
                        env_cfg=env.unwrapped.cfg,
                        dtype=next(policy.parameters()).dtype,
                    )
                else:
                    obs_for_policy = obs

                if is_actor_recurrent_policy(policy):
                    actor_hidden = ensure_actor_hidden_state_initialized(
                        policy,
                        batch_size=env.num_envs,
                        device=env.unwrapped.device,
                        dtype=next(policy.parameters()).dtype,
                    )
                    registered_state = leapp.annotate.state_tensors(
                        policy_node_name, state_dict_from_actor_hidden(actor_hidden))
                    set_actor_hidden_state(
                        policy, actor_hidden_from_registered(registered_state, actor_hidden))

                actions = policy(obs_for_policy)

                if is_actor_recurrent_policy(policy):
                    actor_hidden_after = get_actor_hidden_state(policy)
                    leapp.annotate.update_state(policy_node_name,
                                                state_dict_from_actor_hidden(actor_hidden_after))

                if contract is not None:
                    contract.export_action(
                        graph_name,
                        actions,
                        env.unwrapped.cfg,
                        env.unwrapped.device,
                        next(policy.parameters()).dtype,
                        export_method,
                        agent_cfg.clip_actions,
                    )
                    obs = env.get_observations()
                else:
                    obs, _, _, _ = env.step(actions)

        leapp.stop()
        leapp_started = False
        leapp.compile_graph(
            visualize=not args_cli.disable_graph_visualization,
            validate=args_cli.validation_steps > 0,
            graph_configs=create_graph_configs(env_cfg),
        )
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli: argparse.Namespace, hydra_args: list[str]) -> bool:
    """Resolve Hydra task configuration and export one RSL-RL policy."""
    disable_torchscript_for_export()

    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import isaaclab_training.tasks  # noqa: F401

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False
    try:

        @hydra_task_config(args_cli.task, args_cli.agent)
        def _main(env_cfg, agent_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_rsl_rl_agent(args_cli, env_cfg, agent_cfg)

        _main()
    finally:
        sys.argv = original_argv

    return exported


def main(argv: list[str] | None = None) -> bool:
    """Run the export command."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export_with_hydra(args_cli, hydra_args)


if __name__ == "__main__":
    main()
