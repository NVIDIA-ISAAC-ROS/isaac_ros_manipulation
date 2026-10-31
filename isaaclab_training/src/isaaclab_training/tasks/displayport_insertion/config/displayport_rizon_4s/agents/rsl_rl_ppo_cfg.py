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

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

_BASE_NUM_STEPS_PER_ENV = 512
_BASE_MAX_ITERATIONS = 1500
_BASE_SAVE_INTERVAL = 50
TASK_SPACE_NUM_STEPS_PER_ENV = 128
TASK_SPACE_MAX_ITERATIONS = 1000
NEWTON_MAX_ITERATIONS = 1500


@configclass
class Rizon4sGravDisplayportInsertionRNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _BASE_NUM_STEPS_PER_ENV
    max_iterations = _BASE_MAX_ITERATIONS
    save_interval = _BASE_SAVE_INTERVAL
    experiment_name = "displayport_insertion_rizon4s"
    clip_actions = 1.0
    resume = False
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlRNNModelCfg.HeteroscedasticGaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg(
        Rizon4sGravDisplayportInsertionRNNPPORunnerCfg):
    """Task-space runner with a memory-efficient recurrent rollout."""

    num_steps_per_env = TASK_SPACE_NUM_STEPS_PER_ENV
    max_iterations = TASK_SPACE_MAX_ITERATIONS
    save_interval = _BASE_SAVE_INTERVAL


@configclass
class Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg(
        Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg):
    """RSL-RL runner for the Newton task-space training profile."""

    seed = 123
    max_iterations = NEWTON_MAX_ITERATIONS
    experiment_name = "displayport_insertion_rizon4s_newton_osc"
