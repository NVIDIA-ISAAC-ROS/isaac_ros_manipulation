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
"""Task registrations exposed to the Isaac Lab CLI."""

import os

# Let the PyTorch CUDA caching allocator grow segments in place instead of keeping one cached block
# per
# size. The recurrent PPO update allocates differently sized buffers every minibatch, which
# otherwise
# leaves several GiB reserved but unused. Allocator behaviour only; computation is unchanged. Must
# be set
# before the first CUDA allocation, which is why it lives in the module the CLI imports to register
# tasks.
# An explicit PYTORCH_CUDA_ALLOC_CONF set by the user takes precedence.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab_tasks.utils import import_packages  # noqa: E402

import_packages(__name__, ["isaaclab_training.mdp"])


def register() -> None:
    """Register the tasks in every process of a multi-GPU run.

    Pass as ``--external_callback isaaclab_training.tasks.register``. ``isaaclab train`` loads
    this package through its entry point, but ``isaaclab train_multigpu`` workers do not.
    Importing this module registers the tasks, so the callback has nothing left to do.
    """
