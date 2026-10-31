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

"""Spawner configurations used by the training tasks."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass


@configclass
class UsdFileWithMassOverrideCfg(UsdFileCfg):
    """:class:`UsdFileCfg` with optional mass properties and removal of disabled colliders.

    With the defaults this spawns exactly like :class:`UsdFileCfg`.
    """

    func: Callable | str = "isaaclab_training.utils.spawners:spawn_usd_with_mass_override"

    remove_disabled_colliders: bool = False
    """Strip disabled collision schemas before cloning.

    Applies to prims with ``physics:collisionEnabled = False``.

    The prims stay active, so meshes that double as render geometry remain visible. PhysX includes
    such shapes when it derives the centre of mass and inertia, so on a dynamic body set the
    explicit mass properties below as well to keep its dynamics unchanged.
    """

    center_of_mass: tuple[float, float, float] | None = None
    """Centre of mass in the rigid body's frame [m]. Set together with the two fields below."""

    principal_axes_wxyz: tuple[float, float, float, float] | None = None
    """Orientation (w, x, y, z) of the principal inertia axes relative to the rigid body frame."""

    diagonal_inertia: tuple[float, float, float] | None = None
    """Principal moments of inertia [kg m^2] about :attr:`principal_axes_wxyz`."""

    collision_parts_file: str | None = None
    """``.npz`` with replacement collision parts in the rigid-body frame: ``names`` and, per name,
    ``<name>_v`` (vertices [m]), ``<name>_f`` (triangle indices) and ``<name>_kind`` (``"sdf"`` or
    ``"convexHull"``, at most 64 vertices). Pin the mass properties when replacing a dynamic body's
    colliders."""

    collision_parts_template: str | None = None
    """Collider prim (path relative to the asset root) whose applied schemas, PhysX collision / SDF
    settings and physics material are copied onto the replacement parts."""

    remove_colliders: tuple[str, ...] = ()
    """Collider prims (paths relative to the asset root) replaced by :attr:`collision_parts_file`.

    Their collision schemas are stripped; the prims stay active so render geometry is kept."""
