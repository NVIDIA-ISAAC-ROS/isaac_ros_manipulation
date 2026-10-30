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

"""Spawn USD assets with explicit mass properties and configurable colliders.

Supports replacement collision parts and removal of disabled colliders.

Imported lazily through the ``func`` string of
:class:`~isaaclab_training.utils.spawners_cfg.UsdFileWithMassOverrideCfg`, i.e. only once the
simulation app is running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

if TYPE_CHECKING:
    from .spawners_cfg import UsdFileWithMassOverrideCfg

# Collision settings copied from the template collider onto replacement parts.
_COPIED_ATTR_PREFIXES = ("physxCollision:", "physics:collisionEnabled")
_SDF_ATTR_PREFIX = "physxSDFMeshCollision:"


def _strip_collision(prim: Usd.Prim) -> None:
    """Remove every collision schema from ``prim`` so PhysX creates no shape for it.

    The prim stays active, so a mesh that is also render geometry remains visible.
    """
    for schema in prim.GetAppliedSchemas():
        if "Collision" in schema:
            prim.RemoveAppliedSchema(schema)


def _add_collision_part(root: Usd.Prim, name: str, vertices, faces, kind: str,
                        template: Usd.Prim) -> Usd.Prim:
    """Author a mesh collider under ``root`` in the rigid-body frame.

    Collision settings are copied from ``template``.
    """
    stage = root.GetStage()
    mesh = UsdGeom.Mesh.Define(stage, root.GetPath().AppendChild(name))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32).reshape(-1)))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only, never rendered
    prim = mesh.GetPrim()

    # applied schemas: those of the template (keeps unregistered PhysX schemas), adjusted for the
    # part kind
    tmpl_schemas = (list(template.GetMetadata("apiSchemas").GetAddedOrExplicitItems())
                    if template.HasMetadata("apiSchemas") else [])
    schemas = [
        s for s in tmpl_schemas
        if s not in ("PhysxSDFMeshCollisionAPI", "PhysxConvexHullCollisionAPI", "PhysicsMassAPI")
    ]
    for s in ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"):
        if s not in schemas:
            schemas.append(s)
    schemas.append("PhysxSDFMeshCollisionAPI" if kind == "sdf" else "PhysxConvexHullCollisionAPI")
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))

    for attr in template.GetAttributes():
        n = attr.GetName()
        if (n.startswith(_COPIED_ATTR_PREFIXES) or
                (kind == "sdf" and n.startswith(_SDF_ATTR_PREFIX))) and attr.HasAuthoredValue():
            prim.CreateAttribute(n, attr.GetTypeName(), custom=False).Set(attr.Get())
    prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token, custom=False).Set(kind)
    if kind == "convexHull":
        prim.CreateAttribute("physxConvexHullCollision:hullVertexLimit",
                             Sdf.ValueTypeNames.Int,
                             custom=False).Set(64)
    for rel_name in ("material:binding:physics", ):
        rel = template.GetRelationship(rel_name)
        if rel and rel.GetTargets():
            prim.CreateRelationship(rel_name, custom=False).SetTargets(rel.GetTargets())
    return prim


@clone
def spawn_usd_with_mass_override(
    prim_path: str,
    cfg: UsdFileWithMassOverrideCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD asset with optional mass and collider overrides.

    PhysX derives a rigid body's centre of mass and inertia from all of its collision shapes,
    including shapes with ``physics:collisionEnabled = False``. Authoring the centre of mass,
    principal axes and
    principal moments explicitly makes PhysX use them as given, after which collision shapes can be
    replaced or removed without changing the body's dynamics. All steps run on the prototype prim
    before :func:`clone` copies it to the other environments.
    """
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)

    if cfg.center_of_mass is not None:
        if cfg.principal_axes_wxyz is None or cfg.diagonal_inertia is None:
            raise ValueError(
                "center_of_mass, principal_axes_wxyz and diagonal_inertia must be set together.")
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*cfg.center_of_mass))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(*cfg.principal_axes_wxyz))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*cfg.diagonal_inertia))

    if cfg.collision_parts_file is not None:
        template = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath(
            cfg.collision_parts_template))
        if not template.IsValid():
            raise ValueError(
                f"collision_parts_template '{cfg.collision_parts_template}' not found under "
                f"{prim.GetPath()}"
            )
        data = np.load(cfg.collision_parts_file)
        for name in data["names"]:
            name = str(name)
            _add_collision_part(prim, name, data[f"{name}_v"], data[f"{name}_f"],
                                str(data[f"{name}_kind"]), template)
        for rel_path in cfg.remove_colliders:
            target = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath(rel_path))
            if not target.IsValid():
                raise ValueError(
                    f"remove_colliders entry '{rel_path}' not found under {prim.GetPath()}")
            _strip_collision(target)

    if cfg.remove_disabled_colliders:
        disabled = []
        for child in Usd.PrimRange(prim):
            attr = child.GetAttribute("physics:collisionEnabled")
            if attr and attr.HasAuthoredValue() and attr.Get() is False:
                disabled.append(child)
        for child in disabled:
            _strip_collision(child)
    return prim
