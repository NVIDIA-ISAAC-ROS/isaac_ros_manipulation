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

"""Generate the DisplayPort replacement collision parts.

See src/.../displayport_insertion/collision/README.md.

Usage:
    python tools/generate_dp_collision_parts.py <displayport_plug.usd> \\
        <displayport_socket_no_protrusions.usd>

Needs ``pxr`` (usd-core or Isaac Sim), ``trimesh`` and ``shapely>=2.1`` (constrained Delaunay for
the caps).
"""

from collections import Counter
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom
import shapely
from shapely.geometry import MultiPolygon, Polygon
import trimesh

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "isaaclab_training", "tasks",
                   "displayport_insertion", "collision")

# plug frame: insertion axis is -z; the socket mouth sits at z = 23.94 mm when the plug rests on
# its physical seat
PLUG_MOUTH_Z = 0.02394
PLUG_CONNECTOR_TOP = 0.0017  # connector SDF below 1.7 mm above the mouth
# Casing hull from 1.8 mm above the mouth (casing bottom face at 1.85 mm).
PLUG_CASING_BOTTOM = 0.0018
SOCKET_BODY8_CUT_X = 0.025  # socket frame: keep Body8 SDF above X = 25 mm (mouth at 37.5 mm)
HULL_VERTEX_LIMIT = 64  # PhysX GPU convex hull limit


def load_mesh(usd_path: str, root: str, prim_path: str) -> trimesh.Trimesh:
    """Load a mesh prim in the frame of ``root`` (the asset's rigid-body prim)."""
    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()
    root_inv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root)).GetInverse()
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    faces, o = [], 0
    for c in counts:
        faces += [(idx[o], idx[o + k], idx[o + k + 1]) for k in range(1, c - 1)]
        o += c
    m = np.array(cache.GetLocalToWorldTransform(prim) * root_inv).T
    return trimesh.Trimesh((np.c_[pts, np.ones(len(pts))] @ m.T)[:, :3],
                           np.array(faces),
                           process=False)


def capped_slice(mesh: trimesh.Trimesh, origin, normal) -> trimesh.Trimesh:
    """Keep the side ``normal`` points to and cap its boundary loops to make it watertight."""
    origin, normal = np.asarray(origin, float), np.asarray(normal, float) / np.linalg.norm(normal)
    part = trimesh.intersections.slice_mesh_plane(mesh,
                                                  plane_normal=normal,
                                                  plane_origin=origin,
                                                  cap=False)
    part.merge_vertices(digits_vertex=10)
    boundary = [e for e, c in Counter(map(tuple, np.sort(part.edges, axis=1))).items() if c == 1]
    adj = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while nxt := [n for n in adj[cur] if n != prev and n not in seen]:
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uv = np.c_[(part.vertices - origin) @ u, (part.vertices - origin) @ v]
    rings = sorted((Polygon(uv[lp]) for lp in loops if len(lp) >= 3), key=lambda p: -abs(p.area))
    outers = []
    for r in rings:
        host = next((o for o in outers if o["shell"].contains(r.representative_point())), None)
        if host is None:
            outers.append({"shell": r, "holes": []})
        else:
            host["holes"].append(r)
    polys = [
        Polygon(o["shell"].exterior.coords, [h.exterior.coords for h in o["holes"]])
        for o in outers
    ]
    geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    lookup = {(round(uv[i, 0], 12), round(uv[i, 1], 12)): i for lp in loops for i in lp}
    cap = []
    for t in shapely.get_parts(shapely.constrained_delaunay_triangles(geom)):
        ids = [
            lookup.get((round(x, 12), round(y, 12))) for x, y in np.array(t.exterior.coords)[:3]
        ]
        if None not in ids:
            cap.append(ids)
    out = trimesh.Trimesh(part.vertices.copy(),
                          np.vstack([part.faces, np.array(cap)]),
                          process=False)
    trimesh.repair.fix_normals(out)
    if not out.is_watertight:
        raise RuntimeError("capped slice is not watertight")
    return out


def reduced_hull(points: np.ndarray, max_vertices: int = HULL_VERTEX_LIMIT) -> trimesh.Trimesh:
    """Convex hull of at most ``max_vertices`` input points, chosen greedily to minimise the
    outside distance."""
    cand = trimesh.convex.convex_hull(points).vertices
    c = cand - cand.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    idx = sorted({int(i) for ax in vt for i in ((c @ ax).argmax(), (c @ ax).argmin())})
    while len(idx) < max_vertices:
        hull = trimesh.convex.convex_hull(cand[idx])
        out = -trimesh.proximity.signed_distance(hull, cand)
        out[idx] = -np.inf
        j = int(out.argmax())
        if out[j] <= 1e-7:
            break
        idx.append(j)
    return trimesh.convex.convex_hull(cand[idx])


def save(path, parts):
    data = {"names": np.array([n for n, _, _ in parts])}
    for name, mesh, kind in parts:
        data[f"{name}_v"] = mesh.vertices.astype(np.float32)
        data[f"{name}_f"] = mesh.faces.astype(np.int32)
        data[f"{name}_kind"] = np.array(kind)
    np.savez_compressed(path, **data)
    print("wrote", path, [(n, len(m.vertices), len(m.faces), k) for n, m, k in parts])


def main(plug_usd: str, socket_usd: str):
    plug = load_mesh(plug_usd, "/plug", "/plug/collision_mesh")
    connector = capped_slice(plug, [0, 0, PLUG_MOUTH_Z - PLUG_CONNECTOR_TOP], [0, 0, 1])
    height = PLUG_MOUTH_Z - plug.vertices[:, 2]
    casing = reduced_hull(plug.vertices[height > PLUG_CASING_BOTTOM])
    save(
        os.path.join(OUT, "plug_connector_sdf_casing_hull.npz"),
        [("connector_sdf", connector, "sdf"), ("casing_hull", casing, "convexHull")],
    )

    body8 = load_mesh(socket_usd, "/socket", "/socket/tn__2584N111_DisplayportCord_jP/Body8/Mesh")
    top = capped_slice(body8, [SOCKET_BODY8_CUT_X, 0, 0], [1, 0, 0])
    lower_pts = np.vstack([
        body8.vertices[body8.vertices[:, 0] <= SOCKET_BODY8_CUT_X + 1e-4],
        top.vertices[np.abs(top.vertices[:, 0] - SOCKET_BODY8_CUT_X) < 1e-7],
    ])
    save(
        os.path.join(OUT, "socket_body8_top_sdf_lower_hull.npz"),
        [("housing_top_sdf", top, "sdf"),
         ("housing_lower_hull", reduced_hull(lower_pts), "convexHull")],
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
