# DisplayPort replacement collision parts

Precomputed collision geometry used by the task-space DisplayPort configs
(`config/displayport_rizon_4s/task_space_env_cfg.py`) through
`isaaclab_training.utils.spawners_cfg.UsdFileWithMassOverrideCfg`. Vertices are in the asset's
rigid-body frame [m]; each file holds `names` and, per name, `<name>_v`, `<name>_f`, `<name>_kind`.

| File | Part | Kind | Replaces | Notes |
|---|---|---|---|---|
| `plug_connector_sdf_casing_hull.npz` | `connector_sdf` | SDF (resolution 256, settings copied from `collision_mesh`) | `/plug/collision_mesh` | Connector that mates with the socket: the original mesh below a plane 1.7 mm above the socket mouth at the physical seat, capped watertight (17.5k of 40.2k triangles). |
| | `casing_hull` | convex hull, 64 vertices | | Casing the gripper holds (convex in the asset): original vertices above +1.8 mm (casing bottom face at +1.85 mm). Surface deviation p50 0.02 mm, p99 0.11 mm, max 0.16 mm. |
| `socket_body8_top_sdf_lower_hull.npz` | `housing_top_sdf` | SDF (resolution 256, settings copied from Body8) | `Body8/Mesh` | Top 12.5 mm of the housing (flange, top face, upper cavity walls; deepest plug tip reaches 9 mm below the mouth), capped watertight (5.3k of 43.4k triangles). |
| | `housing_lower_hull` | convex hull, 64 vertices | | Lower housing (rear block, cable tube), never contacted by the plug. |

**Defaults:** the socket Body8 split is enabled. The plug split is opt-in (`_USE_PLUG_CASING_HULL` in
`task_space_env_cfg.py`): it gives +21-30 % simulation throughput, but a 64-vertex hull cannot follow the curved
casing closer than ~0.1-0.16 mm, which shifts the plug ~0.2 mm / ~0.7 deg in the gripper and jammed 2 of 8
tight-offset scripted insertions (+0.25 / +0.5 mm on the 0.08 mm-clearance axis) that the SDF casing seats.
Splitting the casing into 2-4 hulls or a dedicated grip-band hull did not improve the fit below ~0.1 mm.

The plug's mass properties are pinned in the config to the values PhysX derives from the original asset
shapes, so replacing its colliders does not change its dynamics; the socket is kinematic.

Regenerate with `tools/generate_dp_collision_parts.py` (needs `trimesh`, `shapely>=2.1`) if the assets change.
