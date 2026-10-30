# Training development branch

This development snapshot imports `newton` from [IsaacLabTraining](https://github.com/ashwinvkNV/IsaacLabTraining) at commit `7d38c1eec99d3fca0db9a0228bfe43a5e39fa97c`.

Monorepo base: `338987c25855899b096a706d123b07516357b287` (`ashwinvk/isaaclab_training_code`). The imported baseline package matches upstream `main` at `c9fc28c7d1a610b002a80816732eb038c2353bb3`.

The original project layout is restored here: Python modules live in `src/isaaclab_training/`, alongside the branch-specific `pyproject.toml`, `uv.lock`, tests, documentation, and assets. Run the training setup commands from this directory. The dependency files retain their upstream contents; the lockfile has not been regenerated.

Existing monorepo copyright and Apache-2.0 notices are retained. Upstream notices on additional files are retained, with the upstream BSD-3-Clause license in `LICENSE.upstream`. `LICENSE` retains the monorepo Apache-2.0 license.

The nested GitHub Pages workflow and pre-commit configuration are upstream reference files. GitLab uses the monorepo CI configuration and Python style checks.

This is an experimental development branch. Import and packaging checks do not establish training success or real-robot validation.
