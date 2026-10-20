#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
exec "$project_dir/.pixi/envs/default/bin/python" "$project_dir/3d_scene/layered_sphere.py"
