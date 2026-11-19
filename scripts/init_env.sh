#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo/.julia-env"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$env_dir/depot:}"

mkdir -p "$env_dir"
cd "$env_dir"
julia --startup-file=no --project=. -e '
    using Pkg
    Pkg.develop(path="..")
    Pkg.instantiate()
'
