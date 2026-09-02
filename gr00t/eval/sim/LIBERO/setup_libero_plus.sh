#!/usr/bin/env bash
set -euxo pipefail

# Setup the LIBERO-plus fork (sylvestf/LIBERO-plus) in a dedicated uv venv.
#
# The fork is a drop-in replacement for the vanilla LIBERO package
# (same `import libero` name), so it gets its own venv next to libero_uv.
# The vanilla LIBERO env (libero_uv) is left completely untouched.
#
# Prerequisites:
#   - git clone https://github.com/sylvestf/LIBERO-plus external_dependencies/LIBERO-plus
#   - assets.zip from HF Sylvest/LIBERO-plus extracted into
#     external_dependencies/LIBERO-plus/libero/libero/assets/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_PLUS_REPO="$SCRIPT_DIR/../../../../external_dependencies/LIBERO-plus"
LIBERO_PLUS_UV_ENV="$SCRIPT_DIR/libero_plus_uv"
# Dedicated LIBERO config dir so the fork never reads ~/.libero/config.yaml
# written by the vanilla LIBERO install.
LIBERO_PLUS_CONFIG_DIR="$SCRIPT_DIR/libero_plus_config"

if [[ ! -d "$LIBERO_PLUS_REPO/libero" ]]; then
    echo "LIBERO-plus repo not found at $LIBERO_PLUS_REPO" >&2
    exit 1
fi

if [[ ! -d "$LIBERO_PLUS_REPO/libero/libero/assets" ]]; then
    echo "assets/ missing under $LIBERO_PLUS_REPO/libero/libero/" >&2
    echo "Download assets.zip from HF Sylvest/LIBERO-plus and extract it first." >&2
    exit 1
fi

rm -rf "$LIBERO_PLUS_UV_ENV"
mkdir -p "$LIBERO_PLUS_UV_ENV"
uv venv "$LIBERO_PLUS_UV_ENV/.venv" --python 3.10
source "$LIBERO_PLUS_UV_ENV/.venv/bin/activate"

# Fork requirements + extra requirements (robosuite 1.4.0, bddl, usd-core, wand, scikit-image, ...)
uv pip install --requirements "$LIBERO_PLUS_REPO/requirements.txt"
uv pip install --requirements "$LIBERO_PLUS_REPO/extra_requirements.txt"
uv pip install -e "$LIBERO_PLUS_REPO" --config-settings editable_mode=compat

# robosuite 1.4.0 reads sim.data.qM which was removed from newer mujoco
# bindings; pin to the same version the working vanilla libero_uv env uses.
uv pip install mujoco==3.8.1

# Isaac-GR00T package (editable, no deps) so gr00t.eval.sim.LIBERO.libero_env imports work.
uv pip install --editable "$PROJECT_REPO" --no-deps

# Same runtime stack as libero_uv (mirrors setup_libero.sh final state).
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install numpy==1.26.4

uv pip install --editable "$PROJECT_REPO" --no-deps

# Point the fork at its own config dir (auto-creates config.yaml on first import).
rm -rf "$LIBERO_PLUS_CONFIG_DIR"
mkdir -p "$LIBERO_PLUS_CONFIG_DIR"

export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Pre-create config.yaml so the fork's interactive first-import prompt
# ("Do you want to specify a custom path for the dataset folder?") is skipped
# and default paths (bddl/init_states/assets inside the fork) are used.
python - <<'PY'
import os
import yaml

fork_libero = os.path.abspath(
    os.path.join(os.environ["LIBERO_PLUS_REPO"], "libero", "libero")
)
config_file = os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml")
config = {
    "benchmark_root": fork_libero,
    "bddl_files": os.path.join(fork_libero, "bddl_files"),
    "init_states": os.path.join(fork_libero, "init_files"),
    "datasets": os.path.join(fork_libero, "..", "datasets"),
    "assets": os.path.join(fork_libero, "assets"),
}
with open(config_file, "w") as f:
    yaml.dump(config, f)
print("wrote", config_file)
PY

python - <<'PY'
import os
from libero.libero import get_libero_path
print("config:", os.environ["LIBERO_CONFIG_PATH"])
for key in ["bddl_files", "init_states", "datasets", "assets"]:
    print(key, "->", get_libero_path(key))
PY
