#!/usr/bin/env bash
# Convert Sylvest/libero_plus_lerobot (LeRobot v2.1) to LeWM H5 + mapping files.
#
# Input : /root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot
# Output: FLARE-VLA/outputs/libero_plus/h5/libero_plus_lerobot__observation_images_{front,wrist}.h5
#         FLARE-VLA/outputs/libero_plus/mapping/*.jsonl|parquet|json
#
# Requires: dataset downloaded via hf download Sylvest/libero_plus_lerobot
#           ffmpeg with libdav1d for AV1 decode.
#
# Usage: bash FLARE-VLA/commands/libero_plus/01_convert.sh

set -euo pipefail

LEWM_PY=/root/autodl-tmp/envs/LeWorldModel/bin/python
SOURCE_DIR=/root/autodl-tmp/Isaac-GR00T/examples/LIBERO/libero_plus_lerobot
OUTPUT_DIR=/root/autodl-tmp/FLARE-VLA/outputs/libero_plus/h5

echo "============================================"
echo "  Gripper convention: raw LIBERO -> LeRobot"
echo "  (in-place parquet action rewrite + stats)"
echo "============================================"
if [[ -f "${SOURCE_DIR}/meta/.gripper_converted" ]]; then
    echo "  [SKIP] marker exists — gripper conversion already done"
else
    "${LEWM_PY}" \
      /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/00b_convert_gripper_convention.py \
      --source-dir "${SOURCE_DIR}"
fi

echo
echo "============================================"
echo "  Regenerating gr00t meta files (post-gripper stats)"
echo "============================================"
"${LEWM_PY}" \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/00_prepare_meta_patch.py \
  --source-dir "${SOURCE_DIR}"

echo
echo "============================================"
echo "  Libero-Plus LeRobot -> LeWM H5 Conversion"
echo "  Source: ${SOURCE_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

# NOTE: front camera only. The wrist H5 has no downstream consumer in this
# pipeline (LeWM training, GT latents and scoring all use the front camera)
# and the H5 files are ~180GB each, so the wrist H5 is not generated.
"${LEWM_PY}" \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/01_convert_libero_plus_to_lewm_h5.py \
  --source-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --camera-keys observation.images.front \
  --image-size 224

echo
echo "============================================"
echo "  Generating mapping files"
echo "============================================"

"${LEWM_PY}" \
  /root/autodl-tmp/FLARE-VLA/scripts/libero_plus/02_generate_mapping.py \
  --h5-path "${OUTPUT_DIR}/libero_plus_lerobot__observation_images_front.h5" \
  --source-dir "${SOURCE_DIR}" \
  --mapping-dir /root/autodl-tmp/FLARE-VLA/outputs/libero_plus/mapping

# LeWM reads <STABLEWM_HOME>/<dataset.name>.h5 (see stable_worldmodel/data/utils.py)
# Only the front camera is used by LeWM training / GT latents / scoring, so
# only the front H5 is copied (the wrist H5 stays in outputs/ as archive).
echo ""
echo "============================================"
echo "  Copying front H5 to LeWM cache dir"
echo "============================================"
cp -v \
  "${OUTPUT_DIR}/libero_plus_lerobot__observation_images_front.h5" \
  /root/autodl-tmp/le-wm/stablewm_data/

echo ""
echo "Done."
